import types
from typing import cast, Callable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF

from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.ops.comm import split_for_sequence_parallel
from xtuner.v1.utils import get_logger, get_padding_length, get_device
from xtuner.v1.model import BaseModel

from .modeling_vision import init_world_mesh
from .modeling_projector import VideoChat3MultiModalProjector
from typing_extensions import override
from xtuner.v1.config import FSDPConfig
from .videochat3_config import VideoChat3BaseConfig, VideoChat3LACTVisionConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import CELossContext
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

DEVICE = get_device()
logger = get_logger()


def to_hf_key_list_wrapper(fn: Callable[[str], list[str]], convertor: Callable[[str], str]):
    def wrapper(self, *args, **kwargs):
        return [convertor(i) for i in fn(*args, **kwargs)]
    return wrapper


class VideoChat3ForConditionalGeneration(BaseModel):
    def __init__(self, config: VideoChat3BaseConfig):
        super().__init__()
        self.config = config

        vision_config = config.vision_config
        text_config = config.text_config

        self.vision_tower = vision_config.build()
        self.multi_modal_projector = VideoChat3MultiModalProjector(config.projector_config)
        self.language_model = text_config.build()

        # TODO(YHC): This is a hack to make the language model compatible with HF
        _hf_prefix = "model.language_model."
        self.language_model.to_hf_key_list = types.MethodType(to_hf_key_list_wrapper(  # type: ignore
            fn=self.language_model.to_hf_key_list,
            convertor=lambda x: x.replace('model.', _hf_prefix)),
            self.language_model)
        self.language_model._init_load_spec()

        self.img_context_token_id = config.image_token_id
        self.video_context_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.vision_end_token_id = config.vision_end_token_id
        self._hf_path: Path | None = None

        # Note: global load spec mapping for save_hf
        self.load_spec_mapping = {}
        for key, value in self.vision_tower.load_spec_mapping.items():
            self.load_spec_mapping['vision_tower.' + key] = value
        for key, value in self.multi_modal_projector.load_spec_mapping.items():
            self.load_spec_mapping['multi_modal_projector.' + key] = value
        for key, value in self.language_model.load_spec_mapping.items():
            self.load_spec_mapping['language_model.' + key] = value

        self._freeze_modules()

    def _freeze_modules(self):
        freeze_vision = self.config.freeze_vision
        if freeze_vision:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
            logger.info("Freeze vision tower")
        freeze_projector = self.config.freeze_projector
        if freeze_projector:
            self.multi_modal_projector.requires_grad_(False)
            self.multi_modal_projector.eval()
            logger.info("Freeze multi modal projector")
        freeze_language = self.config.freeze_language
        if freeze_language:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
            logger.info("Freeze language model")
        if getattr(self.config, "train_lact_only", False):
            if freeze_vision:
                raise ValueError("train_lact_only requires freeze_vision=False")
            if not hasattr(self.vision_tower, "_is_lact_state_key"):
                raise TypeError("train_lact_only requires a VideoChat3 LACT vision tower")
            self.vision_tower.requires_grad_(False)
            trainable_numel = 0
            for name, parameter in self.vision_tower.named_parameters():
                if self.vision_tower._is_lact_state_key(name):
                    parameter.requires_grad_(True)
                    trainable_numel += parameter.numel()
            logger.info(
                f"Train LACT-only vision parameters: {trainable_numel / 1e6:.1f}M"
            )
        if getattr(self.config, "freeze_lact_memory_gate", False):
            if not hasattr(self.vision_tower, "_is_lact_state_key"):
                raise TypeError(
                    "freeze_lact_memory_gate requires a VideoChat3 LACT vision tower"
                )
            frozen_gate_numel = 0
            for name, parameter in self.vision_tower.named_parameters():
                if name.endswith("memory_gate"):
                    parameter.requires_grad_(False)
                    frozen_gate_numel += parameter.numel()
            if frozen_gate_numel == 0:
                raise ValueError("No LACT memory gates were found to freeze")
            logger.info(
                f"Freeze LACT memory gates: {frozen_gate_numel / 1e6:.3f}M"
            )

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        # TODO: 判断其余模块是否已经被 fsdp 切分了
        # NOTE: 暂时只能在这个地方进行 checkpoint_wrapper
        # TODO: 当只训练某个部分时候，不能开启 checkpoint，否则 grad 是 None, 后续有需要再支持。
        # self.multi_modal_projector = checkpoint_wrapper(self.multi_modal_projector,  # type: ignore
        #                                                     checkpoint_impl=CheckpointImpl.REENTRANT)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )

        self.fsdp_mesh = init_world_mesh()
        # Note: 非常关键，不能删除这个 assert
        assert self.fsdp_mesh is not None

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

        self.language_model.embed_tokens.set_modules_to_forward_prefetch(   # type: ignore
            [self.vision_tower.encoder.blocks[0]])
        self.vision_tower.encoder.blocks[-1].set_modules_to_forward_prefetch(   # type: ignore
            [self.multi_modal_projector])
        self.multi_modal_projector.set_modules_to_forward_prefetch([self.language_model])  # type: ignore
        self.language_model.set_modules_to_forward_prefetch([self.language_model.layers["0"]])  # type: ignore
        self._to_empty_meta()
        return self

    def from_hf(self, hf_path: str | Path, strict=True):
        self._hf_path = Path(hf_path)

        if isinstance(hf_path, Path):
            hf_path = str(hf_path)

        _, _, missing_llm_keys = self.language_model.from_hf(hf_path, strict=False)
        _, _, missing_vision_keys = self.vision_tower.from_hf(hf_path, strict=False)
        _, _, missing_project_keys = self.multi_modal_projector.from_hf(hf_path, strict=False)

        missing = missing_llm_keys | missing_vision_keys | missing_project_keys
        if strict:
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)}. ")

    @override
    def save_hf(
        self,
        hf_dir: str | Path,
        save_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().save_hf(hf_dir=hf_dir, save_dtype=save_dtype)
        self.export_custom_hf_artifacts(hf_dir)

    def export_custom_hf_artifacts(self, hf_dir: str | Path) -> None:
        """Finalize custom model/processor files after generic HF asset saves."""
        if isinstance(self.config.vision_config, VideoChat3LACTVisionConfig):
            if not dist.is_initialized() or dist.get_rank() == 0:
                from .hf_lact_export import export_lact_hf_artifacts

                export_lact_hf_artifacts(hf_dir, self.config)
            if dist.is_initialized():
                dist.barrier()
        elif (
            self.config.vision_config.macro_temporal_compression_factor > 1
            or self.config.vision_config.chunk_query
        ):
            if not dist.is_initialized() or dist.get_rank() == 0:
                from .hf_macro_export import export_macro_hf_artifacts

                export_macro_hf_artifacts(hf_dir, self.config)
            if dist.is_initialized():
                dist.barrier()

    def scale_and_reduce_grad(self):
        self.language_model.scale_and_reduce_grad()

    def extract_feature(self, pixel_values, grid_thws):
        vit_embeds = self.vision_tower(
            pixel_values=pixel_values, grid_thws=grid_thws)
        vit_embeds = self.multi_modal_projector(vit_embeds)
        return vit_embeds


    def forward(
        self,
        seq_ctx: SequenceContext,
        loss_ctx: CELossContext | None = None,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        pixel_values = seq_ctx.pixel_values
        image_grid_thw = seq_ctx.image_grid_thw
        # NOTE image和video share pixel_values
        sequence_parallel_mesh = seq_ctx.sequence_parallel_mesh

        inputs_embeds = self.language_model.embed_tokens(input_ids)  # type: ignore

        if pixel_values is not None and image_grid_thw is not None:

            # in-place op on custom-function outputs will spoil autograd
            inputs_embeds = inputs_embeds.clone()

            if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
                raise NotImplementedError("VideoChat3 does not support sequence parallel now!")

            vit_embeds = self.extract_feature(pixel_values, image_grid_thw)
 
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids = cast(torch.LongTensor, input_ids.reshape(B * N))
            selected = (input_ids == self.img_context_token_id) | (input_ids == self.video_context_token_id)

            try:
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.sum() * 0

            inputs_embeds = inputs_embeds.reshape(B, N, C)
        else:
            # 构建假数据，考虑到 moe 特性，最好不要构建全 0 数据
            pixel_values_dump = torch.randn(4, 588, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_grid_thw = torch.tensor([[1, 2, 2]], device=inputs_embeds.device)
            vit_embeds = self.extract_feature(pixel_values_dump, image_grid_thw)
            inputs_embeds = inputs_embeds + vit_embeds.sum() * 0.0
            inputs_embeds = inputs_embeds.clone()


        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            inputs_embeds = split_for_sequence_parallel(inputs_embeds, dim=1, sp_mesh=sequence_parallel_mesh)

        # NOTE: 一定不要原地覆盖，否则第二次 forward 会缺少数据
        lang_seq_ctx = SequenceContext(input_ids=None,
                                       cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
                                       cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
                                       max_length_q=seq_ctx.max_length_q,
                                       max_length_k=seq_ctx.max_length_k,
                                       position_ids=seq_ctx.position_ids,
                                       num_padding=seq_ctx.num_padding,
                                       sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
                                       inputs_embeds=inputs_embeds)

        outputs = self.language_model(
            lang_seq_ctx,
            loss_ctx
        )
        return outputs

    @override
    def init_weights(self) -> None:
        self.vision_tower.init_weights()
        self.language_model.init_weights()
        self.multi_modal_projector.init_weights()


class VideoChat3LACTForConditionalGeneration(
    VideoChat3ForConditionalGeneration
):
    """VideoChat3 with the dedicated recurrent LACT vision encoder."""

    def __init__(self, config: VideoChat3BaseConfig):
        if not isinstance(config.vision_config, VideoChat3LACTVisionConfig):
            raise TypeError(
                "VideoChat3LACTForConditionalGeneration requires a "
                "VideoChat3LACTVisionConfig"
            )
        super().__init__(config)
