"""VideoChat3 with LVSM token memory and final-bank visual output."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel

from .configuration_videochat3_tas import VideoChat3TASConfig, VideoChat3TASVisionConfig
from .modeling_videochat3 import (
    VideoChat3VisionModel, VideoChat3Model, VideoChat3ForConditionalGeneration,
    VideoChat3PreTrainedModel, VideoChat3MultiModalProjector, VideoChat3ModelOutputWithPast,
)
from .tas_memory import TASMemory


class VideoChat3TASVisionModel(VideoChat3VisionModel):
    config_class = VideoChat3TASVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.tas = TASMemory(config)
        self.checkpoint_chunks = True
        self.state_mode = "continuous"

    def _chunk(self, x, old, identity, grid, times):
        # All readers observe the exact same old bank. Write only after all layers.
        rope = self.encoder.rope_2d.get_freqs_cis(grid_thws=grid)
        cu = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        for i, block in enumerate(self.encoder.blocks):
            z = x + block.attention_qkvpacked(block.norm0(x), cu, rope_freqs_cis=rope)
            if i == len(self.encoder.blocks) - 1:
                h = self.tas.wfl_norm(z)
            else:
                y = z + self.tas.readers[i](z, old, identity)
                x = y + block.mlp(block.norm1(y))
        return self.tas.update(h, old, identity, times, int(grid[0, 1] * grid[0, 2]))

    def forward(self, pixel_values, grid_thws, frame_times=None):
        # Images retain the original path and pretrained projector semantics.
        if frame_times is None and all(t == 1 for t, _, _ in grid_thws.tolist()):
            return super().forward(pixel_values, grid_thws)
        if self.config.tas_time_encoding and frame_times is None:
            raise ValueError("Missing TAS frame times for video input")
        if frame_times is not None and frame_times.shape != (int(grid_thws[:, 0].sum()), 2):
            raise ValueError("TAS timestamps must contain one (seconds, duration) row per frame")
        outputs = []
        pixel_offset = time_offset = 0
        for t, h, w in grid_thws.tolist():
            count = self.config.tas_memory_tokens or self.config.temporal_merge_size * h * w
            old = identity = None
            for start in range(0, t, self.config.temporal_merge_size):
                frames = min(self.config.temporal_merge_size, t-start)
                grid = grid_thws.new_tensor([[frames, h, w]])
                pixels = pixel_values[pixel_offset:pixel_offset + frames*h*w]
                x = self.patch_embed(pixels, grid)
                if old is None or self.state_mode == "reset_state":
                    old, identity = self.tas.start(count, x.dtype)
                times = None if frame_times is None else frame_times[time_offset:time_offset+frames]
                if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                    old = checkpoint(self._chunk, x, old, identity, grid, times,
                                     use_reentrant=False, preserve_rng_state=False)
                else:
                    old = self._chunk(x, old, identity, grid, times)
                pixel_offset += frames*h*w
                time_offset += frames
            state = self.encoder.final_layernorm(old.squeeze(0))
            # One slot -> one LM token. Repeat that SAME slot into the pretrained
            # projector's four channels; do not pool arbitrary neighboring slots.
            area = self.config.merge_kernel_size[0] * self.config.merge_kernel_size[1]
            outputs.append(state[:, None, :].expand(-1, area, -1))
        return outputs


class VideoChat3TASModel(VideoChat3Model):
    config_class = VideoChat3TASConfig

    def __init__(self, config):
        VideoChat3PreTrainedModel.__init__(self, config)
        self.vision_tower = VideoChat3TASVisionModel._from_config(config.vision_config)
        self.multi_modal_projector = VideoChat3MultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config, trust_remote_code=True)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, pixel_values=None,
                pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                cache_position=None, tas_frame_times=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        image_embeds = video_embeds = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).to(inputs_embeds)
            mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
        if pixel_values_videos is not None:
            features = self.vision_tower(pixel_values_videos, video_grid_thw, frame_times=tas_frame_times)
            video_embeds = self.multi_modal_projector(features).to(inputs_embeds)
            _, mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask, video_embeds)
        out = self.language_model(attention_mask=attention_mask, position_ids=position_ids,
                                  past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                  cache_position=cache_position, **kwargs)
        return VideoChat3ModelOutputWithPast(last_hidden_state=out.last_hidden_state,
                 past_key_values=out.past_key_values, hidden_states=out.hidden_states,
                 attentions=out.attentions, image_hidden_states=image_embeds, video_hidden_states=video_embeds)


class VideoChat3TASForConditionalGeneration(VideoChat3ForConditionalGeneration):
    config_class = VideoChat3TASConfig

    def __init__(self, config):
        VideoChat3PreTrainedModel.__init__(self, config)
        self.model = VideoChat3TASModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(self, *args, tas_frame_times=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if inputs.get("pixel_values_videos") is not None:
            inputs["tas_frame_times"] = tas_frame_times
        return inputs

    def freeze_for_tas_training(self):
        self.requires_grad_(False)
        self.model.vision_tower.requires_grad_(True)
        self.model.multi_modal_projector.requires_grad_(True)
        # Canonical LVSM WFL uses last-layer post-attention/pre-read h. These
        # retained pretrained tensors have no path to the final bank objective.
        last = self.model.vision_tower.encoder.blocks[-1]
        last.norm1.requires_grad_(False)
        last.mlp.requires_grad_(False)
        self.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
