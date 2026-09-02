import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.activations import ACT2FN

from .configuration_videochat3_lact import (
    VideoChat3LACTConfig,
    VideoChat3LACTVisionConfig,
)
from .modeling_videochat3 import (
    Rope2DPosEmb,
    VideoChat3ForConditionalGeneration,
    VideoChat3Model,
    VideoChat3MultiModalProjector,
    VideoChat3PreTrainedModel,
    VideoChat3VisionLayer,
    VideoChat3VisionPatchEmbed,
    VideoChat3VisionPreTrainedModel,
    patch_merger,
)


_NS_GRAD_RATIO_RHO = 1.0
_NS_GRAD_RATIO_EPS = 1e-12
_STATE_GRAD_RATIO_RHO = 1.0
_STATE_GRAD_RATIO_EPS = 1e-12


def _compress_chunk_outputs(
    chunk_outputs: list[torch.Tensor],
    video_clip_counts: list[int],
    factor: int,
    mode: str = "auto",
) -> list[torch.Tensor]:
    """Compress final-layer chunk outputs without crossing videos."""
    if factor not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported macro temporal compression factor: {factor}")
    if mode == "auto":
        mode = "select_last"
    if mode not in ("mean", "select_last", "video_last"):
        raise ValueError(f"Unsupported macro temporal compression mode: {mode!r}")
    if sum(video_clip_counts) != len(chunk_outputs):
        raise ValueError(
            f"video_clip_counts={video_clip_counts} do not cover "
            f"{len(chunk_outputs)} outputs"
        )
    if factor == 1 and mode != "video_last":
        return chunk_outputs
    compressed = []
    offset = 0
    for clip_count in video_clip_counts:
        if clip_count <= 0:
            raise ValueError(f"clip_count must be positive, got {clip_count}")
        video_outputs = chunk_outputs[offset : offset + clip_count]
        if mode == "video_last":
            compressed.append(video_outputs[-1])
            offset += clip_count
            continue
        for start in range(0, clip_count, factor):
            group = video_outputs[start : min(start + factor, clip_count)]
            if mode == "select_last":
                compressed.append(group[-1])
                continue
            reference_shape = group[0].shape
            if any(item.shape != reference_shape for item in group[1:]):
                raise ValueError(
                    "Chunk output shapes must match within a video for temporal mean pooling: "
                    f"{[tuple(item.shape) for item in group]}"
                )
            compressed.append(group[0] if len(group) == 1 else torch.stack(group).mean(dim=0))
        offset += clip_count
    return compressed


def inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def inverse_sigmoid(value: float) -> float:
    if not 0 < value < 1:
        raise ValueError(f"Expected a value in (0, 1), got {value}")
    return math.log(value / (1 - value))


def _zeropower_via_newtonschulz5_impl(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    if steps == 0:
        return matrix
    if matrix.ndim != 3:
        raise ValueError(f"Expected a 3D matrix batch, got shape {matrix.shape}")

    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.bfloat16()
    transpose = matrix.shape[-2] > matrix.shape[-1]
    if transpose:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        polynomial = b * gram + (c * gram) @ gram
        x = a * x + polynomial @ x
    if transpose:
        x = x.transpose(-2, -1)
    return x


class _NS5WithBoundedBackward(torch.autograd.Function):
    """Run exact NS5 forward and cap each matrix's realized backward gain."""

    @staticmethod
    def forward(ctx, matrix: torch.Tensor, steps: int) -> torch.Tensor:
        ctx.steps = int(steps)
        ctx.save_for_backward(matrix)
        return _zeropower_via_newtonschulz5_impl(matrix, ctx.steps)

    @staticmethod
    def backward(ctx, grad_update: torch.Tensor):
        (matrix,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None

        with torch.enable_grad():
            recomputed_matrix = matrix.detach().requires_grad_(True)
            recomputed_update = _zeropower_via_newtonschulz5_impl(recomputed_matrix, ctx.steps)
            (exact_grad,) = torch.autograd.grad(
                recomputed_update,
                recomputed_matrix,
                grad_update,
                create_graph=False,
            )

        update_norm = torch.linalg.vector_norm(grad_update.float(), dim=(-2, -1), keepdim=True)
        exact_norm = torch.linalg.vector_norm(exact_grad.float(), dim=(-2, -1), keepdim=True)
        needs_clip = exact_norm > _NS_GRAD_RATIO_RHO * update_norm
        scale = torch.where(
            needs_clip,
            _NS_GRAD_RATIO_RHO * update_norm / exact_norm.clamp_min(_NS_GRAD_RATIO_EPS),
            torch.ones_like(exact_norm),
        )
        return exact_grad * scale.to(exact_grad.dtype), None


class _NS5WithRecomputedBackward(torch.autograd.Function):
    """Checkpoint NS5 internals while returning the exact, unclipped VJP."""

    @staticmethod
    def forward(ctx, matrix: torch.Tensor, steps: int) -> torch.Tensor:
        ctx.steps = int(steps)
        ctx.save_for_backward(matrix)
        return _zeropower_via_newtonschulz5_impl(matrix, ctx.steps)

    @staticmethod
    def backward(ctx, grad_update: torch.Tensor):
        (matrix,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        with torch.enable_grad():
            recomputed_matrix = matrix.detach().requires_grad_(True)
            recomputed_update = _zeropower_via_newtonschulz5_impl(
                recomputed_matrix,
                ctx.steps,
            )
            (exact_grad,) = torch.autograd.grad(
                recomputed_update,
                recomputed_matrix,
                grad_update,
                create_graph=False,
            )
        return exact_grad, None


class _StateGradRatioContext:
    def __init__(self):
        self.next_state_grad_norm = None


def _joint_state_grad_norm(
    gradients: tuple[torch.Tensor | None, ...],
    *,
    norm_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    squared_norms = [
        torch.linalg.vector_norm(
            gradient,
            dim=(-2, -1),
            dtype=torch.float32,
        ).square()
        for gradient in gradients
        if gradient is not None
    ]
    if not squared_norms:
        return torch.zeros(norm_shape, device=device, dtype=torch.float32)
    return torch.sqrt(sum(squared_norms))


class _CaptureNextStateGradient(torch.autograd.Function):
    """Capture the full adjoint of the state produced by one FW update."""

    @staticmethod
    def forward(ctx, fast_w0, fast_w1, fast_w2, master_w0, master_w1, master_w2, ratio_context):
        ctx.ratio_context = ratio_context
        ctx.norm_shape = fast_w0.shape[:-2]
        ctx.device = fast_w0.device
        return fast_w0, fast_w1, fast_w2, master_w0, master_w1, master_w2

    @staticmethod
    def backward(ctx, grad_fast_w0, grad_fast_w1, grad_fast_w2, grad_master_w0, grad_master_w1, grad_master_w2):
        gradients = (
            grad_fast_w0,
            grad_fast_w1,
            grad_fast_w2,
            grad_master_w0,
            grad_master_w1,
            grad_master_w2,
        )
        ctx.ratio_context.next_state_grad_norm = _joint_state_grad_norm(
            gradients,
            norm_shape=ctx.norm_shape,
            device=ctx.device,
        )
        return *gradients, None


class _BoundPreviousStateGradient(torch.autograd.Function):
    """Cap the full previous-state adjoint by the next-state adjoint norm."""

    @staticmethod
    def forward(ctx, fast_w0, fast_w1, fast_w2, master_w0, master_w1, master_w2, ratio_context):
        ctx.ratio_context = ratio_context
        ctx.norm_shape = fast_w0.shape[:-2]
        ctx.device = fast_w0.device
        return fast_w0, fast_w1, fast_w2, master_w0, master_w1, master_w2

    @staticmethod
    def backward(ctx, grad_fast_w0, grad_fast_w1, grad_fast_w2, grad_master_w0, grad_master_w1, grad_master_w2):
        gradients = (
            grad_fast_w0,
            grad_fast_w1,
            grad_fast_w2,
            grad_master_w0,
            grad_master_w1,
            grad_master_w2,
        )
        next_state_grad_norm = ctx.ratio_context.next_state_grad_norm
        if next_state_grad_norm is None:
            raise RuntimeError("Next-state adjoint was not captured before previous-state backward")
        previous_state_grad_norm = _joint_state_grad_norm(
            gradients,
            norm_shape=ctx.norm_shape,
            device=ctx.device,
        )
        needs_clip = previous_state_grad_norm > _STATE_GRAD_RATIO_RHO * next_state_grad_norm
        scale = torch.where(
            needs_clip,
            _STATE_GRAD_RATIO_RHO
            * next_state_grad_norm
            / previous_state_grad_norm.clamp_min(_STATE_GRAD_RATIO_EPS),
            torch.ones_like(previous_state_grad_norm),
        )
        scale = scale[..., None, None]
        clipped_gradients = tuple(
            None if gradient is None else gradient * scale.to(gradient.dtype)
            for gradient in gradients
        )
        return *clipped_gradients, None


class _CaptureNextLinearStateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, ratio_context):
        ctx.ratio_context = ratio_context
        return state

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        ctx.ratio_context.next_state_grad_norm = torch.linalg.vector_norm(
            gradient,
            dim=(-2, -1),
            dtype=torch.float32,
        )
        return gradient, None


class _BoundPreviousLinearStateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, ratio_context):
        ctx.ratio_context = ratio_context
        return state

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        next_norm = ctx.ratio_context.next_state_grad_norm
        if next_norm is None:
            raise RuntimeError(
                "Next linear-state adjoint was not captured before backward"
            )
        previous_norm = torch.linalg.vector_norm(
            gradient,
            dim=(-2, -1),
            dtype=torch.float32,
        )
        scale = torch.where(
            previous_norm > _STATE_GRAD_RATIO_RHO * next_norm,
            _STATE_GRAD_RATIO_RHO
            * next_norm
            / previous_norm.clamp_min(_STATE_GRAD_RATIO_EPS),
            torch.ones_like(previous_norm),
        )
        return gradient * scale[..., None, None].to(gradient.dtype), None


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor,
    steps: int = 5,
    *,
    clip_ns_grad_ratio: bool = False,
    recompute_ns5_backward: bool = True,
) -> torch.Tensor:
    """VideoLACT NS5 with independent exact-backward checkpointing."""
    if steps == 0 or not matrix.requires_grad:
        return _zeropower_via_newtonschulz5_impl(matrix, steps)
    if clip_ns_grad_ratio:
        return _NS5WithBoundedBackward.apply(matrix, steps)
    if recompute_ns5_backward:
        return _NS5WithRecomputedBackward.apply(matrix, steps)
    return _zeropower_via_newtonschulz5_impl(matrix, steps)


class FastWeightSwiGLU(nn.Module):
    """VideoLACT multi-head SwiGLU fast weights."""

    def __init__(
        self,
        dim: int,
        inter_multi: float = 2,
        num_heads: int = 1,
        share_proj: bool = False,
        norm_epsilon: float = 1e-5,
        inner_optim: str = "muon",
        clip_ns_grad_ratio: bool = False,
        recompute_ns5_backward: bool = True,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if inter_multi <= 0:
            raise ValueError(f"inter_multi must be positive, got {inter_multi}")
        if inner_optim not in ("muon", "delta", "sgd"):
            raise ValueError(
                "inner_optim must be 'muon', 'delta', or the legacy 'sgd' "
                f"alias, got {inner_optim!r}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.hidden_dim = int(dim * inter_multi)
        self.share_proj = share_proj
        self.inner_optim = inner_optim
        self.clip_ns_grad_ratio = clip_ns_grad_ratio
        self.recompute_ns5_backward = recompute_ns5_backward

        if share_proj:
            self.apply_proj = None
            self.update_proj = None
            self.output_proj = None
        else:
            self.apply_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
            self.update_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
            self.output_proj = nn.Linear(dim, dim, bias=False)
        self.apply_norm = nn.RMSNorm(dim, eps=norm_epsilon, elementwise_affine=False)
        self.output_norm = nn.RMSNorm(dim, eps=norm_epsilon, elementwise_affine=False)
        self.w0 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.hidden_dim))
        self.w1 = nn.Parameter(torch.empty(num_heads, self.hidden_dim, self.head_dim))
        self.w2 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.hidden_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.w0, std=1 / math.sqrt(self.head_dim))
        nn.init.normal_(self.w1, std=1 / math.sqrt(self.hidden_dim))
        nn.init.normal_(self.w2, std=1 / math.sqrt(self.head_dim))
        if self.apply_proj is not None:
            nn.init.trunc_normal_(self.apply_proj[0].weight, std=0.02)
            nn.init.trunc_normal_(self.update_proj[0].weight, std=0.02)
            nn.init.zeros_(self.output_proj.weight)

    def init_fast_weights(self, batch_size: int):
        master_weights = tuple(
            weight.float().unsqueeze(0).repeat(batch_size, 1, 1, 1) for weight in (self.w0, self.w1, self.w2)
        )
        fast_dtype = torch.bfloat16 if master_weights[0].is_cuda else master_weights[0].dtype
        fast_weights = tuple(F.normalize(weight, dim=2, eps=1e-5).to(fast_dtype) for weight in master_weights)
        return fast_weights, master_weights

    def _apply_fast_weights(self, x: torch.Tensor, fast_weights):
        w0, w1, w2 = fast_weights
        output_dtype = x.dtype
        x = x.to(w0.dtype)
        batch_size, seq_len, _ = x.shape
        x_heads = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        gate = torch.einsum("blhd,bhdk->blhk", x_heads, w0)
        up = torch.einsum("blhd,bhdk->blhk", x_heads, w2)
        hidden = F.silu(gate) * up
        output = torch.einsum("blhk,bhkd->blhd", hidden, w1)
        output = output.reshape(batch_size, seq_len, self.dim).to(output_dtype)
        return output, x_heads, gate, up, hidden

    def forward(self, x: torch.Tensor, fast_weights):
        if self.apply_proj is not None:
            x = self.apply_norm(self.apply_proj(x))
        output, _, _, _, _ = self._apply_fast_weights(x, fast_weights)
        output = self.output_norm(output)
        if self.output_proj is not None:
            output = self.output_proj(output)
        return output

    def update(
        self,
        memory_input: torch.Tensor,
        target: torch.Tensor,
        learning_rates: torch.Tensor,
        fast_weights,
        master_weights,
        muon_update_steps: int,
    ):
        key = memory_input
        if self.update_proj is not None:
            key = self.apply_norm(self.update_proj(key))
        return self.update_preprojected(
            key,
            target,
            learning_rates,
            fast_weights,
            master_weights,
            muon_update_steps,
        )

    def update_preprojected(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        learning_rates: torch.Tensor,
        fast_weights,
        master_weights,
        muon_update_steps: int,
    ):
        w0, w1, w2 = fast_weights
        master_w0, master_w1, master_w2 = master_weights
        batch_size, seq_len, _ = key.shape
        output, key_heads, gate, up, hidden = self._apply_fast_weights(key, fast_weights)

        error = (target.float() - output.float()) / seq_len
        error_heads = error.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        d_hidden = torch.einsum(
            "blhd,bhdk->blhk",
            error_heads,
            w1.float().transpose(-1, -2),
        )
        d_up = d_hidden * F.silu(gate)
        d_gate = d_hidden * up
        sigmoid = torch.sigmoid(gate)
        d_gate_pre = d_gate * sigmoid * (1 + gate * (1 - sigmoid))

        lr0, lr1, lr2 = learning_rates.float().split(1, dim=-1)
        key_heads = key_heads.float()
        hidden = hidden.float()
        error_heads = error_heads.float()
        d_gate_pre = d_gate_pre.float()
        d_up = d_up.float()
        batch_heads = batch_size * self.num_heads
        gradient_left = (
            torch.stack(
                (
                    key_heads * lr0.unsqueeze(2),
                    error_heads,
                    key_heads * lr2.unsqueeze(2),
                )
            )
            .permute(0, 1, 3, 4, 2)
            .reshape(3 * batch_heads, self.head_dim, seq_len)
        )
        gradient_right = (
            torch.stack(
                (
                    d_gate_pre,
                    hidden * lr1.unsqueeze(2),
                    d_up,
                )
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(3 * batch_heads, seq_len, self.hidden_dim)
        )
        w0_grad, w1_grad_transposed, w2_grad = (
            torch.bmm(gradient_left, gradient_right)
            .reshape(
                3,
                batch_size,
                self.num_heads,
                self.head_dim,
                self.hidden_dim,
            )
            .unbind(0)
        )
        w1_grad = w1_grad_transposed.transpose(-1, -2)

        if self.inner_optim in ("delta", "sgd"):
            w0_update = w0_grad
            w1_update = w1_grad
            w2_update = w2_grad
        else:
            transpose_w02 = self.head_dim > self.hidden_dim
            if transpose_w02:
                muon_gradients = torch.stack(
                    (
                        w0_grad.transpose(-1, -2),
                        w1_grad,
                        w2_grad.transpose(-1, -2),
                    )
                )
            else:
                muon_gradients = torch.stack(
                    (w0_grad, w1_grad.transpose(-1, -2), w2_grad)
                )
            muon_updates = zeropower_via_newtonschulz5(
                muon_gradients.flatten(0, 2),
                muon_update_steps,
                clip_ns_grad_ratio=self.clip_ns_grad_ratio,
                recompute_ns5_backward=self.recompute_ns5_backward,
            ).reshape_as(muon_gradients)
            w0_update, w1_update_oriented, w2_update = muon_updates.unbind(0)
            if transpose_w02:
                w0_update = w0_update.transpose(-1, -2)
                w1_update = w1_update_oriented
                w2_update = w2_update.transpose(-1, -2)
            else:
                w1_update = w1_update_oriented.transpose(-1, -2)
        master_weights = (
            master_w0 + w0_update.to(master_w0.dtype),
            master_w1 + w1_update.to(master_w1.dtype),
            master_w2 + w2_update.to(master_w2.dtype),
        )
        fast_weights = tuple(F.normalize(weight, dim=2, eps=1e-5).to(w0.dtype) for weight in master_weights)
        return fast_weights, master_weights


class FastWeightLinear(nn.Module):
    """Multi-head linear fast memory with one chunk-level state update."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        inner_optim: str = "muon",
        norm_epsilon: float = 1e-5,
        clip_ns_grad_ratio: bool = False,
        recompute_ns5_backward: bool = True,
    ):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError(
                f"dim={dim} must be divisible by positive num_heads={num_heads}"
            )
        if inner_optim not in ("muon", "delta", "sgd"):
            raise ValueError(
                "inner_optim must be 'muon', 'delta', or the legacy 'sgd' "
                f"alias, got {inner_optim!r}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_optim = inner_optim
        self.clip_ns_grad_ratio = clip_ns_grad_ratio
        self.recompute_ns5_backward = recompute_ns5_backward
        self.apply_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        self.update_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.apply_norm = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            elementwise_affine=False,
        )
        self.output_norm = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            elementwise_affine=False,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.apply_proj[0].weight, std=0.02)
        nn.init.trunc_normal_(self.update_proj[0].weight, std=0.02)
        nn.init.zeros_(self.output_proj.weight)

    def init_state(self, batch_size: int) -> torch.Tensor:
        weight = self.apply_proj[0].weight
        dtype = torch.bfloat16 if weight.is_cuda else weight.dtype
        return torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=weight.device,
            dtype=dtype,
        )

    def _project_heads(
        self,
        x: torch.Tensor,
        projection: nn.Sequential,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        projected = self.apply_norm(projection(x))
        projected = projected.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        return F.normalize(projected, dim=-1, eps=1e-5)

    def project_key(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_heads(x, self.update_proj)

    def project_query(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_heads(x, self.apply_proj)

    def apply_projected(
        self,
        query: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.einsum("blhd,bhde->blhe", query, state)
        return output.reshape(query.shape[0], query.shape[1], self.dim)

    def project_output(self, output: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.output_norm(output))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        query = self.project_query(x)
        return self.project_output(self.apply_projected(query, state))

    def update_projected(
        self,
        key_heads: torch.Tensor,
        target: torch.Tensor,
        learning_rates: torch.Tensor,
        state: torch.Tensor,
        muon_update_steps: int,
    ) -> torch.Tensor:
        target_heads = target.reshape(
            target.shape[0],
            target.shape[1],
            self.num_heads,
            self.head_dim,
        ).to(state.dtype)
        prediction = torch.einsum("blhd,bhde->blhe", key_heads, state)
        error = target_heads - prediction
        weighted_error = error * learning_rates.to(error.dtype).unsqueeze(-1)
        state_gradient = torch.einsum(
            "blhd,blhe->bhde",
            key_heads,
            weighted_error,
        ) / key_heads.shape[1]
        if self.inner_optim == "muon":
            state_update = zeropower_via_newtonschulz5(
                state_gradient.flatten(0, 1),
                muon_update_steps,
                clip_ns_grad_ratio=self.clip_ns_grad_ratio,
                recompute_ns5_backward=self.recompute_ns5_backward,
            ).reshape_as(state_gradient)
        else:
            state_update = state_gradient
        return state + state_update.to(state.dtype)

    def update(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        learning_rates: torch.Tensor,
        state: torch.Tensor,
        muon_update_steps: int,
    ) -> torch.Tensor:
        key_heads = self.project_key(key).to(state.dtype)
        return self.update_projected(
            key_heads,
            target,
            learning_rates,
            state,
            muon_update_steps,
        )


class VideoChat3LACTVisionLayer(VideoChat3VisionLayer):
    """Original four-frame attention followed by VideoLACT fast weights."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_impl: str = "eager",
        activation=F.gelu,
        attn_bias: bool = False,
        fw_inter_multi: float = 2,
        fw_num_heads: int = 1,
        fw_base_lr: float = 0.01,
        fw_muon_update_steps: int = 5,
        memory_type: str = "swiglu",
        inner_optim: str = "muon",
        fw_share_proj: bool = False,
        fw_share_init: bool = True,
        fw_norm_epsilon: float = 1e-5,
        clip_ns_grad_ratio: bool = False,
        recompute_ns5_backward: bool = True,
        clip_state_grad_ratio: bool = True,
        lact_inference_state_mode: str = "continuous",
    ):
        super().__init__(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            attn_impl=attn_impl,
            activation=activation,
            attn_bias=attn_bias,
        )
        if memory_type not in ("swiglu", "linear"):
            raise ValueError(
                f"memory_type must be 'swiglu' or 'linear', got {memory_type!r}"
            )
        if memory_type == "linear" and fw_share_proj:
            raise ValueError("linear memory currently requires private projections")
        self.memory_type = memory_type
        self.fw_share_proj = fw_share_proj
        self.fw_share_init = fw_share_init
        self.clip_state_grad_ratio = clip_state_grad_ratio
        if lact_inference_state_mode not in ("continuous", "reset_state"):
            raise ValueError(f"Unsupported LACT inference state mode: {lact_inference_state_mode!r}")
        self.lact_inference_state_mode = lact_inference_state_mode
        self.memory_norm = nn.RMSNorm(hidden_dim, eps=fw_norm_epsilon)
        if memory_type == "linear":
            self.memory = FastWeightLinear(
                hidden_dim,
                num_heads=fw_num_heads,
                inner_optim=inner_optim,
                norm_epsilon=fw_norm_epsilon,
                clip_ns_grad_ratio=clip_ns_grad_ratio,
                recompute_ns5_backward=recompute_ns5_backward,
            )
        else:
            self.memory = FastWeightSwiGLU(
                hidden_dim,
                inter_multi=fw_inter_multi,
                num_heads=fw_num_heads,
                share_proj=fw_share_proj,
                norm_epsilon=fw_norm_epsilon,
                inner_optim=inner_optim,
                clip_ns_grad_ratio=clip_ns_grad_ratio,
                recompute_ns5_backward=recompute_ns5_backward,
            )
        self.value_proj = None if fw_share_proj else nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_gate = nn.Parameter(torch.zeros(hidden_dim))
        if memory_type == "linear":
            self.lr_proj = None
            self.beta_proj = nn.Linear(hidden_dim, fw_num_heads, bias=False)
            self.base_lr_inverse = inverse_sigmoid(fw_base_lr)
        else:
            self.lr_proj = nn.Linear(hidden_dim, 3, bias=False)
            self.beta_proj = None
            self.base_lr_inverse = inverse_softplus(fw_base_lr)
        self.muon_update_steps = fw_muon_update_steps
        self.reset_lact_parameters()

    @torch.no_grad()
    def reset_lact_parameters(self) -> None:
        nn.init.ones_(self.memory_norm.weight)
        if getattr(self.memory_norm, "bias", None) is not None:
            nn.init.zeros_(self.memory_norm.bias)
        self.memory.reset_parameters()
        if self.lr_proj is not None:
            nn.init.trunc_normal_(self.lr_proj.weight, std=0.02)
        if self.beta_proj is not None:
            nn.init.trunc_normal_(self.beta_proj.weight, std=0.02)
        nn.init.zeros_(self.memory_gate)
        if self.value_proj is not None:
            nn.init.trunc_normal_(self.value_proj.weight, std=0.02)
        if self.fw_share_init and not self.fw_share_proj:
            query, key, value = self.wqkv.weight.detach().chunk(3, dim=0)
            self.memory.apply_proj[0].weight.copy_(query)
            self.memory.update_proj[0].weight.copy_(key)
            self.value_proj.weight.copy_(value)
            self.memory.output_proj.weight.copy_(self.wo.weight.detach())

    def init_fast_weights(self, batch_size: int):
        if self.memory_type != "swiglu":
            raise RuntimeError("init_fast_weights only applies to SwiGLU memory")
        return self.memory.init_fast_weights(batch_size)

    def init_linear_state(self, batch_size: int) -> torch.Tensor:
        if self.memory_type != "linear":
            raise RuntimeError("init_linear_state only applies to linear memory")
        return self.memory.init_state(batch_size)

    def _shared_qkv(self, memory_input: torch.Tensor):
        query, key, value = self.wqkv(memory_input).chunk(3, dim=-1)
        query = self.memory.apply_norm(F.silu(query))
        key = self.memory.apply_norm(F.silu(key))
        return query, key, value.contiguous()

    def forward_attention(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attention = self.attention_qkvpacked(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis=rope_freqs_cis,
        )
        return residual + attention

    def apply_memory(self, hidden_states: torch.Tensor, fast_weights):
        memory_input = self.memory_norm(hidden_states)
        if self.fw_share_proj:
            query, key, target = self._shared_qkv(memory_input)
            memory_output = self.wo(self.memory(query, fast_weights))
        else:
            key = target = memory_input
            memory_output = self.memory(memory_input, fast_weights)
        hidden_states = hidden_states + memory_output * self.memory_gate
        return hidden_states, memory_input, key, target

    def update_fast_weights(
        self,
        memory_input: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        fast_weights,
        master_weights,
    ):
        if self.memory_type != "swiglu":
            raise RuntimeError("update_fast_weights only applies to SwiGLU memory")
        prediction_input = F.rms_norm(
            memory_input,
            normalized_shape=(memory_input.shape[-1],),
            eps=1e-5,
        )
        update_input = key
        if not self.fw_share_proj:
            update_input = memory_input
            target = self.value_proj(prediction_input)
        with torch.autocast(device_type=memory_input.device.type, enabled=False):
            learning_rates = F.softplus(
                F.linear(prediction_input.float(), self.lr_proj.weight.float()) + self.base_lr_inverse
            )
        return self.memory.update(
            update_input,
            target,
            learning_rates,
            fast_weights,
            master_weights,
            self.muon_update_steps,
        )

    def update_linear_state(
        self,
        memory_input: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if self.memory_type != "linear":
            raise RuntimeError("update_linear_state only applies to linear memory")
        prediction_input = F.rms_norm(
            memory_input,
            normalized_shape=(memory_input.shape[-1],),
            eps=1e-5,
        )
        target = self.value_proj(prediction_input)
        with torch.autocast(device_type=memory_input.device.type, enabled=False):
            learning_rates = torch.sigmoid(
                F.linear(
                    prediction_input.float(),
                    self.beta_proj.weight.float(),
                )
                + self.base_lr_inverse
            )
        return self.memory.update(
            memory_input,
            target,
            learning_rates,
            state,
            self.muon_update_steps,
        )

    def _forward_linear_memory_scan(
        self,
        hidden_states: torch.Tensor,
        clip_slices: list[tuple[int, int]],
        video_clip_counts: list[int],
    ) -> torch.Tensor:
        if sum(video_clip_counts) != len(clip_slices):
            raise ValueError(
                f"video_clip_counts={video_clip_counts} do not cover "
                f"{len(clip_slices)} clips"
            )
        clip_index = 0
        reset_state = self.lact_inference_state_mode == "reset_state"
        if reset_state and self.training:
            raise RuntimeError("reset_state is an inference-only LACT state mode")
        if reset_state:
            return hidden_states

        packed_memory_input = self.memory_norm(hidden_states.unsqueeze(0))
        packed_query = self.memory.project_query(packed_memory_input)
        packed_key = self.memory.project_key(packed_memory_input)
        prediction_input = F.rms_norm(
            packed_memory_input,
            normalized_shape=(packed_memory_input.shape[-1],),
            eps=1e-5,
        )
        packed_target = self.value_proj(prediction_input)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            packed_learning_rates = torch.sigmoid(
                F.linear(
                    prediction_input.float(),
                    self.beta_proj.weight.float(),
                )
                + self.base_lr_inverse
            )

        outputs = []
        for clip_count in video_clip_counts:
            if clip_count <= 0:
                raise ValueError(f"clip_count must be positive, got {clip_count}")
            video_start = clip_slices[clip_index][0]
            video_memory_outputs = []
            state = self.init_linear_state(batch_size=1)
            for video_clip_index in range(clip_count):
                will_update = video_clip_index + 1 < clip_count
                state_ratio_context = None
                if will_update and self.clip_state_grad_ratio:
                    state_ratio_context = _StateGradRatioContext()
                    group_state = _BoundPreviousLinearStateGradient.apply(
                        state,
                        state_ratio_context,
                    )
                else:
                    group_state = state
                start, end = clip_slices[clip_index]
                memory_output = self.memory.apply_projected(
                    packed_query[:, start:end],
                    group_state,
                )
                if will_update:
                    state = self.memory.update_projected(
                        packed_key[:, start:end].to(group_state.dtype),
                        packed_target[:, start:end],
                        packed_learning_rates[:, start:end],
                        group_state,
                        self.muon_update_steps,
                    )
                    if state_ratio_context is not None:
                        state = _CaptureNextLinearStateGradient.apply(
                            state,
                            state_ratio_context,
                        )
                video_memory_outputs.append(memory_output)
                clip_index += 1
            video_end = clip_slices[clip_index - 1][1]
            memory_output = self.memory.project_output(
                torch.cat(video_memory_outputs, dim=1)
            ).squeeze(0)
            outputs.append(
                hidden_states[video_start:video_end]
                + memory_output * self.memory_gate
            )
        return torch.cat(outputs, dim=0)

    def forward_memory_scan(
        self,
        hidden_states: torch.Tensor,
        clip_slices: list[tuple[int, int]],
        video_clip_counts: list[int],
    ) -> torch.Tensor:
        if self.memory_type == "linear":
            return self._forward_linear_memory_scan(
                hidden_states,
                clip_slices,
                video_clip_counts,
            )
        if self.lact_inference_state_mode == "reset_state":
            if self.training:
                raise RuntimeError("reset_state is an inference-only LACT state mode")
            outputs = []
            clip_index = 0
            for clip_count in video_clip_counts:
                for _ in range(clip_count):
                    fast_weights, _ = self.init_fast_weights(batch_size=1)
                    start, end = clip_slices[clip_index]
                    clip_hidden = hidden_states[start:end].unsqueeze(0)
                    clip_hidden, _, _, _ = self.apply_memory(clip_hidden, fast_weights)
                    outputs.append(clip_hidden.squeeze(0))
                    clip_index += 1
            if clip_index != len(clip_slices):
                raise ValueError(
                    f"Consumed {clip_index} clips, but layout contains {len(clip_slices)}"
                )
            return torch.cat(outputs, dim=0)

        outputs = []
        clip_index = 0
        for clip_count in video_clip_counts:
            fast_weights, master_weights = self.init_fast_weights(batch_size=1)
            for video_clip_index in range(clip_count):
                will_update = video_clip_index + 1 < clip_count
                state_ratio_context = None
                if will_update and self.clip_state_grad_ratio:
                    state_ratio_context = _StateGradRatioContext()
                    bounded_state = _BoundPreviousStateGradient.apply(
                        *fast_weights,
                        *master_weights,
                        state_ratio_context,
                    )
                    group_fast_weights = bounded_state[:3]
                    group_master_weights = bounded_state[3:]
                else:
                    group_fast_weights = fast_weights
                    group_master_weights = master_weights
                start, end = clip_slices[clip_index]
                clip_hidden = hidden_states[start:end].unsqueeze(0)
                clip_hidden, memory_input, key, target = self.apply_memory(
                    clip_hidden,
                    group_fast_weights,
                )
                if will_update:
                    fast_weights, master_weights = self.update_fast_weights(
                        memory_input,
                        key,
                        target,
                        group_fast_weights,
                        group_master_weights,
                    )
                    if state_ratio_context is not None:
                        captured_state = _CaptureNextStateGradient.apply(
                            *fast_weights,
                            *master_weights,
                            state_ratio_context,
                        )
                        fast_weights = captured_state[:3]
                        master_weights = captured_state[3:]
                outputs.append(clip_hidden.squeeze(0))
                clip_index += 1
        if clip_index != len(clip_slices):
            raise ValueError(f"Consumed {clip_index} clips, but layout contains {len(clip_slices)}")
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        clip_slices: list[tuple[int, int]],
        video_clip_counts: list[int],
    ) -> torch.Tensor:
        hidden_states = self.forward_attention(hidden_states, cu_seqlens, rope_freqs_cis)
        hidden_states = self.forward_memory_scan(hidden_states, clip_slices, video_clip_counts)
        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        return residual + hidden_states


class VideoChat3LACTVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict):
        super().__init__()
        self.rope_2d = Rope2DPosEmb(block_cfg["hidden_dim"] // block_cfg["num_heads"], 1024, 1024)
        self.blocks = nn.ModuleList([VideoChat3LACTVisionLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
        video_clip_counts: list[int],
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws)
        lengths = (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).tolist()
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + int(length))
        clip_slices = list(zip(offsets[:-1], offsets[1:]))
        cu_seqlens = torch.tensor(offsets, device=hidden_states.device, dtype=torch.int32)
        if sum(video_clip_counts) != len(clip_slices):
            raise ValueError(f"video_clip_counts={video_clip_counts} do not cover {len(clip_slices)} clips")
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis,
                clip_slices,
                video_clip_counts,
            )
        return self.final_layernorm(hidden_states)


class VideoChat3LACTVisionModel(VideoChat3VisionPreTrainedModel):
    config_class = VideoChat3LACTVisionConfig
    _no_split_modules = ["VideoChat3LACTVisionLayer"]

    def __init__(self, config: VideoChat3LACTVisionConfig):
        super().__init__(config)
        self.config = config
        self.patch_embed = VideoChat3VisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            max_clip_length=config.temporal_merge_size,
        )
        self.encoder = VideoChat3LACTVisionEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": ACT2FN["gelu_pytorch_tanh"],
                "attn_bias": True,
                "attn_impl": config.attn_impl,
                "fw_inter_multi": config.fw_inter_multi,
                "fw_num_heads": config.fw_num_heads,
                "fw_base_lr": config.fw_base_lr,
                "fw_muon_update_steps": config.fw_muon_update_steps,
                "memory_type": config.memory_type,
                "inner_optim": config.inner_optim,
                "fw_share_proj": config.fw_share_proj,
                "fw_share_init": config.fw_share_init,
                "fw_norm_epsilon": config.fw_norm_epsilon,
                "clip_ns_grad_ratio": config.clip_ns_grad_ratio,
                "recompute_ns5_backward": config.recompute_ns5_backward,
                "clip_state_grad_ratio": config.clip_state_grad_ratio,
                "lact_inference_state_mode": config.lact_inference_state_mode,
            },
        )
        self.post_init()
        self.reset_lact_parameters()

    @torch.no_grad()
    def reset_lact_parameters(self) -> None:
        for block in self.encoder.blocks:
            block.reset_lact_parameters()

    @staticmethod
    def split_grid_thws_clip_by_clip_with_counts(
        grid_thws: torch.Tensor, temporal_merge_size: int
    ) -> tuple[torch.Tensor, list[int]]:
        split_grids = []
        video_clip_counts = []
        for time, height, width in grid_thws.tolist():
            if time <= 0:
                raise ValueError(f"Temporal grid size must be positive, got {time}")
            remaining = time
            clip_count = 0
            while remaining > 0:
                clip_time = min(remaining, temporal_merge_size)
                split_grids.append([clip_time, height, width])
                remaining -= clip_time
                clip_count += 1
            video_clip_counts.append(clip_count)
        return (
            torch.tensor(split_grids, device=grid_thws.device, dtype=grid_thws.dtype),
            video_clip_counts,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor):
        split_grid_thws, video_clip_counts = self.split_grid_thws_clip_by_clip_with_counts(
            grid_thws, self.config.temporal_merge_size
        )
        hidden_states = self.patch_embed(pixel_values, split_grid_thws)
        hidden_states = self.encoder(hidden_states, split_grid_thws, video_clip_counts)
        chunk_outputs = patch_merger(
            hidden_states,
            split_grid_thws,
            merge_kernel_size=self.config.merge_kernel_size,
        )
        return _compress_chunk_outputs(
            chunk_outputs,
            video_clip_counts,
            self.config.macro_temporal_compression_factor,
            mode=getattr(self.config, "macro_temporal_compression_mode", "auto"),
        )

    def set_lact_inference_state_mode(self, mode: str) -> None:
        if mode not in ("continuous", "reset_state"):
            raise ValueError(f"Unsupported LACT inference state mode: {mode!r}")
        self.config.lact_inference_state_mode = mode
        for block in self.encoder.blocks:
            block.lact_inference_state_mode = mode


class VideoChat3LACTModel(VideoChat3Model):
    config_class = VideoChat3LACTConfig

    def __init__(self, config: VideoChat3LACTConfig):
        VideoChat3PreTrainedModel.__init__(self, config)
        self.vision_tower = VideoChat3LACTVisionModel._from_config(config.vision_config)
        self.multi_modal_projector = VideoChat3MultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config, trust_remote_code=True)
        self.post_init()


class VideoChat3LACTForConditionalGeneration(VideoChat3ForConditionalGeneration):
    config_class = VideoChat3LACTConfig

    def __init__(self, config: VideoChat3LACTConfig):
        VideoChat3PreTrainedModel.__init__(self, config)
        self.model = VideoChat3LACTModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()


__all__ = [
    "FastWeightSwiGLU",
    "VideoChat3LACTForConditionalGeneration",
    "VideoChat3LACTModel",
    "VideoChat3LACTVisionModel",
]
