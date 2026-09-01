import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from xtuner.v1.model.compose.videochat3.modeling_vision_lact import (
    FastWeightSwiGLU,
    VideoChat3VisionLACTModel,
    _BoundPreviousStateGradient,
    _CaptureNextStateGradient,
    _StateGradRatioContext,
    zeropower_via_newtonschulz5,
)
from xtuner.v1.model.compose.videochat3.videochat3_config import (
    VideoChat3LACTVisionConfig,
    VideoChat3VisionConfig,
)


def _vision_kwargs():
    return {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "patch_size": 2,
        "merge_kernel_size": [2, 2],
        "temporal_merge_size": 4,
        "init_pos_emb_height": 2,
        "init_pos_emb_width": 2,
        "attn_impl": "eager_attention",
    }


def _flatten_outputs(outputs):
    return torch.cat([output.flatten() for output in outputs])


def _copy_baseline_weights(baseline, lact):
    missing, unexpected = lact.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert missing
    for block in lact.encoder.blocks:
        block.reset_lact_parameters()


def _ns5_vjp(matrix, cotangent, *, clip_ns_grad_ratio):
    matrix = matrix.clone().requires_grad_(True)
    update = zeropower_via_newtonschulz5(
        matrix,
        clip_ns_grad_ratio=clip_ns_grad_ratio,
    )
    (gradient,) = torch.autograd.grad(update, matrix, cotangent)
    return update.detach(), gradient


def _state_tensors(*, requires_grad=True):
    shapes = (
        (2, 3, 4, 5),
        (2, 3, 5, 4),
        (2, 3, 4, 5),
        (2, 3, 4, 5),
        (2, 3, 5, 4),
        (2, 3, 4, 5),
    )
    return tuple(torch.randn(shape, requires_grad=requires_grad) for shape in shapes)


def _joint_state_norm(tensors):
    return torch.sqrt(
        sum(
            torch.linalg.vector_norm(tensor.float(), dim=(-2, -1)).square()
            for tensor in tensors
        )
    )


def _state_ratio_vjp(previous_state, next_cotangents, direct_cotangents, multiplier):
    ratio_context = _StateGradRatioContext()
    bounded_previous = _BoundPreviousStateGradient.apply(
        *previous_state,
        ratio_context,
    )
    next_state = _CaptureNextStateGradient.apply(
        *(multiplier * tensor for tensor in bounded_previous),
        ratio_context,
    )
    loss = sum(
        (tensor * cotangent).sum()
        for tensor, cotangent in zip(next_state, next_cotangents)
    )
    loss = loss + sum(
        (tensor * cotangent).sum()
        for tensor, cotangent in zip(bounded_previous, direct_cotangents)
    )
    return next_state, torch.autograd.grad(loss, previous_state)


def _reference_update(memory, key, target, learning_rates, fast_weights, master_weights, steps):
    w0, w1, w2 = fast_weights
    master_w0, master_w1, master_w2 = master_weights
    batch_size, seq_len, _ = key.shape
    output, key_heads, gate, up, hidden = memory._apply_fast_weights(key, fast_weights)
    error = (target.float() - output.float()) / seq_len
    error_heads = error.reshape(batch_size, seq_len, memory.num_heads, memory.head_dim)
    d_hidden = torch.einsum("blhd,bhdk->blhk", error_heads, w1.float().transpose(-1, -2))
    d_up = d_hidden * F.silu(gate)
    d_gate = d_hidden * up
    sigmoid = torch.sigmoid(gate)
    d_gate_pre = d_gate * sigmoid * (1 + gate * (1 - sigmoid))
    lr0, lr1, lr2 = learning_rates.float().split(1, dim=-1)

    w0_grad = torch.einsum("blhd,blhk->bhdk", key_heads.float() * lr0.unsqueeze(2), d_gate_pre.float())
    w1_grad = torch.einsum("blhk,blhd->bhkd", hidden.float() * lr1.unsqueeze(2), error_heads.float())
    w2_grad = torch.einsum("blhd,blhk->bhdk", key_heads.float() * lr2.unsqueeze(2), d_up.float())

    def muon(gradient):
        shape = gradient.shape
        return zeropower_via_newtonschulz5(gradient.flatten(0, 1), steps).reshape(shape)

    master_weights = (
        master_w0 + muon(w0_grad).to(master_w0.dtype),
        master_w1 + muon(w1_grad).to(master_w1.dtype),
        master_w2 + muon(w2_grad).to(master_w2.dtype),
    )
    fast_weights = tuple(F.normalize(weight, dim=2, eps=1e-5).to(w0.dtype) for weight in master_weights)
    return fast_weights, master_weights


def test_ns5_ratio_clip_preserves_forward_and_bounds_each_matrix_vjp():
    torch.manual_seed(1)
    matrix = torch.randn(3, 5, 7) * 1e-3
    cotangent = torch.randn_like(matrix)

    exact_update, exact_gradient = _ns5_vjp(matrix, cotangent, clip_ns_grad_ratio=False)
    clipped_update, clipped_gradient = _ns5_vjp(matrix, cotangent, clip_ns_grad_ratio=True)
    torch.testing.assert_close(clipped_update, exact_update, rtol=0, atol=0)

    actual_cotangent = cotangent.to(clipped_update.dtype)
    input_norm = torch.linalg.vector_norm(actual_cotangent.float(), dim=(-2, -1))
    clipped_norm = torch.linalg.vector_norm(clipped_gradient.float(), dim=(-2, -1))
    assert torch.all(clipped_norm <= input_norm * (1 + 1e-6))

    exact_norm = torch.linalg.vector_norm(exact_gradient.float(), dim=(-2, -1), keepdim=True)
    input_norm = input_norm[:, None, None]
    expected_scale = torch.where(
        exact_norm > input_norm,
        input_norm / exact_norm.clamp_min(1e-12),
        torch.ones_like(exact_norm),
    )
    torch.testing.assert_close(clipped_gradient, exact_gradient * expected_scale, rtol=1e-6, atol=1e-7)


def test_ns5_ratio_clip_is_exact_below_rho_and_handles_zero_cotangent():
    torch.manual_seed(2)
    matrix = torch.randn(2, 4, 6)
    cotangent = torch.randn_like(matrix)
    _, exact_gradient = _ns5_vjp(matrix, cotangent, clip_ns_grad_ratio=False)
    _, clipped_gradient = _ns5_vjp(matrix, cotangent, clip_ns_grad_ratio=True)
    torch.testing.assert_close(clipped_gradient, exact_gradient, rtol=0, atol=0)

    _, zero_gradient = _ns5_vjp(matrix, torch.zeros_like(matrix), clip_ns_grad_ratio=True)
    assert torch.count_nonzero(zero_gradient).item() == 0


def test_ns5_ratio_clip_supports_checkpoint_recomputation():
    torch.manual_seed(4)
    cotangent = torch.randn(2, 4, 6).bfloat16()
    for use_reentrant in (True, False):
        matrix = (torch.randn(2, 4, 6) * 1e-3).requires_grad_(True)
        update = checkpoint(
            lambda value: zeropower_via_newtonschulz5(
                value,
                clip_ns_grad_ratio=True,
            ),
            matrix,
            use_reentrant=use_reentrant,
        )
        (update * cotangent).sum().backward()
        input_norm = torch.linalg.vector_norm(cotangent.float(), dim=(-2, -1))
        gradient_norm = torch.linalg.vector_norm(matrix.grad.float(), dim=(-2, -1))
        assert torch.all(gradient_norm <= input_norm * (1 + 1e-6))


def test_state_ratio_clip_preserves_forward_and_bounds_full_previous_adjoint():
    torch.manual_seed(5)
    previous_state = _state_tensors()
    next_cotangents = tuple(torch.randn_like(tensor) for tensor in previous_state)
    direct_cotangents = tuple(torch.randn_like(tensor) for tensor in previous_state)
    next_state, previous_gradients = _state_ratio_vjp(
        previous_state,
        next_cotangents,
        direct_cotangents,
        multiplier=4.0,
    )

    for actual, previous in zip(next_state, previous_state):
        torch.testing.assert_close(actual, 4.0 * previous, rtol=0, atol=0)
    next_norm = _joint_state_norm(next_cotangents)
    previous_norm = _joint_state_norm(previous_gradients)
    assert torch.all(previous_norm <= next_norm * (1 + 1e-6))

    exact_gradients = tuple(
        4.0 * next_cotangent + direct_cotangent
        for next_cotangent, direct_cotangent in zip(
            next_cotangents,
            direct_cotangents,
        )
    )
    exact_norm = _joint_state_norm(exact_gradients)
    expected_scale = torch.where(
        exact_norm > next_norm,
        next_norm / exact_norm.clamp_min(1e-12),
        torch.ones_like(exact_norm),
    )[..., None, None]
    for actual, exact in zip(previous_gradients, exact_gradients):
        torch.testing.assert_close(
            actual,
            exact * expected_scale.to(exact.dtype),
            rtol=1e-6,
            atol=1e-6,
        )


def test_state_ratio_clip_is_exact_below_rho_and_handles_zero_next_adjoint():
    torch.manual_seed(6)
    previous_state = _state_tensors()
    next_cotangents = tuple(torch.randn_like(tensor) for tensor in previous_state)
    zero_direct = tuple(torch.zeros_like(tensor) for tensor in previous_state)
    _, previous_gradients = _state_ratio_vjp(
        previous_state,
        next_cotangents,
        zero_direct,
        multiplier=0.5,
    )
    for actual, cotangent in zip(previous_gradients, next_cotangents):
        torch.testing.assert_close(actual, 0.5 * cotangent, rtol=0, atol=0)

    previous_state = _state_tensors()
    zero_next = tuple(torch.zeros_like(tensor) for tensor in previous_state)
    direct_cotangents = tuple(torch.randn_like(tensor) for tensor in previous_state)
    _, previous_gradients = _state_ratio_vjp(
        previous_state,
        zero_next,
        direct_cotangents,
        multiplier=1.0,
    )
    assert all(torch.count_nonzero(gradient).item() == 0 for gradient in previous_gradients)


def test_state_ratio_clip_supports_checkpoint_recomputation():
    torch.manual_seed(8)
    next_cotangents = _state_tensors(requires_grad=False)
    for use_reentrant in (True, False):
        previous_state = _state_tensors()

        def transition(*state):
            ratio_context = _StateGradRatioContext()
            bounded_previous = _BoundPreviousStateGradient.apply(
                *state,
                ratio_context,
            )
            return _CaptureNextStateGradient.apply(
                *(3.0 * tensor for tensor in bounded_previous),
                ratio_context,
            )

        next_state = checkpoint(
            transition,
            *previous_state,
            use_reentrant=use_reentrant,
        )
        torch.autograd.backward(next_state, next_cotangents)
        next_norm = _joint_state_norm(next_cotangents)
        previous_norm = _joint_state_norm(
            tuple(tensor.grad for tensor in previous_state)
        )
        assert torch.all(previous_norm <= next_norm * (1 + 1e-6))


def test_frozen_base_vision_keeps_lact_gradients_with_nonreentrant_checkpoint():
    torch.manual_seed(9)
    lact = VideoChat3LACTVisionConfig(
        **_vision_kwargs(),
        fw_muon_update_steps=0,
    ).build()
    lact.requires_grad_(False)
    for name, parameter in lact.named_parameters():
        if lact._is_lact_state_key(name):
            parameter.requires_grad_(True)

    pixel_values = torch.randn(32, 12)
    grid_thws = torch.tensor([[8, 2, 2]], dtype=torch.int32)

    def vision_forward(values):
        return _flatten_outputs(lact(values, grid_thws))

    output = checkpoint(
        vision_forward,
        pixel_values,
        use_reentrant=False,
    )
    output.sum().backward()

    assert all(block.memory_gate.grad is not None for block in lact.encoder.blocks)
    assert all(
        torch.count_nonzero(block.memory_gate.grad).item() > 0
        for block in lact.encoder.blocks
    )
    assert all(block.wqkv.weight.grad is None for block in lact.encoder.blocks)


def test_fused_fast_weight_update_matches_reference_formula():
    torch.manual_seed(3)
    memory = FastWeightSwiGLU(dim=8, inter_multi=2, num_heads=1, share_proj=True)
    key = torch.randn(2, 7, 8)
    target = torch.randn(2, 7, 8)
    learning_rates = torch.rand(2, 7, 3) * 0.02
    fast_weights, master_weights = memory.init_fast_weights(batch_size=2)

    actual = memory.update_preprojected(
        key,
        target,
        learning_rates,
        fast_weights,
        master_weights,
        muon_update_steps=2,
    )
    expected = _reference_update(
        memory,
        key,
        target,
        learning_rates,
        fast_weights,
        master_weights,
        steps=2,
    )
    for actual_group, expected_group in zip(actual, expected):
        for actual_weight, expected_weight in zip(actual_group, expected_group):
            torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)


def test_zero_memory_gate_preserves_baseline_and_receives_gradient():
    torch.manual_seed(7)
    baseline = VideoChat3VisionConfig(**_vision_kwargs()).build()
    torch.manual_seed(11)
    lact = VideoChat3LACTVisionConfig(**_vision_kwargs(), fw_muon_update_steps=0).build()
    _copy_baseline_weights(baseline, lact)

    pixel_values = torch.randn(32, 12)
    grid_thws = torch.tensor([[8, 2, 2]], dtype=torch.int32)
    baseline_output = _flatten_outputs(baseline(pixel_values, grid_thws))
    lact_output = _flatten_outputs(lact(pixel_values, grid_thws))
    torch.testing.assert_close(lact_output, baseline_output, rtol=0, atol=0)

    lact_output.sum().backward()
    for block in lact.encoder.blocks:
        assert block.memory_gate.grad is not None
        assert torch.count_nonzero(block.memory_gate.grad).item() > 0


def test_fast_state_updates_later_clip_and_resets_at_video_boundary():
    torch.manual_seed(13)
    lact = VideoChat3LACTVisionConfig(**_vision_kwargs(), fw_muon_update_steps=0).build()
    with torch.no_grad():
        for block in lact.encoder.blocks:
            block.memory_gate.fill_(0.1)

    first_video = torch.randn(32, 12)
    second_video = torch.randn(32, 12)
    one_video_grid = torch.tensor([[8, 2, 2]], dtype=torch.int32)
    split_grid = torch.tensor([[4, 2, 2], [4, 2, 2]], dtype=torch.int32)

    joined_clips = lact(first_video, one_video_grid)
    reset_clips = lact(first_video, split_grid)
    assert not torch.allclose(joined_clips[1], reset_clips[1])

    batched = lact(
        torch.cat((first_video, second_video)),
        torch.tensor([[8, 2, 2], [8, 2, 2]], dtype=torch.int32),
    )
    separate = lact(first_video, one_video_grid) + lact(second_video, one_video_grid)
    assert len(batched) == len(separate)
    for batched_clip, separate_clip in zip(batched, separate):
        torch.testing.assert_close(batched_clip, separate_clip, rtol=1e-5, atol=1e-6)


def test_lact_config_builds_separate_vision_model():
    config = VideoChat3LACTVisionConfig(**_vision_kwargs())
    model = config.build()
    model.init_weights()
    assert config.model_type == "videochat3_lact_vision"
    assert isinstance(model, VideoChat3VisionLACTModel)
    assert config.clip_ns_grad_ratio is False
    assert config.clip_state_grad_ratio is True
    assert all(block.memory.clip_ns_grad_ratio is False for block in model.encoder.blocks)
    assert all(block.clip_state_grad_ratio is True for block in model.encoder.blocks)
    assert all(block.memory_gate.shape == (16,) for block in model.encoder.blocks)
    assert all(torch.count_nonzero(block.memory_gate).item() == 0 for block in model.encoder.blocks)

    unclipped = VideoChat3LACTVisionConfig(**_vision_kwargs(), clip_ns_grad_ratio=False).build()
    assert all(block.memory.clip_ns_grad_ratio is False for block in unclipped.encoder.blocks)

    unclipped_state = VideoChat3LACTVisionConfig(
        **_vision_kwargs(),
        clip_state_grad_ratio=False,
    ).build()
    assert all(block.clip_state_grad_ratio is False for block in unclipped_state.encoder.blocks)
