"""State semantics, timestamp transport, HF export and causal generation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from xtuner.v1.model.compose.videochat3.hf_tas_export import prepare_tas_assets
from xtuner.v1.model.compose.videochat3.hf_tas_export.tas_memory import TASMemory, assigned_write

BASE = Path("/mnt/localssd/VideoChat3/VideoChat3-4B")


@pytest.fixture(scope="module")
def hf(tmp_path_factory):
    root = tmp_path_factory.mktemp("tas")
    prepare_tas_assets(BASE, root, hidden_size=32, intermediate_size=64,
                       num_attention_heads=4, num_hidden_layers=2,
                       init_pos_emb_height=4, init_pos_emb_width=4,
                       tas_max_memory_tokens=64, attn_impl="eager")
    config = AutoConfig.from_pretrained(root, trust_remote_code=True)
    config.text_config.hidden_size = 32
    config.text_config.intermediate_size = 64
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention"] * 2
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.head_dim = 8
    config.text_config._attn_implementation = "eager"
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], root)
    torch.manual_seed(42)
    model = cls(config).float()
    return root, model


def times(frames):
    return torch.stack((torch.arange(frames).float()/2, torch.full((frames,), frames/2)), -1)


def test_assigned_write_formula_and_gradients():
    torch.manual_seed(1)
    q, k, v = [torch.randn(2, n, 2, 4, requires_grad=True) for n in (3, 5, 5)]
    actual = assigned_write(q, k, v)
    logits = torch.einsum("bnhd,bshd->bhns", k, q) / 2
    mass = torch.softmax(logits, -1)
    expected = torch.einsum("bhns,bnhd->bshd", mass / (mass.sum(-2, keepdim=True)+1e-6), v)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_lvsm_ema_and_slot_identity():
    cfg = SimpleNamespace(hidden_size=16, num_attention_heads=2, num_hidden_layers=2,
                          tas_max_memory_tokens=8, tas_ema_init=.1, tas_time_encoding=False)
    m = TASMemory(cfg)
    old, identity = m.start(8, torch.float32)
    h = torch.randn(12, 16)
    u = m.write(m.write_norm(old)+identity, h[None])
    candidate = u + m.mlp(m.mlp_norm(u))
    actual = m.update(h, old, identity, None, 4)
    torch.testing.assert_close(actual, .9*old+.1*candidate)
    assert torch.equal(m.initial_state, m.slot_embedding)
    assert m.initial_state.data_ptr() != m.slot_embedding.data_ptr()


def test_terminal_write_history_gradient_checkpoint_and_reset(hf):
    _, model = hf
    vision = model.model.vision_tower
    vision.train()
    grid = torch.tensor([[9, 2, 2]])
    x = torch.randn(36, 3*14**2, requires_grad=True)
    vision.checkpoint_chunks = False
    a = vision(x, grid, times(9))[0]
    assert a.shape == (16, 4, 32)  # one FULL chunk bank, even with a short tail
    grad = torch.autograd.grad(a.square().sum(), x)[0]
    assert grad[:16].abs().sum() > 0 and grad[-4:].abs().sum() > 0
    vision.checkpoint_chunks = True
    b = vision(x, grid, times(9))[0]
    grad_b = torch.autograd.grad(b.square().sum(), x)[0]
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(grad, grad_b)
    vision.state_mode = "reset_state"
    c = vision(x, grid, times(9))[0]
    g = torch.autograd.grad(c.square().sum(), x)[0]
    assert g[:-4].count_nonzero() == 0 and g[-4:].abs().sum() > 0
    assert not torch.allclose(a, c)
    vision.state_mode = "continuous"


def test_video_isolation_and_time_content(hf):
    _, model = hf
    v = model.model.vision_tower.eval()
    x = torch.randn(8*4, 3*14**2)
    single = v(x, torch.tensor([[8, 2, 2]]), times(8))[0]
    joint = v(torch.cat((x, x)), torch.tensor([[8, 2, 2], [8, 2, 2]]), torch.cat((times(8), times(8))))
    torch.testing.assert_close(single, joint[0])
    torch.testing.assert_close(single, joint[1])
    shifted = times(8); shifted[:, 0] += 1
    changed = v(x, torch.tensor([[8, 2, 2]]), shifted)[0]
    assert not torch.allclose(single, changed)


def test_processor_true_times_and_final_slot_count(hf):
    root, _ = hf
    processor = AutoProcessor.from_pretrained(root, trust_remote_code=True)
    video = np.zeros((8, 28, 28, 3), dtype=np.uint8)
    metadata = dict(total_num_frames=100, fps=10., duration=10., width=28, height=28,
                    frames_indices=[0, 2, 5, 9, 18, 40, 70, 99], video_backend="decord")
    inputs = processor(text="<|vision_start|><|video_pad|><|vision_end|> Question",
                       videos=[video], video_metadata=[metadata], do_sample_frames=False,
                       size={"shortest_edge": 784, "longest_edge": 784}, return_tensors="pt")
    assert inputs["video_grid_thw"].tolist() == [[8, 2, 2]]
    assert int((inputs["input_ids"] == processor.video_token_id).sum()) == 16
    torch.testing.assert_close(inputs["tas_frame_times"][:, 0], torch.tensor(metadata["frames_indices"])/10)
    assert "seconds" not in processor.tokenizer.decode(inputs["input_ids"][0])
    assert "tas_frame_times" in processor.model_input_names


def test_full_model_generation_and_roundtrip(hf, tmp_path):
    root, model = hf
    model.eval()
    ids = torch.tensor([[42] + [model.config.video_token_id]*16 + [43]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  pixel_values_videos=torch.randn(8*4, 3*14**2),
                  video_grid_thw=torch.tensor([[8, 2, 2]]), tas_frame_times=times(8))
    with torch.no_grad():
        logits = model(**inputs, logits_to_keep=1).logits
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
    assert generated.shape[-1] == ids.shape[-1]+2
    prepare_tas_assets(root, tmp_path)
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(logits, loaded(**inputs, logits_to_keep=1).logits)
