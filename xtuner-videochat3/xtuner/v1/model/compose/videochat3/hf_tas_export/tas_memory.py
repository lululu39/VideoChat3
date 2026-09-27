"""LVSM fixed token states; no fast weights, inner optimizer, or detached history.

Reference: Zefan-Cai/ttt_lvsm yibo_dev f34753d6, memory_block_tokens.py.
This module is shared verbatim by training and standalone HF exports.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def rms(x, weight=None, eps=1e-5):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms(x, self.weight)


def assigned_write(q, k, v):
    """Token softmax over slots, followed by per-slot mass normalization."""
    scores = (k.transpose(1, 2) @ q.transpose(1, 2).transpose(-1, -2)) / math.sqrt(q.shape[-1])
    assignment = scores.float().softmax(-1)
    weights = assignment.transpose(-1, -2)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
    return (weights.to(v.dtype) @ v.transpose(1, 2)).transpose(1, 2)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, assigned=False):
        super().__init__()
        self.heads = heads
        self.assigned = assigned
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, context, key_context=None):
        key_context = context if key_context is None else key_context
        q = self.q_proj(query).unflatten(-1, (self.heads, -1))
        k = self.k_proj(key_context).unflatten(-1, (self.heads, -1))
        v = self.v_proj(context).unflatten(-1, (self.heads, -1))
        if self.assigned:
            out = assigned_write(q, k, v)
        elif q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
            ).transpose(1, 2)
        return self.o_proj(out.flatten(-2))


class MemoryMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 2 * dim, bias=False)
        self.up = nn.Linear(dim, 2 * dim, bias=False)
        self.down = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TimeEncoding(nn.Module):
    """Absolute seconds and normalized progress enter writer K AND V.

Only observed frame timestamps and the media duration are inputs; never labels.
Absolute periods range from 1 to 1024 seconds; relative periods from 1 to 1/128.
"""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(39, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, times):
        seconds, duration = times.float().unbind(-1)
        relative = seconds / duration.clamp_min(1e-6)
        absolute_phase = seconds[:, None] * (2 * math.pi / 2 ** torch.arange(11, device=times.device))
        relative_phase = relative[:, None] * (2 * math.pi * 2 ** torch.arange(8, device=times.device))
        features = torch.cat((absolute_phase.sin(), absolute_phase.cos(),
                              relative_phase.sin(), relative_phase.cos(),
                              duration.log1p()[:, None]), -1)
        return self.proj(features.to(self.proj[0].weight.dtype))


class TASReader(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.state_norm = RMSNorm(dim)
        self.attn = CrossAttention(dim, heads)

    def forward(self, z, old, identity):
        h = self.input_norm(z)
        state = self.state_norm(old)
        return self.attn(h.unsqueeze(0), state, key_context=state + identity).squeeze(0)


class TASMemory(nn.Module):
    """One shared bank/writer, independent readers; last-layer WFL before read."""
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.ema_init = config.tas_ema_init
        self.initial_state = nn.Parameter(torch.empty(config.tas_max_memory_tokens, dim))
        self.slot_embedding = nn.Parameter(torch.empty_like(self.initial_state))
        # Final-layer read/MLP cannot affect a pre-read WFL bank-only output.
        self.readers = nn.ModuleList(TASReader(dim, config.num_attention_heads)
                                     for _ in range(config.num_hidden_layers - 1))
        self.wfl_norm = RMSNorm(dim)
        self.write_norm = RMSNorm(dim)
        self.write = CrossAttention(dim, config.num_attention_heads, assigned=True)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MemoryMLP(dim)
        self.gate_weight = nn.Parameter(torch.zeros(1, 2 * dim))
        self.gate_bias = nn.Parameter(torch.full((1,), math.log(config.tas_ema_init / (1-config.tas_ema_init))))
        self.time_encoding = TimeEncoding(dim) if config.tas_time_encoding else None
        self.collect_metrics = False
        self.last_gate = None
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.initial_state, std=0.02)
        with torch.no_grad():
            self.slot_embedding.copy_(self.initial_state)
            self.gate_weight.zero_()
            self.gate_bias.fill_(math.log(self.ema_init / (1-self.ema_init)))

    def start(self, count, dtype):
        if not 0 < count <= self.initial_state.shape[0]:
            raise ValueError(f"TAS requests {count} slots, capacity is {self.initial_state.shape[0]}")
        return self.initial_state[:count].to(dtype).unsqueeze(0), rms(self.slot_embedding[:count].to(dtype))

    def update(self, h, old, identity, frame_times, spatial_tokens):
        if self.time_encoding is not None:
            if frame_times is None:
                raise ValueError("TAS time encoding requires actual sampled frame timestamps")
            h = h + self.time_encoding(frame_times).to(h.dtype).repeat_interleave(spatial_tokens, 0)
        u = self.write(self.write_norm(old) + identity, h.unsqueeze(0))
        candidate = u + self.mlp(self.mlp_norm(u))
        with torch.autocast(device_type=old.device.type, enabled=False):
            gate = F.linear(torch.cat((rms(old.float()), rms(u.float())), -1),
                            self.gate_weight.float(), self.gate_bias.float()).sigmoid()
            updated = (1 - gate) * old.float() + gate * candidate.float()
        if self.collect_metrics:
            self.last_gate = gate.detach()
        return updated.to(old.dtype)
