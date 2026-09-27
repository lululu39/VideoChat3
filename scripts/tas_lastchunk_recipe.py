"""Exact v19/v29 pack grouping and square-weighted update recipe for TAS."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset
from xtuner.v1.datasets.packing import ExpandSoftPackDataset
from xtuner.v1.datasets.sampler import LengthGroupedSampler

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CACHE = ROOT / "xtuner-videochat3/dataset_cache/cache_videochat3_4B_lact_linear16_delta_3drope_parallel_gate0_lastchunk_timelens_random12624_s1k_v19/3144d8b00136f2a5/4b53522cc69ad397_ef46db3751d8e999/num_tokens.npy"


class TokenLengths(Dataset):
    def __init__(self, values):
        self.num_tokens = values

    def __len__(self):
        return len(self.num_tokens)


class Mesh:
    def __init__(self, rank, world):
        self.rank, self.world = rank, world

    def get_local_rank(self):
        return self.rank

    def size(self):
        return self.world


def build_plan(cache=LEGACY_CACHE, seed=42, world=8, global_packs=16):
    lengths = np.load(cache)
    if len(lengths) != 12624 or not ((lengths > 0) & (lengths <= 1024)).all():
        raise ValueError("Expected the complete v19/v29 12,624-row last-chunk token cache")
    packed = ExpandSoftPackDataset([TokenLengths(lengths)], pack_max_length=1024,
                                   global_pack=False, pack_extra_buffer_size=20,
                                   pack_chunk_size=10000, pack_workers=1, seed=seed)
    if len(packed) != 1815:
        raise ValueError(f"Expected 1,815 v19/v29 packs, got {len(packed)}")
    rank_packs = [list(LengthGroupedSampler(packed, global_packs, Mesh(rank, world), seed=seed))
                  for rank in range(world)]
    local_packs = global_packs//world
    steps = []
    for start in range(0, len(rank_packs[0]), local_packs):
        steps.append([sum((packed.pack_infos[index]["indices"]
                           for index in order[start:start+local_packs]), []) for order in rank_packs])
    counts = [sum(map(len, step)) for step in steps]
    flat = [i for step in steps for rank in step for i in rank]
    assert set(flat) == set(range(12624)) and len(steps) == 114
    return dict(reference="v19/v29", cache=str(cache), cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
                seed=seed, world=world, global_packs=global_packs, unique_packs=len(packed),
                padded_packs=len(rank_packs[0])*world, total_steps=len(steps),
                unique_examples=len(set(flat)), example_occurrences=len(flat),
                examples_per_step=dict(min=min(counts), max=max(counts), mean=float(np.mean(counts))),
                rank_pack_order=rank_packs, steps=steps)


def aligned_lr(step, total=114, peak=2e-5, floor=1e-6):
    """Zero-indexed optimizer update; matches XTuner SequentialLR exactly."""
    warmup = int(.03*total)
    if step < warmup:
        return peak*step/warmup
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def square_denominator(lengths):
    return torch.as_tensor(lengths, dtype=torch.float32).sqrt().sum()
