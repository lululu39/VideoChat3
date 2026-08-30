from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xtuner.v1.config import LRConfig, VisionAdamWConfig
from xtuner.v1.engine.vision_compose_train_engine import VisionComposeTrainEngine
from xtuner.v1.train.trainer import Trainer


class FakeLACTVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.memory = nn.Linear(2, 2)
        self.memory_gate = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _is_lact_state_key(key: str) -> bool:
        return ".memory." in f".{key}" or key.endswith("memory_gate")


class FakeComposeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = FakeLACTVision()
        self.multi_modal_projector = nn.Linear(2, 2)
        self.language_model = nn.Linear(2, 2)
        self.multi_modal_projector.requires_grad_(False)
        self.language_model.requires_grad_(False)


def test_lact_optimizer_uses_disjoint_named_lr_groups():
    engine = object.__new__(VisionComposeTrainEngine)
    engine.model = FakeComposeModel()
    config = VisionAdamWConfig(
        lr=2.5e-6,
        vit_lr=2.5e-6,
        lact_lr=2e-5,
        weight_decay=0,
        foreach=False,
    )

    with patch("torch.distributed.get_rank", return_value=0):
        optimizer = engine.build_optimizer(config)

    assert [group["name"] for group in optimizer.param_groups] == ["vit", "lact_fw"]
    assert [group["lr"] for group in optimizer.param_groups] == [2.5e-6, 2e-5]
    vit_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    lact_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    expected_ids = {
        id(parameter)
        for parameter in engine.model.vision_tower.parameters()
        if parameter.requires_grad
    }
    assert vit_ids.isdisjoint(lact_ids)
    assert vit_ids | lact_ids == expected_ids
    assert id(engine.model.vision_tower.memory_gate) in lact_ids


def test_lact_memory_gate_can_use_a_separate_lr_group():
    engine = object.__new__(VisionComposeTrainEngine)
    engine.model = FakeComposeModel()
    config = VisionAdamWConfig(
        lr=2.5e-6,
        vit_lr=2.5e-6,
        lact_lr=2e-5,
        lact_gate_lr=2e-4,
        weight_decay=0,
        foreach=False,
    )

    with patch("torch.distributed.get_rank", return_value=0):
        optimizer = engine.build_optimizer(config)

    assert [group["name"] for group in optimizer.param_groups] == ["vit", "lact_fw", "lact_gate"]
    assert [group["lr"] for group in optimizer.param_groups] == [2.5e-6, 2e-5, 2e-4]
    grouped_ids = [
        {id(parameter) for parameter in group["params"]}
        for group in optimizer.param_groups
    ]
    assert all(
        grouped_ids[left].isdisjoint(grouped_ids[right])
        for left in range(len(grouped_ids))
        for right in range(left + 1, len(grouped_ids))
    )
    expected_ids = {
        id(parameter)
        for parameter in engine.model.vision_tower.parameters()
        if parameter.requires_grad
    }
    assert set().union(*grouped_ids) == expected_ids
    assert grouped_ids[-1] == {id(engine.model.vision_tower.memory_gate)}


def test_group_relative_cosine_schedule_preserves_lr_ratio_and_floors():
    vit = nn.Parameter(torch.zeros(1))
    lact = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [vit], "lr": 2.5e-6, "name": "vit"},
            {"params": [lact], "lr": 2e-5, "name": "lact_fw"},
        ]
    )
    trainer = object.__new__(Trainer)
    trainer._engine = SimpleNamespace(optimizer=optimizer)
    scheduler = trainer.build_lr_scheduler(
        LRConfig(lr_type="cosine", warmup_ratio=0.1, lr_min=0, lr_min_ratio=0.05),
        scheduler_step=20,
    )

    observed = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr())

    for vit_lr, lact_lr in observed:
        assert lact_lr / vit_lr == pytest.approx(8.0)
    assert observed[-1][0] == pytest.approx(1.25e-7)
    assert observed[-1][1] == pytest.approx(1e-6)


def test_group_relative_cosine_schedule_preserves_gate_multiplier():
    lact = nn.Parameter(torch.zeros(1))
    gate = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [lact], "lr": 2e-5, "name": "lact_fw"},
            {"params": [gate], "lr": 2e-4, "name": "lact_gate"},
        ]
    )
    trainer = object.__new__(Trainer)
    trainer._engine = SimpleNamespace(optimizer=optimizer)
    scheduler = trainer.build_lr_scheduler(
        LRConfig(lr_type="cosine", warmup_ratio=0.1, lr_min=0, lr_min_ratio=0.05),
        scheduler_step=20,
    )

    observed = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr())

    for lact_lr, gate_lr in observed:
        assert gate_lr / lact_lr == pytest.approx(10.0)
    assert observed[-1][0] == pytest.approx(1e-6)
    assert observed[-1][1] == pytest.approx(1e-5)
