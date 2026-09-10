from types import SimpleNamespace

import pytest
import torch

from xtuner.v1.config import LRConfig
from xtuner.v1.train.trainer import Trainer


def build(stop, horizon=None):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-5)
    trainer = SimpleNamespace(_engine=SimpleNamespace(optimizer=optimizer))
    config = LRConfig(lr_type="cosine", warmup_ratio=0.03, lr_min=1e-6,
                      schedule_steps=horizon)
    return optimizer, Trainer.build_lr_scheduler(trainer, config, stop)


def test_stop_at_800_preserves_v35_schedule_and_resume():
    base_optim, base = build(5158)
    short_optim, short = build(800, 5158)
    for step in range(800):
        assert short.get_last_lr() == base.get_last_lr()
        if step == 154:
            assert short.get_last_lr() == [2e-5]
        base_optim.step()
        short_optim.step()
        base.step()
        short.step()
        if step == 399:
            state = short.state_dict()
            optimizer_state = short_optim.state_dict()
            short_optim, short = build(800, 5158)
            short_optim.load_state_dict(optimizer_state)
            short.load_state_dict(state)
    assert short.get_last_lr() == base.get_last_lr()
    assert short.get_last_lr()[0] > 1.9e-5


def test_invalid_schedule_horizon():
    with pytest.raises(ValueError, match="positive"):
        build(800, 0)
