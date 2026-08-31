import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_videochat3_v4_fw_residual_sweep import (
    ALPHAS,
    alpha_label,
    interpolate_fw_residual,
    residual_metrics,
)


def test_alpha_labels_are_stable_and_unique():
    assert [alpha_label(alpha) for alpha in ALPHAS] == [
        "alpha_m2",
        "alpha_0",
        "alpha_0p5",
        "alpha_1",
        "alpha_2",
        "alpha_4",
    ]


def test_fw_residual_interpolation_has_exact_endpoints_and_direction():
    base = torch.arange(1, 1 + 12 * 8, dtype=torch.float32).view(12, 8)
    residual = torch.linspace(-2, 3, steps=base.numel()).view_as(base)
    v4 = base + residual

    assert interpolate_fw_residual(base, v4, 0.0) is base
    assert interpolate_fw_residual(base, v4, 1.0) is v4
    assert torch.allclose(interpolate_fw_residual(base, v4, -2.0), base - 2 * residual)
    assert torch.allclose(interpolate_fw_residual(base, v4, 0.5), base + 0.5 * residual)
    assert torch.allclose(interpolate_fw_residual(base, v4, 4.0), base + 4 * residual)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_residual_metrics_recovers_alpha(alpha):
    generator = torch.Generator().manual_seed(7)
    base = torch.randn((32, 16), generator=generator)
    v4 = base + 0.05 * torch.randn((32, 16), generator=generator)
    candidate = interpolate_fw_residual(base, v4, alpha)
    metrics = residual_metrics(base, v4, candidate)
    assert metrics["realized_alpha_projection"] == pytest.approx(alpha, abs=1e-5)
