import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_videochat3_base_visual_token_perturbations import (
    feature_metrics,
    random_perturbation,
    sinusoidal_group_encoding,
    sinusoidal_temporal_perturbation,
)


def test_random_perturbation_is_deterministic_and_calibrated():
    reference = torch.arange(1, 1 + 40 * 16, dtype=torch.float32).view(40, 16)
    first = random_perturbation(reference, relative_l2=0.15, seed=42)
    second = random_perturbation(reference, relative_l2=0.15, seed=42)
    other = random_perturbation(reference, relative_l2=0.15, seed=43)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert feature_metrics(reference, first)["relative_l2_delta"] == pytest.approx(
        0.15, abs=1e-6
    )


def test_sinusoidal_encoding_is_centered_across_groups():
    encoding = sinusoidal_group_encoding(8, 17, device=torch.device("cpu"))
    assert encoding.shape == (8, 17)
    assert torch.allclose(encoding.mean(dim=0), torch.zeros(17), atol=1e-6)
    assert not torch.equal(encoding[0], encoding[1])


def test_sinusoidal_perturbation_is_group_shared_and_calibrated():
    generator = torch.Generator().manual_seed(9)
    reference = torch.randn((32, 18), generator=generator)
    candidate, groups, tokens_per_group = sinusoidal_temporal_perturbation(
        reference,
        grid_time=16,
        frame_group_size=4,
        relative_l2=0.15,
    )

    assert groups == 4
    assert tokens_per_group == 8
    delta = candidate - reference
    for group in range(groups):
        section = delta[group * tokens_per_group : (group + 1) * tokens_per_group]
        assert torch.allclose(section, section[:1].expand_as(section), atol=1e-6)
    assert not torch.allclose(delta[:tokens_per_group], delta[tokens_per_group : 2 * tokens_per_group])
    assert feature_metrics(reference, candidate)["relative_l2_delta"] == pytest.approx(
        0.15, abs=1e-6
    )
