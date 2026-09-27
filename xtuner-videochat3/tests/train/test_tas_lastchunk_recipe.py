"""Check reference packing, global square gradients, schedule and native labels."""
import json
from pathlib import Path
import sys
import math

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"scripts"))
from tas_lastchunk_recipe import build_plan, aligned_lr, square_weight


def test_exact_reference_plan():
    plan = build_plan()
    assert plan["unique_packs"] == 1815
    assert plan["padded_packs"] == 1824
    assert plan["total_steps"] == 114
    assert plan["example_occurrences"] == 12687
    assert plan["steps"][0][0] == [875, 11153, 6878, 11364, 11457, 6009, 11926,
                                     8057, 5594, 11854, 2738, 7573, 4123, 11356]
    assert plan["examples_per_step"]["min"] == 102
    assert plan["examples_per_step"]["max"] == 112


def test_square_gradient_matches_xtuner_global_denominator():
    from xtuner.v1.loss import CELossConfig
    from xtuner.v1.loss.ce_loss import CELossContext
    from xtuner.v1.loss.ce_loss import CELossContextInputItem
    lengths = [3, 7, 4, 9]
    labels = torch.cat([torch.ones(n, dtype=torch.long) for n in lengths])[None]
    boundaries = torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    kwargs = CELossContext.build_batches_loss_kwargs(
        [CELossContextInputItem(shifted_labels=labels)], CELossConfig(loss_reduction="square"),
        cu_seq_lens_list=[boundaries])[0]
    x = torch.linspace(.2, 1.8, sum(lengths), requires_grad=True)
    expected = (x*kwargs.loss_weight.flatten()).sum()
    denominator = sum(math.sqrt(n) for n in lengths)
    # Simulate two DDP ranks with different microbatch counts: 1 versus 3.
    pieces = x.split(lengths)
    local_a = pieces[0].mean()*square_weight(lengths[0],denominator,2)
    local_b = sum(t.mean()*square_weight(n,denominator,2) for t,n in zip(pieces[1:], lengths[1:]))
    actual = (local_a+local_b)/2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])


def test_lr_matches_xtuner_sequential_schedule():
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5, betas=(.9,.95), foreach=False)
    warm = LambdaLR(optimizer, lambda s:s/3 if s<3 else 1)
    decay = CosineAnnealingLR(optimizer, T_max=111, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warm,decay], milestones=[3])
    for step in range(114):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(aligned_lr(step),abs=1e-15)
        optimizer.step(); scheduler.step()


def test_native_answer_labels_on_real_sample():
    from train_videochat3_tas import TimeLensDataset
    from xtuner.v1.data_proto.messages import ChatMessages
    from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
    d=TimeLensDataset('/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json',
                      '/mnt/localssd/VideoChat3/VideoChat3-4B-TAS-init', max_frames=64, limit=1, native_template=True)
    item=d[0]
    original=ChatMessages(messages=[{'role':'user','content':'<VIDEO_CONTEXT>\nquestion'},d.rows[0]['messages'][1]])
    reference=original.tokenize(d.processor.tokenizer, CHAT_TEMPLATE_MAP['videochat3'])
    expected=[x for x in reference['labels'] if x!=-100]
    actual=item['labels'][item['labels']!=-100].tolist()
    assert actual==expected
    assert item['input_ids'][0,-1].item()==d.processor.tokenizer.encode('\n',add_special_tokens=False)[0]
    assert item['labels'][0,-1].item()==-100
