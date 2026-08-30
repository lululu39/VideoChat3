# Experiment Results

## Rules

- Keep one concise section per numbered experiment, including data, training configuration, checkpoint, parameter diagnostics, Base-vs-LACT evaluation, and conclusion.
- Reuse the fixed Base results below for the same evaluation protocol; run only the new LACT checkpoint unless the model, data, prompt, decoding, or benchmark configuration changes.
- Store native evaluation artifacts under `/mnt/localssd/VideoChat3/eval`; do not upload future evaluation results to W&B.

## Fixed Core-Eval Base

Model: `/mnt/localssd/VideoChat3/VideoChat3-4B`.

| Benchmark | Base |
|---|---:|
| Video-MME Short, 2 FPS, limit 1024 | 80.70 |
| Video-MME Long, 2 FPS, limit 1024 | 60.30 |
| MVBench MP4, 64 frames | 70.83 |
| MMBench DEV EN V1.1 | 35.91 |

## v1 - Lightweight, Full Vision Encoder

- Data: original Stage 3 lightweight manifest, 17 annotations / 30,966 references; 70.4% motion and 1.1% CinePile.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`.
- Trainable: full 775M vision encoder; LM and projector frozen.
- Training: 8xH100, global batch 16, 8K packs, up to 3,600 frames, one epoch / 208 steps, LR `2.5e-6`, cosine, 3% warmup, weight decay 0, global grad clip 1, NS5/state-gradient ratio clips at rho 1.
- Checkpoint: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1/20260828223921/hf-208`.
- Training W&B: [`v1`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1).

### Checkpoint Diagnostics

| Parameter group | Result vs initialization |
|---|---:|
| Memory gate RMS | `2.53e-5` |
| FW base relative L2 delta | `1.93e-5` |
| FW private projections relative L2 delta | `1.58e-4` |
| FW value projection relative L2 delta | `6.00e-5` |
| Original attention relative L2 delta | `7.39e-4` |
| Original MLP relative L2 delta | `3.37e-4` |
| LM / projector | Bitwise unchanged |

All 208 pre-clip global gradient norms exceeded 1. Mean loss changed from `0.633` over the first 26 steps to `0.607` over the last 26. The original ViT absorbed most useful movement while the FW residual stayed nearly closed.

### Core Evaluation

| Benchmark | Base | v1 | Delta |
|---|---:|---:|---:|
| Video-MME Short | 80.70 | 80.30 | -0.40 |
| Video-MME Long | 60.30 | 60.10 | -0.20 |
| MVBench MP4 64-frame | 70.83 | 70.97 | +0.14 |
| MMBench DEV EN V1.1 | 35.91 | 34.91 | -1.00 |

Conclusion: v1 is approximately Base parity with no long-video gain; do not repeat this optimizer/trainable-scope combination.

## v2 - Lightweight, FW Only

- Data and initialization: identical to v1 for a controlled comparison.
- Trainable: 358.47M LACT-added parameters only; original ViT, LM, and projector frozen.
- Training: 8xH100, global batch 16, 8K packs, up to 3,600 frames, one epoch / 208 steps, peak FW LR `2e-5`, minimum LR `1e-6`, cosine, 3% warmup, weight decay 0, global grad clip 1, NS5/state-gradient ratio clips at rho 1.
- Checkpoint: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-s3-lite31k-8xh100-gb16-f3600-s8k-fwlr2e5-ns5r1-stgr1-v2/20260829132333/hf-208`.
- Training W&B: [`v2`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-s3-lite31k-8xh100-gb16-f3600-s8k-fwlr2e5-ns5r1-stgr1-v2).

### Checkpoint Diagnostics

| Parameter group | Result vs initialization | v2 / v1 movement |
|---|---:|---:|
| Memory gate RMS | `4.85e-4` | 19.2x |
| FW base relative L2 delta | `7.90e-3` | 410x |
| FW private projections relative L2 delta | `1.35e-2` | 85.8x |
| FW value projection relative L2 delta | `9.42e-3` | 157x |
| FW LR projection relative L2 delta | `1.92e-3` | 62.5x |
| FW memory norm | Bitwise unchanged | unchanged |
| Original ViT / LM / projector | Bitwise unchanged | expected |

Mean pre-clip gradient norm was `0.387`, maximum `1.820`, and only 7/208 steps exceeded 1. Mean loss was `0.638` over the first 26 steps and `0.621` over the last 26. Unlike v1, v2 materially updated the FW channel; the gate is still small in absolute scale, so evaluation must determine whether that movement affects behavior.

### Core Evaluation

| Benchmark | Base | v2 | Delta |
|---|---:|---:|---:|
| Video-MME Short | 80.70 | 80.90 | +0.20 |
| Video-MME Long | 60.30 | 60.10 | -0.20 |
| MVBench MP4 64-frame | 70.83 | 70.80 | -0.03 |
| MMBench DEV EN V1.1 | 35.91 | 35.45 | -0.46 |

Native artifacts: `/mnt/localssd/VideoChat3/eval/videochat3-lact-v2-core`.

Conclusion: v2 moved the FW parameters far more than v1 and avoided almost all global clipping, but remained at Base parity and did not improve Video-MME Long. Freezing the original ViT prevented image capability from drifting as much as v1, yet the learned gate RMS of `4.85e-4` still produced only a small behavioral change. A follow-up should change the supervision mix or gate/FW optimization rather than repeat this lightweight FW-only recipe unchanged.

## v3 - Lightweight, Split ViT/FW Learning Rates

**Status:** Completed. Usable checkpoint, mixed/negative research result.

- Objective: isolate the two variables confounded between v1 and v2. Compared with v1, only the FW LR is higher; compared with v2, only the original ViT is unfrozen at the v1 LR.
- Data and initialization: identical to v1/v2, using the 17-annotation / 30,966-reference lightweight manifest and `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`.
- Trainable: the complete vision encoder; LM and projector frozen. Original ViT and LACT FW parameters are disjoint optimizer groups selected by the vision tower's canonical `_is_lact_state_key()` predicate.
- Training: 8xH100, global batch 16, 8K packs, up to 3,600 frames, one epoch / expected 208 steps, weight decay 0, 3% warmup, and proportional cosine decay to 5% of each peak LR.
- Learning rates: original ViT `2.5e-6 -> 1.25e-7`; LACT FW `2e-5 -> 1e-6`. The 8x ratio follows the chosen v1/v2 LRs and is unrelated to GPU count.
- Stabilization: unchanged from v1/v2, with rho-1 NS5/state-gradient ratio clipping and framework global grad norm 1.
- Training W&B: [`v3`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-fwlr2e5-ns5r1-stgr1-v3).
- Startup validation: FSDP retained disjoint `416.0M` original-ViT and `358.0M` LACT-FW groups. Warmup logs preserve the exact 8:1 LR ratio; first-four-step grad norms were `13.24/13.23/6.88/10.02`, peak allocated memory was 47.3GB, and no OOM or optimizer-group error occurred.
- Checkpoint: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-fwlr2e5-ns5r1-stgr1-v3/20260829215844/hf-208`.

### Checkpoint Diagnostics

| Parameter group | Result vs initialization | Relative to v2 movement |
|---|---:|---:|
| Memory gate RMS | `2.02e-4` | 41.7% |
| FW base relative L2 delta | `1.89e-3` | 24.0% |
| FW private projections relative L2 delta | `5.48e-3` | 40.6% |
| FW value projection relative L2 delta | `2.99e-3` | 31.7% |
| FW LR projection relative L2 delta | `8.49e-4` | 44.3% |
| FW memory norm | Bitwise unchanged | unchanged |
| Original attention relative L2 delta | `7.46e-4` | trained; approximately v1 |
| Original MLP relative L2 delta | `3.39e-4` | trained; approximately v1 |
| LM / projector | Bitwise unchanged | expected |

All 208 steps exceeded global grad norm 1. The median was `5.31`, mean `11.51`, and step 186 spiked to `1085.80`; without that spike the mean was `6.32`. Mean loss changed from `0.633` over the first 26 steps to `0.608` over the last 26. Raising the FW LR materially increased FW movement over v1, but joint global clipping with the unfrozen ViT left only 24%-44% of v2's FW movement.

### Core Evaluation

| Benchmark | Base | v3 | Delta |
|---|---:|---:|---:|
| Video-MME Short | 80.70 | 79.40 | -1.30 |
| Video-MME Long | 60.30 | 60.60 | +0.30 |
| MVBench MP4 64-frame | 70.83 | 70.88 | +0.05 |
| MMBench DEV EN V1.1 | 35.91 | 35.53 | -0.39 |

Native artifacts: `/mnt/localssd/VideoChat3/eval/videochat3-lact-v3-core`.

Conclusion: v3 produced the first positive Video-MME Long delta, but `+0.30` is only a small change and came with a `-1.30` Short regression; MVBench stayed at parity and MMBench remained below Base. Compared with v2, unfreezing the original ViT moved it approximately as much as v1 while shared global clipping reduced FW movement to 24%-44% of v2. This split-LR recipe therefore does not show a stable overall gain and suggests that joint global clipping lets the original ViT compete with, rather than effectively co-adapt with, the FW branch.

## Dataset Decision After v3

- Decision: stop using the official VideoChat3 Stage 3 training data and do not run the planned full-annotation experiment. v1-v3 established that the lightweight mix does not provide enough supervision that specifically rewards persistent cross-window memory, while the 142,645-row full-local mix is dominated by short/local TGIF-style tasks and is not an appropriate scale-up target.
- On 2026-08-30, the extracted 543G Stage 3 dataset at `/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data` and the 359M full-annotation repository at `/mnt/localssd/dataset/VideoChat3/VideoChat3-Training-Data-Annotations` were permanently deleted. `/mnt/localssd` free space increased from 381G to 926G.
- Retained: LACT initialization, v1-v3 HF/DCP checkpoints, training logs/W&B runs, core-evaluation artifacts, code, and these result tables.
- Future experiments will use independently sourced data selected for genuine multi-event, temporal-order, state-tracking, or long-context dependencies rather than merely long raw video duration.
