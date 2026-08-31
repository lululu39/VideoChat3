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

## v4 - VideoChat-Flash LongVid, FW Only

**Status:** Completed. Usable checkpoint, Base-parity result without a meaningful long-video gain.

- Objective: test whether explicit long-video event understanding, event relationship, and event counting supervision can open and train the LACT memory path where the retired lightweight Stage 3 mixture did not.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`; no v1-v3 checkpoint reuse.
- Data: VideoChat-Flash LongVid subset at revision `be87f5516a709be079cec8b727dd2287bf2dd70f`, converted into 5,870 QA rows over 5,478 per-dataset unique videos. Frame directories follow the official recipe: `img2`, 1 FPS semantics, 64-512 uniformly sampled frames, rounded down to a multiple of four. The converted cache contains 15,718,498 tokens, or 1,918.8 ideal 8K packs before packing overhead.
- Trainable scope: the 358.47M LACT-added parameters only, including FW bases, private Q/K/V/O projections, memory gates, LR projections, and memory normalization. Original ViT, LM, and projector remain frozen, matching v2.
- Optimizer and schedule: peak FW LR `2e-5`, minimum LR `1e-6`, cosine decay, 3% warmup, weight decay 0, and one epoch. This intentionally reuses v2's optimizer settings.
- Stabilization: `clip_ns_grad_ratio=True`, `clip_state_grad_ratio=True`, both at rho 1, plus the framework's true FSDP global gradient norm clip at 1.0.
- Hardware and packing: 8xH100, global batch 16, 8K sample/pack length, 1 FPS, 64-512 frames, four-frame LACT groups, and at most 127 effective FW updates per video. Startup packed 5,870 source rows into 2,141 samples for 134 optimizer steps.
- Training W&B: [`v4`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v4).
- Active run directory: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v4/20260830064443`.
- Startup validation: all four datasets loaded, FSDP retained exactly 358.5M LACT-only trainable parameters, and public W&B authenticated as `yibozhong657 (LVSM-Experiment)`. Step 1 completed in 255.35 seconds with global calibrated CE (`total_loss` / `reduced_llm_loss`) `2.79301`, rank-0 pre-all-reduce diagnostic `loss` `0.31798`, pre-clip grad norm `3.072`, max allocated memory `45.75 GB`, and no NaN/OOM; the initial ETA was about 9h26m.
- Final checkpoint: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v4/20260830064443/hf-134`.

### Checkpoint Diagnostics

| Parameter group | Result vs initialization |
|---|---:|
| Memory gate RMS / mean abs / max abs | `2.64e-4` / `1.87e-4` / `1.34e-3` |
| FW base relative L2 delta | `3.51e-3` |
| FW private projections relative L2 delta | `9.29e-3` |
| FW value projection relative L2 delta | `7.03e-3` |
| FW LR projection relative L2 delta | `1.11e-3` |
| FW memory norm | Bitwise unchanged |
| Original attention / MLP / other vision | Bitwise unchanged |
| LM / projector | Bitwise unchanged |

Training completed all 134 steps in 30,040.99 seconds (8h20m41s). The first/last 26-step global CE means were `2.82099/2.83446`, so training loss was effectively flat across the heterogeneous one-pass data. Pre-clip grad norm had mean `2.216`, median `1.760`, maximum `22.141` at step 126, and exceeded 1 on 131/134 steps; all values remained finite and the global clip handled the spikes. Mean step time was 222.78 seconds, maximum allocated memory was 56.64 GB, and the final LR was `1.003e-6`.

### Core Evaluation

Native artifacts: `/mnt/localssd/VideoChat3/eval/videochat3-lact-v4-core/VideoChat3-4B-LACT-v4/T20260830_Gc987fe11`.

| Benchmark | Base | v4 | Delta |
|---|---:|---:|---:|
| Video-MME Short | 80.70 | 80.90 | +0.20 |
| Video-MME Long | 60.30 | 60.40 | +0.10 |
| MVBench MP4 64-frame | 70.83 | 70.83 | +0.00 |
| MMBench DEV EN V1.1 | 35.91 | 35.60 | -0.31 |

Conclusion: v4 preserves the Base model but does not produce a meaningful long-video gain. Relative to v2 it recovers `+0.30` on Video-MME Long while keeping the same Short score, but both remain effectively at Base parity. The LongVid supervision opened the memory gate less than v2 and moved every FW subgroup less, while the one-pass global CE stayed flat; do not add a second epoch without a new controlled hypothesis or a data-specific held-out evaluation.

## Cross-Version Visual-Token Similarity

The Base, LACT initialization, and v1-v4 checkpoints were run in BF16/FlashAttention on identical cached Video-MME frames: three short videos with 148, 172, and 226 frames and three long videos with 1,024 frames each, all at a `24x40` patch grid. `scripts/compare_videochat3_lact_features.py` compares aligned post-ViT/patch-merger tokens and the projected tokens actually inserted into the LM. Values below are unweighted means across the three videos in each duration group; relative L2 is `||variant - Base|| / ||Base||`. Native inputs, Base references, per-sample metrics, and the consolidated result are under `/mnt/localssd/VideoChat3/feature_similarity/base_v1_v4`.

| Model | Short ViT cosine / rel-L2 | Short projected cosine / rel-L2 | Long ViT cosine / rel-L2 | Long projected cosine / rel-L2 |
|---|---:|---:|---:|---:|
| LACT init | `1.000000 / 0.00%` | `1.000000 / 0.00%` | `1.000000 / 0.00%` | `1.000000 / 0.00%` |
| v1 | `0.994671 / 10.49%` | `0.980710 / 27.51%` | `0.995369 / 9.96%` | `0.980676 / 24.84%` |
| v2 | `0.999733 / 2.38%` | `0.998843 / 5.48%` | `0.999766 / 2.26%` | `0.998879 / 4.92%` |
| v3 | `0.994687 / 10.46%` | `0.980751 / 28.49%` | `0.995228 / 10.10%` | `0.980199 / 25.89%` |
| v4 | `0.999745 / 2.34%` | `0.998870 / 5.42%` | `0.999786 / 2.16%` | `0.998959 / 4.75%` |

The hypothesis is strongly supported for FW-only v2/v4: their LM-facing visual tokens remain very close to Base, and Video-MME/MVBench reproduce 95.1%-97.8% of Base's exact output strings and 97.8%-98.7% of its correctness decisions. The original-ViT-trained v1/v3 move features much farther and change more outputs, but the changes are not directionally useful and still cancel in aggregate accuracy. On the three 1,024-frame videos, v2/v4 projected relative L2 only grows from `4.84%/4.65%` in the first temporal quarter to `5.28%/4.94%` in the last; there is no strong horizon-dependent divergence despite 256 recurrent clips. This points to a weak effective FW residual and weak task-aligned learning, rather than evaluation parity being caused only by too few raw videos.

## v5 - VideoChat-Flash LongVid, 10x Gate LR

**Status:** Stopped at step 33; diagnostic-only, with no checkpoint or evaluation.

- Objective: isolate whether v4's zero-gate cold start and small effective outer update budget prevented the FW residual from opening. The only intended v4 recipe change is a 10x optimizer LR for the 31,104 memory-gate parameters.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`; this is a controlled restart and does not reuse the v4 checkpoint.
- Data: the same pinned VideoChat-Flash LongVid manifest and 5,870 QA rows used by v4, with the same deterministic `img2`, 1 FPS, 64-512-frame recipe.
- Trainable scope: the same 358,473,600 LACT-added parameters as v4. The 31,104 memory-gate parameters form an independent optimizer group; the remaining 358,442,496 LACT parameters stay in the normal FW group. Original ViT, LM, and projector remain frozen.
- Optimizer and schedule: non-gate FW `2e-5 -> 1e-6`; memory gate `2e-4 -> 1e-5`; both use the same 3% warmup and proportional cosine schedule, preserving an exact 10x ratio at every step. Weight decay remains 0 and training remains one epoch.
- Stabilization: unchanged NS5 and complete state-adjoint ratio clipping at rho 1, plus true FSDP global gradient norm clipping at 1.0.
- Hardware and packing: unchanged 8xH100, global batch 16, 8K packs, 1 FPS, 64-512 frames, four-frame LACT groups, 2,141 packed samples, and expected 134 optimizer steps.
- Training W&B: [`v5`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-gatelr2e4-ns5r1-stgr1-v5).
- Launcher: `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_longvid_v5.sh`.
- Active run directory: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-gatelr2e4-ns5r1-stgr1-v5/20260830191314`.
- Startup validation: FSDP retained 358,442,496 non-gate LACT parameters at `2e-5` and exactly 31,104 memory-gate parameters at `2e-4`; all other model parameters remain frozen. Public W&B authenticated as `yibozhong657 (LVSM-Experiment)`. Step 1 matched v4's first batch with global CE `2.79301167`, pre-clip grad norm `3.0679`, rank-0 peak memory `45.75 GB`, no NaN/OOM, and an initial ETA of about 9h03m. Step 2 confirmed the live warmup scheduler at `lact_fw=5e-6` and `lact_gate=5e-5`, preserving the exact 10x ratio; global CE was `2.96465445` and pre-clip grad norm was `1.6892`.
- Expected artifact: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-gatelr2e4-ns5r1-stgr1-v5/20260830191314/hf-134`.

The user stopped v5 after step 33 because the paired loss remained extremely close to v4. Through step 31, v4/v5 mean global CE was `2.83995/2.83817`; only the first two steps were exactly equal, but the mean difference was just `-0.00177`. The run was stopped cleanly, its eight GPUs were released, and its W&B state was marked failed. It must not be resumed or evaluated.

## v6 - VideoChat-Flash LongVid, Random Gate Initialization

**Status:** Completed. Usable checkpoint, but Base-parity evaluation despite substantially larger visual-token control.

- Objective: give the FW residual direct control over the initial ViT output while changing only gate initialization. All optimizer and data settings return exactly to v4; v5's gate-specific LR is not used.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-random-gate-init`, derived from the retained zero-gate LACT init with seed 42. Its 27 memory gates use PyTorch `nn.Linear`'s default fan-in uniform distribution, `U(-1/sqrt(1152), 1/sqrt(1152))`; gate RMS is `0.01694`, range is `[-0.02942, 0.02917]`, and all non-gate tensors are bitwise unchanged.
- Initial functional effect: across the fixed three-short/three-long Video-MME probe, post-ViT relative L2 is approximately `6.3%` and LM-facing projected-token relative L2 is approximately `14%-15%` versus Base. The same probe loaded and ran without numerical errors.
- Data: the same pinned VideoChat-Flash LongVid manifest, 5,870 QA rows, deterministic `img2`, 1 FPS, and 64-512-frame recipe as v4/v5.
- Trainable scope: the same 358,473,600 LACT-added parameters as v4, in one uniform optimizer group. Original ViT, LM, and projector remain frozen.
- Optimizer and schedule: all LACT parameters, including gate, use the v4 outer LR `2e-5 -> 1e-6`, 3% warmup, cosine decay, weight decay 0, and one epoch. There is no gate-specific optimizer group.
- Stabilization, hardware, and packing: unchanged rho-1 NS5/state-adjoint clipping, FSDP global clip 1.0, 8xH100, global batch 16, 8K packs, 1 FPS, 64-512 frames, 2,141 packed samples, and expected 134 optimizer steps.
- Training W&B: [`v6`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-randgate-seed42-ns5r1-stgr1-v6).
- Launcher: `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_longvid_v6.sh`.
- Active run directory: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-randgate-seed42-ns5r1-stgr1-v6/20260830212914`.
- Startup validation: FSDP placed all 358,473,600 LACT parameters, including the random gates, in one uniform `2e-5` optimizer group; original ViT, LM, and projector stayed frozen. Public W&B authenticated as `yibozhong657 (LVSM-Experiment)`. On the same first batch as v4, step 1 global CE was `2.80014372` versus v4's `2.79301167`, proving an immediate functional change; pre-clip grad norm was a finite `3.9165` versus v4's `3.0724`, rank-0 peak memory stayed exactly `45.75 GB`, and no NaN/OOM occurred.
- Final checkpoint: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-randgate-seed42-ns5r1-stgr1-v6/20260830212914/hf-134`.

### Checkpoint Diagnostics

Diagnostics compare the final checkpoint with the seed-42 random-gate initialization, not with the zero-gate LACT init.

| Parameter group | Result vs random-gate initialization |
|---|---:|
| Memory gate final RMS / mean abs / max abs | `0.016970` / `0.014695` / `0.030273` |
| Memory gate relative L2 delta | `1.422%` |
| FW base relative L2 delta | `0.671%` |
| FW private projections relative L2 delta | `0.979%` |
| FW value projection relative L2 delta | `0.882%` |
| FW LR projection relative L2 delta | `0.124%` |
| FW memory norm | Bitwise unchanged |
| Original attention / MLP / other vision | Bitwise unchanged |
| LM / projector | Bitwise unchanged |

The random gate did not collapse: RMS changed only from `0.016945` to `0.016970`. Training completed all 134 steps with first/last-26 global CE means `2.81428/2.79197`. Against the same v4 batches, v6 CE was lower by `0.02877` on average, lower on 125/134 steps, and lower by `0.04249` over the last 26 steps. Pre-clip grad norm had mean `3.688`, median `2.849`, and maximum `20.599` at step 133; all 134 steps exceeded 1 and were handled by the global clip. Mean step time was 221.58 seconds, maximum allocated memory was 56.64 GB, and final LR was `1.003e-6`.

The fixed six-video probe confirms persistent functional control rather than a return to Base. Native per-sample metrics are `/mnt/localssd/VideoChat3/feature_similarity/base_v1_v4/v6.json`.

| Duration | ViT cosine / relative L2 vs Base | Projected cosine / relative L2 vs Base |
|---|---:|---:|
| Short, 3 videos | `0.998068 / 6.38%` | `0.992004 / 15.25%` |
| Long, 3x1,024 frames | `0.998031 / 6.51%` | `0.991189 / 14.56%` |

For the long videos, projected relative L2 was `14.64%` in the first temporal quarter and `14.70%` in the last. The random gate therefore increases overall visual-token control substantially over v4, but still does not create horizon-dependent divergence across 256 recurrent clips.

### Core Evaluation

Native artifacts: `/mnt/localssd/VideoChat3/eval/videochat3-lact-v6-core/VideoChat3-4B-LACT-v6/T20260831_G9a1c4f2d`.

| Benchmark | Base | v6 | Delta |
|---|---:|---:|---:|
| Video-MME Short | 80.70 | 80.10 | -0.60 |
| Video-MME Long | 60.30 | 60.30 | +0.00 |
| MVBench MP4 64-frame | 70.83 | 70.85 | +0.02 |
| MMBench DEV EN V1.1 | 35.91 | 35.76 | -0.15 |

All 900 Short, 900 Long, and 4,000 MVBench predictions were produced and scored successfully. Against Base, exact output strings remained identical for `95.22%/95.44%/96.30%` of Short/Long/MVBench examples, and correctness remained identical for `98.11%/97.78%/97.63%`.

Conclusion: random nonzero gates solve the narrow control problem but not the downstream capability problem. Compared with v4, v6 keeps a roughly 3x larger LM-facing feature deviation (`14.6%` versus `4.8%` on the long probe), lowers paired training CE on 125/134 batches, and materially increases FW gradients, yet it remains at Base parity on every core benchmark. Its long-video feature deviation is also flat from the first to last temporal quarter. More feature movement alone is therefore insufficient; v7 tests whether preventing the optimizer from changing the random gate forces the remaining FW parameters to adapt more usefully.

## v7 - VideoChat-Flash LongVid, Frozen Random Gate

**Status:** Stopped by user at step 6; diagnostic-only, with no checkpoint or evaluation.

- Objective: prevent the optimization from reducing FW control by closing the gate. This is a single-variable comparison against v6: the identical nonzero random gates participate in every forward but remain frozen.
- Initialization: the same seed-42 `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-random-gate-init` used by v6. All 31,104 gate values remain fixed at RMS `0.01694`; there is no reuse of the trained v6 checkpoint.
- Data: unchanged VideoChat-Flash LongVid manifest, 5,870 QA rows, `img2`, 1 FPS, and 64-512 frames.
- Trainable scope: 358,442,496 non-gate LACT parameters. The 31,104 memory-gate parameters, original ViT, projector, and LM are frozen.
- Optimizer and schedule: one uniform non-gate FW group with the v4/v6 LR `2e-5 -> 1e-6`, 3% warmup, cosine decay, weight decay 0, and one epoch. No gate optimizer group exists.
- Stabilization, hardware, and packing: unchanged rho-1 NS5/state-adjoint clipping, FSDP global clip 1.0, 8xH100, global batch 16, 8K packs, 1 FPS, 64-512 frames, 2,141 packed samples, and expected 134 optimizer steps.
- Training W&B: [`v7`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-randgate-frozen-seed42-ns5r1-stgr1-v7).
- Launcher: `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_longvid_v7.sh`.
- Active run directory: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-randgate-frozen-seed42-ns5r1-stgr1-v7/20260831144318`.
- Startup validation: every rank reported `Freeze LACT memory gates: 0.031M`; the optimizer contained one uniform non-gate FW group and no gate group. Step 1 matched v6 exactly in global CE (`2.80014372`) because both use the identical random-gate initialization; its pre-clip grad norm was `3.1464`, lower than v6's `3.9165` because gate gradients were absent. Steps 1-6 were finite with no NaN/OOM.

The user stopped v7 after step 6 because the current training configuration was judged to have no further research value. All eight GPUs were released, the W&B state was marked failed, and no checkpoint was produced. Do not resume or evaluate this run.
