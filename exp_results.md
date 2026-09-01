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

Conclusion: random nonzero gates solve the narrow control problem but not the downstream capability problem. Compared with v4, v6 keeps a roughly 3x larger LM-facing feature deviation (`14.6%` versus `4.8%` on the long probe), lowers paired training CE on 125/134 batches, and materially increases FW gradients, yet it remains at Base parity on every core benchmark. Its long-video feature deviation is also flat from the first to last temporal quarter. More feature movement alone is therefore insufficient.

### Teacher-Forced Causal Ablation

Native artifacts: `/mnt/localssd/VideoChat3/eval/v6-teacher-forced-ablation`. The reproducible implementation is `scripts/eval_videochat3_lact_teacher_forced_ablation.py` with the eight-GPU launcher `scripts/eval_videochat3_lact_v6_teacher_forced.sh`.

The probe samples 96 Video-MME Long videos with seed 42 and evaluates all three questions per video, for 288 paired questions. Every answer letter is exactly one tokenizer token. Normal and perturbed conditions share the same cached 1,024 frames and teacher-forced prompt; shuffle permutes non-overlapping four-frame chunks while keeping timestamp positions ordered, reset reinitializes FW state before every chunk, and gate-zero bypasses the FW scan and is algebraically Base because every original tensor is frozen. Confidence intervals use 10,000 bootstrap resamples clustered by video.

| Condition | Mean answer NLL | Median NLL | Mean correct-answer probability |
|---|---:|---:|---:|
| Sequential FW | 1.26768 | 0.62716 | 51.86% |
| Shuffle 4-frame chunks | 1.36070 | 0.72620 | 49.27% |
| Reset state every chunk | 1.26346 | 0.60323 | 52.05% |
| Gate zero / Base | 1.27245 | 0.62638 | 51.83% |

| Paired contrast | Mean NLL delta | Median delta | Positive fraction | Video-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Shuffle - sequential | +0.09302 | +0.00962 | 56.94% | `[+0.02979, +0.15645]` |
| Reset - sequential | -0.00421 | -0.00171 | 43.06% | `[-0.02185, +0.01376]` |
| Gate-zero - sequential | +0.00477 | +0.00054 | 53.47% | `[-0.01105, +0.02118]` |

The ordered visual stream matters: chunk shuffling significantly worsens NLL, although this contrast measures temporal order/content-to-timestamp alignment rather than pure video-versus-text dependence. In contrast, resetting FW state does not hurt and is slightly better on average, with a confidence interval centered around zero. Removing the entire FW residual also has no reliable effect. Thus v6 provides no measurable evidence that its cross-chunk recurrent state or even its static FW residual helps answer Video-MME Long questions; the frozen Base LM uses the ordered clip tokens directly.

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

## v8 - NExT-QA Open-Ended, v4 FW-Only Recipe

**Status:** Stopped by user at step 147/343; diagnostic-only, with no HF checkpoint or evaluation.

- Objective: test whether manually annotated causal and temporal video QA provides a cleaner trainable signal for the zero-gate LACT fast-weight path than the contaminated VideoChat-Flash LongVid data. This is a data-only comparison against v4; initialization, trainable scope, optimizer, stabilization, batch/packing, frame limits, and epoch count are unchanged.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`; no v4-v7 checkpoint reuse. The 27 memory gates are zero initialized exactly as in v4.
- Data: the NExT-QA open-ended train split from `lmms-lab/NExTQA` revision `a0d7729e38399da9c8a70c59aa4ad7f6996d3c00`, paired with complete train media from the `NeXT-QA` portion of `Video-R1/Video-R1-data` revision `9ecf5eff38945e9ae4958058b83c9337f54aadd4`. The source contains 37,523 QA rows over 3,870 videos. Conversion emits 37,496 rows over 3,867 valid videos: 21 rows are omitted for two absent source videos and six rows for one released MP4 that is truncated from 2,697 annotated frames to 178 decoded frames. Every emitted media path is present and non-empty. The full-video recipe matches v4 at 1 FPS, 64-512 uniformly sampled frames, rounded down to a multiple of four; MP4 decoding uses `decord` instead of LongVid's extracted-JPG `img2` backend.
- Trainable scope: the same 358,473,600 LACT-added parameters as v4, including FW bases, private Q/K/V/O projections, memory gates, LR projections, and memory normalization. Original ViT, LM, and projector remain frozen.
- Optimizer and schedule: the exact v4 recipe for all LACT parameters, including the gates: peak LR `2e-5`, minimum LR `1e-6`, cosine decay, 3% warmup, weight decay 0, and one epoch. There is no gate-specific optimizer group.
- Stabilization: unchanged `clip_ns_grad_ratio=True` and `clip_state_grad_ratio=True` at rho 1, plus true FSDP global gradient norm clipping at 1.0.
- Hardware and packing: unchanged 8xH100, global batch 16, 8K sample/pack length, 1 FPS, 64-512 frames, and four-frame LACT groups. Deterministic caching packs the 37,496 source rows into 5,473 samples for 343 optimizer steps.
- Training W&B: [`v8`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-nextqa-oe37496-8xh100-gb16-video1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v8).
- Launcher: `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_nextqa_v8.sh`.
- Prepared data: `/mnt/localssd/dataset/VideoChat3/NExTQA/NExTQA_OE_Train_VideoChat3.json`; conversion audit is `/mnt/localssd/dataset/VideoChat3/NExTQA/nextqa_conversion_summary.json`.
- Active run directory: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-nextqa-oe37496-8xh100-gb16-video1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v8/20260831173621`.
- Startup validation: all ranks loaded the expected 37,496 rows and v4-equivalent optimizer scope; the framework reports the same LACT-only parameters under its aggregate `ViT` group at `2e-5`, with projector and LM frozen. Public W&B authenticated as `yibozhong657 (LVSM-Experiment)`. Steps 1-3 had global CE `4.89555/4.92955/4.89434`, finite pre-clip norms `1.4981/3.5470/1.1570`, peak memory no higher than `46.66 GB`, and the expected warmup LR `0/2e-6/4e-6`. There was no NaN, OOM, or media error. After dataloader warmup, steps 2-3 took `93.06/91.32` seconds, giving an initial ETA of about 8h40m from step 3.
- Expected artifact if completed: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-nextqa-oe37496-8xh100-gb16-video1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v8/20260831173621/hf-343`.

The user stopped v8 after step 147 to replace another incomplete FW training run with a direct Base-model visual-token sensitivity test. The run stopped cleanly, all eight GPUs were released, and public W&B was marked failed. No HF checkpoint exists. The retained `ckpt-step-100` DCP state is diagnostic-only and must not be resumed or evaluated. Through step 147, first/last-29 global CE means were `4.97207/4.97358`, so the harder NExT-QA loss remained flat. Mean step time was `92.72` seconds; pre-clip grad norm had mean `1.419`, median `1.168`, maximum `6.945`, and exceeded 1 on 110/147 steps. Maximum allocated memory was `46.98 GB`.

## Base Visual-Token Perturbation - v8 Follow-up

**Status:** Completed. The 15% random and sinusoidal feature changes have no statistically reliable aggregate NLL effect.

- Objective: determine whether a visual-feature change as large as the approximately 15% LM-facing divergence produced by v6 materially changes the correct-answer likelihood, and whether a structured cross-clip temporal signal behaves differently from an equal-magnitude random change.
- Model and data: unmodified `/mnt/localssd/VideoChat3/VideoChat3-4B` on exactly the same seed-42 sample of 96 Video-MME Long videos and three questions per video used by the v6 teacher-forced causal ablation, for 288 paired questions. Cached 1,024-frame visual inputs, prompts, correct one-token answer letters, and all pixel/token budgets are unchanged.
- Intervention point: projected visual tokens immediately before they replace `<video>` placeholders in the frozen Base LM. Every intervention is calibrated independently per video so the realized global `||delta V||_2 / ||V||_2` is 15%.
- Conditions: (1) unmodified Base tokens; (2) deterministic per-video iid Gaussian additive noise; (3) a centered standard sinusoidal vector for each non-overlapping four-frame group, shared by every projected token from that group. Centering removes the feature-wise constant component, so the third intervention contains temporal variation rather than a common embedding bias. Random and sinusoidal conditions have the same 15% perturbation budget.
- Metric: teacher-forced mean NLL of the true answer-letter token, mean correct-answer probability, paired NLL deltas, and 10,000-resample 95% bootstrap confidence intervals clustered by video. Feature cosine, norm ratio, and realized relative L2 are audited separately for every video.
- Reproducible implementation: `scripts/eval_videochat3_base_visual_token_perturbations.py`; eight-GPU launcher `scripts/eval_videochat3_base_visual_token_perturbations.sh`.
- Expected artifact: `/mnt/localssd/VideoChat3/eval/base-visual-token-perturbations-r015`.
- Smoke validation: one 1,024-frame video and three questions completed. Random/sinusoidal realized relative L2 was `15.0009%/15.0008%`, both had feature cosine `0.988936`, and all projected-token counts matched the LM placeholders. Smoke NLL is not interpreted because it contains only one video.

Native artifacts: `/mnt/localssd/VideoChat3/eval/base-visual-token-perturbations-r015`. All 96 videos and 288 questions completed once, all eight rank files were present, and the GPU-exclusive watchdog recorded a clean exit. The unmodified Base NLL exactly reproduces the prior v6 gate-zero condition, independently validating sample and prompt identity.

| Condition | Mean answer NLL | Median NLL | Mean correct-answer probability | Feature relative L2 / cosine / norm ratio |
|---|---:|---:|---:|---:|
| Base | `1.27245` | `0.62638` | `51.83%` | `0.00% / 1.000000 / 1.0000` |
| Random additive noise | `1.28310` | `0.65612` | `51.33%` | `15.0009% / 0.988935 / 1.0112` |
| Four-frame sinusoidal encoding | `1.27414` | `0.67183` | `51.49%` | `15.0009% / 0.988935 / 1.0112` |

| Paired contrast | Mean NLL delta | Median delta | Positive fraction | Video-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Random - Base | `+0.01065` | `+0.00054` | `53.47%` | `[-0.00651, +0.02909]` |
| Sinusoidal - Base | `+0.00169` | `-0.00020` | `47.92%` | `[-0.01708, +0.02141]` |
| Sinusoidal - Random | `-0.00896` | `-0.00080` | `47.92%` | `[-0.02704, +0.00935]` |

The small means are not solely cancellation of ubiquitous large effects. For random perturbations, `34.7%/57.3%/74.0%` of question-level absolute NLL changes are below `0.01/0.05/0.10`; the corresponding sinusoidal fractions are `33.3%/55.9%/69.1%`. There are real tails in both directions, but neither perturbation produces a consistent population-level degradation or improvement.

Conclusion: a `15%` global projected-feature relative L2 change, despite lowering cosine to about `0.989`, is not intrinsically a large functional intervention for the frozen LM. This explains how v6 could move LM-facing features by roughly `14%-15%` while remaining Base-equivalent: most of that movement can lie in directions to which the answer logits are locally insensitive. The equal-budget sinusoidal encoding also provides no reliable gain, but this does not show that temporal position is useless; the Base LM was never trained to interpret this arbitrary additive basis and already receives explicit timestamp tokens. Feature L2 should therefore not be used as a proxy for learned control without downstream NLL/logit sensitivity or task-Jacobian-aligned evidence.

## v4 FW Residual Scaling Sweep

**Status:** Completed. No evidence that v4 is under-scaled; the residual is directional but not beneficial at its learned sign/magnitude.

- Objective: directly test whether v4's learned FW direction is useful but under-scaled. At the LM-facing projected-token boundary, compute `h(alpha) = h_base + alpha * (h_v4 - h_base)` for `alpha in {-2, 0, 0.5, 1, 2, 4}`. This exact output-level interpolation is used instead of multiplying internal gates, which would introduce nonlinear layer-to-layer changes and would not satisfy the stated formula.
- Model and data: v4 checkpoint `20260830064443/hf-134` on the identical seed-42 sample of 96 Video-MME Long videos and 288 one-token-answer questions used by both prior teacher-forced experiments. Inputs, prompts, cached 1,024 frames, and token/pixel budgets are unchanged.
- Endpoints: `h_base` is computed by bypassing the complete v4 FW memory scan and is algebraically Base because every original tensor is bitwise unchanged; `h_v4` is the normal sequential v4 vision output. `alpha=0/1` directly reuse these tensors without round-trip arithmetic.
- Metrics: mean/median correct-answer NLL, mean correct-answer probability, paired contrasts against `alpha=1`, equal-magnitude directionality contrast `alpha=2` versus `alpha=-2`, and 10,000-resample 95% confidence intervals clustered by video. Every condition also audits relative feature L2, cosine, norm ratio, and realized projection onto the learned v4 residual.
- Reproducible implementation: `scripts/eval_videochat3_v4_fw_residual_sweep.py`; eight-GPU launcher `scripts/eval_videochat3_v4_fw_residual_sweep.sh`.
- Expected artifact: `/mnt/localssd/VideoChat3/eval/v4-fw-residual-alpha-sweep`.
- Smoke validation: one 1,024-frame video and three questions completed. `alpha=0` exactly reproduced the independent Base smoke NLL `1.11639204`; realized alpha projections were `-1.99995/0/0.50002/1/1.99986/3.99996`. The video's learned v4 residual was `4.1641%` relative L2 at `alpha=1` and scaled to `8.3277%/16.6563%` at `alpha=2/4`. Smoke NLL is not interpreted because it contains only one video.

Native artifacts: `/mnt/localssd/VideoChat3/eval/v4-fw-residual-alpha-sweep`. All 96 videos and 288 questions completed once across eight rank files, and the GPU-exclusive watchdog recorded a clean exit. `alpha=0` again exactly reproduces the prior Base/gate-zero aggregate NLL `1.27245085`, providing an independent endpoint check. Mean realized alpha projections were within `1.1e-4` of their targets.

| Alpha | Mean answer NLL | Median NLL | Mean correct-answer probability | Feature relative L2 / cosine vs Base |
|---:|---:|---:|---:|---:|
| `-2` | `1.64566` | `1.43448` | `38.15%` | `9.5379% / 0.995449` |
| `0` (Base) | `1.27245` | `0.62638` | `51.83%` | `0.0000% / 1.000000` |
| `0.5` | `1.27588` | `0.64615` | `51.63%` | `2.3926% / 0.999710` |
| `1` (v4) | `1.27922` | `0.63219` | `51.52%` | `4.7690% / 0.998849` |
| `2` | `1.31829` | `0.70730` | `50.06%` | `9.5377% / 0.995410` |
| `4` | `2.28455` | `2.27868` | `21.88%` | `19.0762% / 0.981950` |

| Paired contrast | Mean NLL delta | Video-cluster bootstrap 95% CI |
|---|---:|---:|
| `alpha=-2` - Base | `+0.37321` | `[+0.20105, +0.56062]` |
| `alpha=0.5` - Base | `+0.00343` | `[-0.00466, +0.01176]` |
| `alpha=1` - Base | `+0.00677` | `[-0.00415, +0.01766]` |
| `alpha=2` - Base | `+0.04584` | `[-0.01727, +0.12871]` |
| `alpha=4` - Base | `+1.01210` | `[+0.74212, +1.28546]` |
| `alpha=2` - `alpha=-2` | `-0.32738` | `[-0.49756, -0.16461]` |

Conclusion: larger positive FW amplitude does not recover hidden utility. `alpha=0/0.5/1/2` is statistically flat, with point estimates monotonically worsening as positive amplitude grows; `alpha=4` causes a large, significant collapse. Thus v4 is not limited by a gate that is simply too small. At equal `9.54%` feature divergence, however, `alpha=2` is decisively better than `alpha=-2`, so the learned residual is not sign-symmetric random noise and contains directional structure. The precise conclusion is a weak directional residual that provides no measurable benefit over Base, not an under-amplified useful residual. Together with the equal-budget random perturbation result, this also shows that downstream sensitivity depends strongly on feature direction: arbitrary 15% noise is mostly ignored, while extrapolating the learned v4 direction to 19% is highly destructive.

## v9 - Random-Half TimeLens Grounding, v4 FW-Only Recipe

**Status:** Prepared for launch.

- Objective: run the same clean temporal-grounding control as planned for v9 while reducing the approximately 22-24 hour full balanced run to roughly half that wall time. The stopped full-data attempt and its W&B run are intentionally removed from the experiment record.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`; no prior checkpoint reuse. All 27 memory gates start at zero and every original Base tensor is bitwise unchanged.
- Data: a deterministic simple random sample of 12,624 rows from the prepared 25,247-event TimeLens visual duration-balanced manifest, using Python seed 42 with no additional stratification or test split. The subset covers 8,985 unique videos and lives at `/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json`. Source media and annotations remain pinned to `TencentARC/TimeLens-100K` revision `75e03f54a19b814de6dc8f5fceb19090625f4844`.
- Supervision: unchanged official TimeLens grounding prompt with `The event happens in <start> - <end> seconds.` targets. The subset is drawn only from the already validated pure-video pool.
- Trainable scope: exactly the v4 358,473,600 LACT-added parameters. Original ViT, multimodal projector, and LM remain frozen.
- Optimizer and schedule: exact v4 settings for every LACT parameter including gates: uniform peak LR `2e-5`, minimum `1e-6`, 3% warmup, cosine decay, weight decay 0, and one epoch. No gate-specific or original-ViT group exists.
- Stabilization: unchanged rho-1 NS5/state-adjoint clipping and true-global gradient norm clip at 1.0.
- Hardware and packing: 8xH100, global batch 16, 8K packs, 2 FPS, 64-448 frames rounded to four, total-pixel budget 14,680,064, and at most 112 clips / 111 effective updates. Exact packed samples and optimizer steps will be recorded after startup cache construction; the expected count is approximately 417 steps.
- Training W&B: [`v9`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-fwonly-timelens-rand12624-8xh100-gb16-video2fps-f448-s8k-fwlr2e5-ns5r1-stgr1-v9).
- Launcher: `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_timelens_v9.sh`.
- Sampling utility and audit: `scripts/sample_timelens_videochat3.py`; `/mnt/localssd/dataset/VideoChat3/TimeLens-100K/timelens_100k_random_12624_summary.json`.
- Expected artifact: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-timelens-rand12624-8xh100-gb16-video2fps-f448-s8k-fwlr2e5-ns5r1-stgr1-v9/<timestamp>/hf-<final-step>`.
- Planned completion checks: final HF checkpoint, gate/FW deltas against zero-gate initialization, original ViT/LM/projector integrity, loss/gradient/clipping behavior, TimeLens-held-out grounding evaluation, and fixed core regression suite.
