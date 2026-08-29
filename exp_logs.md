# Experiment Logs

## Rules

- Keep one concise, immutable section per experiment version; append follow-up versions instead of rewriting prior outcomes.
- Record the objective, initialization, data, trainable scope, core recipe, checkpoint, W&B links, evaluation, and conclusion.
- Diagnostic runs before the first usable checkpoint are not assigned version numbers; their stable lessons remain in `AGENTS.md`.

## v1 - Lightweight, Full-ViT Training

**Status:** Completed. Usable checkpoint, negative/parity research result.

### Setup

- Objective: train VideoChat3-LACT from the function-preserving LACT initialization while keeping the LM and projector frozen.
- Initialization: `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init`.
- Data: original local lightweight Stage 3 manifest, 17 annotations / 30,966 references; 70.4% motion and 1.1% CinePile.
- Trainable scope: the entire 775M-parameter vision encoder, including original ViT and all LACT/FW parameters.
- Recipe: 8xH100, global batch 16, 8K packs, f3600, one epoch / 208 optimizer steps, ViT LR `2.5e-6`, cosine decay, 3% warmup, weight decay 0, global grad norm 1.
- Stabilization: `clip_ns_grad_ratio=True` and `clip_state_grad_ratio=True`, both with rho 1.
- Training W&B: [`vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1).

### Artifact

- HF checkpoint: `xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1/20260828223921/hf-208`.
- Training checkpoint: the sibling `checkpoints/ckpt-step-208`; `hf-latest` points to `hf-208`.

### Evaluation

| Benchmark | Base | v1 | Delta |
|---|---:|---:|---:|
| Video-MME Short | 80.7 | 80.3 | -0.4 |
| Video-MME Long | 60.3 | 60.1 | -0.2 |
| MVBench MP4 64-frame | 70.83 | 70.97 | +0.14 |
| MMBench DEV EN V1.1 | 35.91 | 34.91 | -1.00 |

- Clean W&B dashboard: [`vc3-lact-lite31k-core-eval-v1`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-lact-lite31k-core-eval-v1-dashboard).
- Raw W&B metrics/artifact run: [`vc3-lact-lite31k-core-eval-v1-raw`](https://wandb.ai/LVSM-Experiment/videochat3/runs/vc3-lact-lite31k-core-eval-v1); artifact `videochat3-lact-core-eval-v1-results`.

### Diagnosis

- Memory-gate RMS reached only `2.52e-5`; the effective memory residual remained negligible.
- Relative checkpoint deltas: FW base weights `1.93e-5`, FW private projections `1.36e-4`, original attention `8.55e-4`, original MLP `5.65e-4`; LM/projector stayed bitwise unchanged.
- All 208 steps triggered global grad clipping. Mean loss was `0.633` over the first 26 steps and `0.607` over the final 26 steps.
- With a zero gate, FW internals receive little initial task gradient; the small shared ViT LR and short run mostly fine-tuned the original ViT. The lightweight mixture also supplied little explicit long-video supervision.

### Conclusion

- v1 approximately preserves the base model but shows no measurable long-video gain and about one point of image-benchmark regression.
- Do not repeat this exact optimizer/data mix at larger scale. A follow-up should prioritize LV/OL data and make the FW channel learn explicitly, for example by freezing or strongly downscaling the original ViT while using a larger FW/gate LR.
