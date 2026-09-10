# Video Training Data Search

Last reviewed: 2026-09-10.

## Selection Criteria

- Prefer recent, influential datasets that were actually used by established Video-VLMs.
- Use a mixed recipe: general short/mid-length video supervision plus a meaningful long-video component.
- Favor directly hosted media, visual-only supervision, clear train splits, and subsets that fit under `/mnt/localssd/dataset/VideoChat3`.
- Keep evaluation data such as Video-MME, MVBench, MMBench, LongVideoBench, LVBench, MLVU, and EgoSchema out of training.

## What VideoChat3 Uses

VideoChat3 combines complementary data rather than relying on one long-video QA set:

- `VideoChat3-Academic2M` re-annotates LLaVA-Video, S-MiT, Vript, STAR, Sports-QA, and Perception-Test for general captioning, QA, and motion understanding.
- `VideoChat3-LV116K` builds an event ledger over CinePile, LongVideoDB, and SciVideo, then synthesizes full-video captions, cross-segment QA, and temporal grounding supervision.
- The previously tested lightweight Stage 3 package was not representative of full LV116K: its 30,966 references were 70.4% motion data and only 1.1% CinePile, so it supplied little persistent-memory supervision.
- Full LV116K media is too large to take wholesale: the released subsets include about 179 GB CinePile, 2.69 TB LongVideoDB, and 1.83 TB SciVideo-Long. The retired VideoChat3 Stage 3/CinePile data remains excluded unless that project decision is explicitly reversed.

Sources: [VideoChat3 paper](https://arxiv.org/abs/2607.14935), [Academic2M](https://huggingface.co/datasets/MCG-NJU/VideoChat3-Academic2M), and [LV116K](https://huggingface.co/datasets/MCG-NJU/VideoChat3-LV116k).

## Candidate Datasets

| Dataset | Scale and role | Adoption | Assessment for LACT |
| --- | --- | --- | --- |
| [TimeLens-100K](https://huggingface.co/datasets/TencentARC/TimeLens-100K) | 146.5 GB download; 19,466 videos and 96,586 temporal-grounding events. Videos average 106 seconds and reach 499 seconds. | CVPR 2026; used to train the released TimeLens-7B/8B models. | Selected replacement source. It directly supervises visual event localization with timestamp answers and ships complete MP4 media. Use the visual-only filtered manifests prepared below. |
| [VideoChat-Flash LongVid subset](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/tree/main/longvid_subset) | 5,870 QA rows over extracted JPG sequences. | Released with VideoChat-Flash and InternVideo2.5. | Rejected after audit and deleted locally on 2026-09-02: 22.3% answer-in-question shortcuts, 963/963 canonical `after X` rows copy `X`, and one third is numeric event counting. |
| [NExT-QA](https://github.com/doc-doc/NExT-QA) | 5,440 videos and about 52K human causal/temporal QA pairs, averaging 44 seconds. | CVPR 2021 and widely reused as a VideoQA benchmark. | Preserve as a benchmark/control rather than the next main training source. The stopped v8 run is diagnostic-only and has no usable HF checkpoint. |
| [Vript](https://huggingface.co/datasets/Mutonix/Vript) | Long-video media is about 716 GB; videos average about 6 minutes, reach 3 hours, and total about 1.3K hours. Rich scene-level and full-video captions. | Used by VideoChat3, VideoChat-Flash, and SmolVLM2; NeurIPS 2024 Datasets and Benchmarks. | Strong visual-only long-video source. Primarily captioning rather than QA, so use a selected shard set or derive cross-event QA. |
| [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | About 600 GB; 43,751 videos, 3,425 hours, average 4.7 minutes, with scene splits, narrative descriptions, and QA. | Used in the SmolVLM2 and InternVL2.5 data mixtures. | Good mid/long narrative base and permissive CC-BY source videos. Some labels use speech transcripts, so filter audio/speech-dependent supervision. |
| [Oryx MovieNet long-form data](https://huggingface.co/datasets/THUdyh/Oryx-SFT-Data) | The two long-form artifacts are about 7.87 GB and 8.28 GB; Oryx recommends mixing about 30K examples. Movie sequences average roughly 45 minutes. | Used by Oryx, ICLR 2025. | Cheap long-context activation set for indexed-frame captioning and frame-difference retrieval. Patch/keyframe format needs conversion and the tasks are synthetic. |
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | Full release is about 1.28 TB and contains 1.3M video-language instructions over videos up to 3 minutes. The 1-3 minute academic and ActivityNet portions are roughly 196 GB. | Used by LLaVA-Video, VideoLLaMA3, VideoChat3, SmolVLM2, and many later models. | Best general VideoQA/caption anchor, but not sufficient by itself for persistent long-range memory. |
| [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) | 40K detailed GPT-4V captions and a 181K SFT mix; most videos are shorter than 2 minutes. | NeurIPS 2024; reused by Oryx, VideoLLaMA3, SmolVLM2, and others. | Useful high-quality short-video caption anchor, not a primary long-memory source. |
| [LongVALE](https://huggingface.co/datasets/ttgeng233/LongVALE) | 8,411 videos and 549 hours with dense event boundaries and omni-modal captions. | CVPR 2025. | Deprioritized for this pure-video model because many boundaries and answers depend on audio, speech, or audio-visual correlation. |

## Current Decision

The user stopped v37 at step 204 and selected its retained `hf-200` for v38 **multi_stage_video_last** TimeLens training on physical GPUs 4–7. Use the v36 eight-stage recipe, fresh TimeLens optimizer/schedule, and fixed all-token packing over the 12,624-row subset. The native Linear loader now preserves all trained FW/beta/gate tensors when loading a complete Linear checkpoint; historical v36's pre-fix loader reset that branch and is not a matched-weight transfer control. Do not resume v37 or use its unsaved post-200 updates. See `exp_results.md` for status and the versioned launcher. LongVid and the retired Stage 3 release remain excluded; NExT-QA remains a benchmark/control.

Short-video QA supplies broader visual-semantic supervision while reducing the recurrent horizon. With 64 sampled frames and four frames per chunk, each layer performs 15 effective updates; the actual TimeLens random-12,624 recipe averages 57.47 updates per training occurrence (median 55, maximum 111). Ordinary VQA recovery and a measurable benefit from persistent FW state are separate acceptance criteria. Use native generation/scoring and matched Base controls; no teacher-forced evaluation.

TimeLens answers contain explicit temporal ranges rather than one-token counts or recipe-class labels, but the labels are Gemini-generated. The prepared conversion therefore filters explicit speech/audio semantics, checks every media path and decoded duration, and retains the original source revision and hashes.

## LLaVA-Video 0–30s Download and Training

- Source: [`lmms-lab/LLaVA-Video-178K`](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K), pinned to `6d8c562dc26d70042a0d9704d1cae58c94b89098`.
- Root: `/mnt/localssd/dataset/VideoChat3/LLaVA-Video-178K`. Download only `0_30_s_academic_v0_1` and `0_30_s_youtube_v0_1`, plus the README and QA-generation definitions. Do not include the repository's ActivityNet-QA, NExT-QA, or Perception Test repackagings.
- The two subsets contain 27 video archives (137,538,029,515 bytes) and six annotation JSONs (747,292,572 bytes): **138.285 GB** total, excluding the small documentation files. This is compressed download size, not extracted occupancy. Retaining both archives and extracted media requires additional space.

| Source | Open QA conversations | Multiple-choice conversations | Individual QA turns | Separate captions |
| --- | ---: | ---: | ---: | ---: |
| Academic | 48,468 | 5,753 | 245,182 | 11,985 |
| YouTube | 420,200 | 39,353 | 2,082,415 | 79,346 |
| Total | 468,668 | 45,106 | 2,327,597 | 91,331 |

These are raw counts before exclusions/deduplication. Conversations contain 1–5 question/answer pairs; raw QA turns count occurrences, not unique questions. Captions are downloaded and converted but excluded from the default QA training manifest. Published processed JSONs do not retain per-question semantic-type labels. Questions include descriptions, counting, spatial relations, temporal order, causality, motion, and attribute changes; no temporal-only quality claim is made.

Run from the repository root in the locked environment:

```bash
uv run --frozen python scripts/prepare_llava_video_0_30s.py
uv run --frozen python scripts/check_llava_video_0_30s.py --output all
```

On a fresh machine, first run `uv sync --frozen`, download the pinned Base checkpoint specified in `AGENTS.md`, then run `uv run --frozen python scripts/create_videochat3_lact_init.py`. This creates `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init` as a fresh parallel Linear16+Delta initialization with fast-Q/K 3D RoPE, zero gates, no queries, all original visual tokens, and no FW ratio clips. It verifies the 28 official Base files, all 734 unchanged Base tensors, all 27 attention-share-initialized Q/K/V/O branches, independent HF loading, and bitwise zero-gate vision/full-VLM parity. The new 8.61 GiB artifact contains 189 added tensors / 143.89M parameters; historical experiments used a SwiGLU initialization with 270 added tensors at this path. This fresh initialization does not contain v35's trained `hf-800` weights required by the v36 transfer launcher. Use a new experiment record/name for future training.

Preparation checks official file sizes at the pinned revision, streams safe archive extraction, verifies all annotation paths, and decodes every frame selected by the 64-frame recipe. Interrupted runs reuse extraction markers and a size/mtime-keyed decode cache. Use `--skip-download` after the snapshot has arrived, and `--skip-download --skip-extract` to reconvert existing media. Optional `--delete-archives` removes compressed shards only after preparation completes; subsequent reruns then require both skip flags to avoid redownloading them.

Data handling:

- Replace the initial LLaVA `<image>` marker with one VideoChat3 video input and `<VIDEO_CONTEXT>`; preserve all original question text, options, answer formatting, and multi-turn order. Original annotation files remain intact.
- Deterministically hold out approximately 2% of source video IDs with seed 42, keeping every question/format for a video in one split. Normalize `ytb_`/`v_` source names and Charades segment IDs; for YouCook2 `split_N.mp4` clips, use the parent YouTube ID so clips from the same original video stay together. Deduplicate exact normalized conversations per media path within each source/task; overlapping individual QA pairs across different conversations remain.
- On the first run, exclude video IDs in the locally available MVBench MP4 and Video-MME TSVs, TimeLens-Bench dictionaries, and all retained NExT-QA OE splits. Persist IDs and source hashes in `benchmark_exclusions.json` so later runs reuse the same exclusion set. Additional benchmark IDs can be passed with repeatable `--exclude-video-ids` (TSV/Parquet with a `video` column, TimeLens-style JSON dictionary, or newline-separated IDs); explicitly supplied files replace the automatic exclusion inputs. Normalize QVHighlights temporal clips to their source YouTube ID. This is source-ID checking, not exhaustive content matching or a guarantee that the pretrained model never saw these sources.
- Reject missing/empty media, decode failures, and videos longer than 30.5 seconds (0.5-second container-boundary tolerance); record counts and errors in the preparation summary. The official archives omit the referenced YouTube video `y6ReUXtm_VE`; its rows are excluded. Preserve other source supervision rather than introducing unvalidated keyword-based semantic filters.
- Training uses the original multi-turn conversations. Held-out QA is emitted as independent single-turn examples so prior gold answers are not provided during native evaluation. Caption annotations have a separate optional training manifest.
- Sampling: 2 FPS with min=max=64, rounded down to a multiple of four and capped by the original frame count; 224px frame-pixel cap and 3,211,264 total pixels. Thus videos with at least 64 original frames have exactly 15 FW updates per layer; shorter source videos have fewer, without duplicated frames.

Generated artifacts:

- `LLaVA_Video_0_30s_train_VideoChat3.json`: default Academic/YouTube open + multiple-choice QA training manifest.
- `LLaVA_Video_0_30s_heldout_VideoChat3.json`: single-turn QA holdout, never part of the default training manifest.
- `LLaVA_Video_0_30s_caption_train_VideoChat3.json`: optional caption supervision.
- `videochat3_annotations/`, `decoded_metadata.json`, `llava_video_0_30s_prepare_summary.json`: converted rows, decoded metadata, source/output hashes, exclusions, counts, and update statistics.
- `LLaVA_Video_0_30s_smoke_VideoChat3.json` and `tokenization_check_all.json`: bounded real-data smoke input and CPU cache/runtime tokenization checks generated by the check script; this command does not train a model.

Prepared result (2026-09-09): 91,495 extracted videos occupy **138.992 GB**, in addition to the retained archives. Of 91,496 annotation-referenced paths, one is missing (`y6ReUXtm_VE`) and one contains only two frames (`3ujEaKQBqqE`); the other **91,494 videos pass decoding of every selected training frame**. No video exceeds the 30.5-second duration tolerance.

| Prepared QA source | Training conversations | Training QA turns | Held-out single-turn QA |
| --- | ---: | ---: | ---: |
| Academic open | 39,868 | 181,766 | 3,079 |
| Academic multiple-choice | 4,766 | 20,324 | 338 |
| YouTube open | 411,832 | 1,876,321 | 38,093 |
| YouTube multiple-choice | 38,547 | 164,501 | 3,447 |
| Total | **495,013** | **2,242,912** | **44,957** |

QA training spans 87,787 media clips / 82,918 source videos; holdout spans 1,742 clips / 1,675 source videos. `training_manifest_validation.json` confirms zero source-ID overlap across splits or with the 13,002 excluded benchmark IDs. Exactly 494,956 training conversations use 15 updates per layer; the other 57 use fewer due to limited original frames, giving a training-occurrence mean of **14.9995 updates**. Separate optional caption training has 87,622 rows. There are 56 exact duplicate QA conversations removed; the six raw annotation files remain available for audit.

Validation: seven preparation tests pass, and eight real no-query/all-token examples have identical cache/runtime token counts with nonempty answer supervision. Full native XTuner cache/packing on CPU (seed 42, 8K packs, extra buffer 20, eight pack workers) retains all 495,013 conversations: **634,111,947 tokens**, mean/max **1,281.00/1,934** per conversation, **82,518 packs**, and **5,158 optimizer steps per epoch at global batch 16**. `packing_all_summary.json` records these measurements; reusable caches are under `token_cache/lact-all-s8192-smoke0`. v35 uses epoch mode over the full split, without a 200-step training cap.

The launcher defaults to v26's parallel Linear16+Delta, fast-Q/K 3D RoPE, zero initial linear gates, joint ViT/FW/projector training, and frozen LM, **with query pooling disabled**. `VIDEOCHAT3_CHUNK_QUERY=0`, compression factor 1, and no spatial token selection preserve the original per-four-frame temporal mean / 2x2 spatial patch merger and all its output tokens. A 224x224, 64-frame input retains 16 chunks x 64 tokens = **1,024 visual tokens**, plus the original timestamps. Dynamic aspect ratios retain their original grid-dependent counts.

Initialization is the retained LACT init. Ordinary FSDP uses global batch 16, 8K sample/pack lengths, AdamW with cosine `2e-5 -> 1e-6`, 3% warmup (at least one step in bounded mode), global gradient clip 1.0, and no FW ratio clips. The generic launcher defaults to eight GPUs; v35 uses the four idle H100s (physical GPUs 4–7), retaining global batch 16. Set `VIDEOCHAT3_MAX_STEPS` for a bounded adaptation run; zero/unset retains epoch mode (`VIDEOCHAT3_TOTAL_EPOCH`, default one). HF and DCP checkpoints default to every 200 steps and the normal final save, retaining only the latest of each. Checkpoints/logs go under `/mnt/localssd/VideoChat3/training/$WANDB_NAME`; token caches stay in the dataset root.

The 200-step value is a **checkpoint interval**, not the training length or a loss-based stopping rule. Reproduce v35 with `bash xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_ALLTOKENS_train_llava_0_30s_v35.sh` after checking that its assigned GPUs are available.

Selected two-stage procedure: monitor the LLaVA loss trend during the short adaptation stage; once a clear decline is observed, save a usable adapted checkpoint and stop rather than completing the full corpus. This is an operator decision, not an automatic single-batch-loss trigger. Start a new TimeLens run from that adapted HF checkpoint, preserving the no-query/all-token architecture and trainable scope, while restoring TimeLens's own 2-FPS, 64–448-frame data recipe. Start a fresh optimizer/schedule and a new W&B run ID for the new dataset; do not accidentally auto-resume the LLaVA dataloader or reload the original initialization.

`--smoke` runs exactly three optimizer steps at constant LR (warmup disabled), with JSONL tracking and checkpoint saving disabled. It is implementation validation, not a numbered learning experiment. For a full run, first create its numbered `exp_results.md` section and push, then set `WANDB_NAME` and run the same launcher without `--smoke`. Stable `WANDB_RUN_ID` supports normal auto-resume.

Optional launcher settings: `VIDEOCHAT3_LLAVA_OUTPUT=all|r4query|uniform` (default `all`; the other modes require an explicit experiment choice); `VIDEOCHAT3_MODEL_VARIANT=base-vit-projector` selects the matched Base encoder and original checkpoint; `VIDEOCHAT3_LLAVA_MANIFEST` selects a prepared subset. Step/epoch limits, batch size, sample/pack length, save intervals, and LR variables explicitly override the recipe. Keep input sampling and output budget equal for Base-vs-LACT comparisons. When changing output mode, rerun the check script with the matching `--output`.

## Prepared TimeLens-100K

The official snapshot is prepared at `/mnt/localssd/dataset/VideoChat3/TimeLens-100K`, pinned to revision `75e03f54a19b814de6dc8f5fceb19090625f4844`.

- All 20 official shards downloaded and extracted successfully into 19,466 MP4s. The redundant compressed shards were removed after validation; the retained extracted release is about 138 GB.
- The source JSONL contains 96,586 valid single-span events. The pure-video filter removes 7,468 explicit speech/audio-semantic queries; two duration-mismatched videos remove another ten events, leaving 89,108 full events over 19,387 videos.
- Reproducing the official seed-42 duration-balanced target-30K selection yields 25,247 events over 13,790 videos because the longest duration buckets are smaller than their 3,333-event quota.
- `scripts/prepare_timelens_100k.py` emits the balanced and full VideoChat3 JSONL/manifests plus `timelens_100k_conversion_summary.json`.
- After conversion, run `uv run --frozen python scripts/sample_timelens_videochat3.py` to reproduce the seed-42 random-half subset: `TimeLens100K_Visual_Random12624_VideoChat3.json`, with 12,624 events over 8,985 videos. `timelens_100k_random_12624_summary.json` records the source, selected-index, and output hashes.
- The default recipe matches TimeLens at 2 FPS, 64-448 frames, and a 14,680,064 total-pixel budget, while rounding frames to four for LACT. Every balanced sample fits the 8K context; mean/P95/max lengths are `3,454/5,305/5,662` tokens.
- A real 498.9-second sample successfully decoded 448 frames and produced matching cache/runtime lengths of 4,766 tokens with 18 supervised answer tokens.
- Use `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_timelens.sh` after creating the next numbered experiment record and setting `WANDB_NAME`.

## TimeLens Multi-Stage Training

The active TimeLens training recipe is `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_multi_stage_video_last_timelens_v36.sh`. It uses the previous seed-42 random-half manifest (12,624 rows / 8,985 videos) and initializes from v35 `hf-800`. One epoch of fixed all-token 8K packing gives 6,658 packs / 417 optimizer steps at global batch 16. Stages all/2/4/8/16/32/64/video-last receive 53/52/52/52/52/52/52/52 steps. Every input frame still traverses LACT; only the LM-facing chunk outputs and matching timestamps/placeholders are removed. Tail groups retain their last chunk. Adam moments and the single cosine schedule continue across all stages; final `hf-417` exports native `video_last` model/processor metadata. Native benchmark evaluation remains separate from this training.

## Prepared TimeLens-Bench

The held-out official benchmark is prepared at `/mnt/localssd/dataset/VideoChat3/TimeLens-Bench`, pinned to `TencentARC/TimeLens-Bench` revision `5fc78c4b401b2dadf7a3a4355d51d566ff28e0c9`.

- Run `uv run --frozen python scripts/prepare_timelens_bench.py --delete-archives --extract-workers 3` to download, extract, validate, and remove compressed shards. The validated release is about 70 GB: 4,279 MP4s and 9,404 queries across Charades, ActivityNet, and QVHighlights.
- VideoChat3 evaluation uses 2 FPS, at most 448 frames, 224px per frame, and a 14,680,064 total-pixel budget. Frame caches live under `/mnt/localssd/dataset/VLMEvalKit/LMUData` and are reusable.
- Configure a checkpoint in `vlmevalkit-videochat3/configs/videochat3_v12_timelens_bench.json`, then run `bash scripts/eval_videochat3_v12_timelens_bench.sh`; the launcher is eight-GPU and resumable.
- Report the official native metrics separately for each subset: R1@0.3, R1@0.5, R1@0.7, and mIoU. Versioned results and native artifact paths belong in `exp_results.md`.

## Retired VideoChat-Flash LongVid Subset

The rejected candidate was pinned to revision `be87f5516a709be079cec8b727dd2287bf2dd70f`; its 382 GB local data directory was deleted on 2026-09-02. Historical code and experiment artifacts remain.

- Four released QA files contain 5,870 rows over 5,478 per-dataset unique videos; every media reference resolves to a non-empty frame directory.
- The released media is an extracted JPG sequence. The official VideoChat-Flash Stage 3 recipe reads it as `img`, treats non-TVQA frame directories as 1 FPS, samples 64-512 frames, and rounds the sampled length down to a multiple of four.
- `scripts/prepare_videochat_flash_longvid.py` validates the source, converts it to VideoChat3 JSONL, and writes `VideoChatFlash_LongVid_VideoChat3.json`.
- The generated manifest selects `img2`, 1 FPS, 64-512 frames, and `video_frame_multiple=4`. VideoChat3 uses lexicographic ordering for the released zero-padded names such as `00001.jpg`.
- The explicit launcher is `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_VE_train_longvid.sh`. It requires `WANDB_NAME` so a numbered experiment must be recorded before training starts.
