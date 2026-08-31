# Video Training Data Search

Last reviewed: 2026-08-31.

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
| [VideoChat-Flash LongVid subset](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/tree/main/longvid_subset) | 5,870 QA rows over extracted JPG sequences. | Released with VideoChat-Flash and InternVideo2.5. | Rejected after audit: 22.3% answer-in-question shortcuts, 963/963 canonical `after X` rows copy `X` as the answer, and one third of the set is numeric event counting. Retained locally only for audit. |
| [NExT-QA](https://github.com/doc-doc/NExT-QA) | 5,440 videos and about 52K human causal/temporal QA pairs, averaging 44 seconds. | CVPR 2021 and widely reused as a VideoQA benchmark. | Preserve as a benchmark/control rather than the next main training source. The stopped v8 run is diagnostic-only and has no usable HF checkpoint. |
| [Vript](https://huggingface.co/datasets/Mutonix/Vript) | Long-video media is about 716 GB; videos average about 6 minutes, reach 3 hours, and total about 1.3K hours. Rich scene-level and full-video captions. | Used by VideoChat3, VideoChat-Flash, and SmolVLM2; NeurIPS 2024 Datasets and Benchmarks. | Strong visual-only long-video source. Primarily captioning rather than QA, so use a selected shard set or derive cross-event QA. |
| [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | About 600 GB; 43,751 videos, 3,425 hours, average 4.7 minutes, with scene splits, narrative descriptions, and QA. | Used in the SmolVLM2 and InternVL2.5 data mixtures. | Good mid/long narrative base and permissive CC-BY source videos. Some labels use speech transcripts, so filter audio/speech-dependent supervision. |
| [Oryx MovieNet long-form data](https://huggingface.co/datasets/THUdyh/Oryx-SFT-Data) | The two long-form artifacts are about 7.87 GB and 8.28 GB; Oryx recommends mixing about 30K examples. Movie sequences average roughly 45 minutes. | Used by Oryx, ICLR 2025. | Cheap long-context activation set for indexed-frame captioning and frame-difference retrieval. Patch/keyframe format needs conversion and the tasks are synthetic. |
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | Full release is about 1.28 TB and contains 1.3M video-language instructions over videos up to 3 minutes. The 1-3 minute academic and ActivityNet portions are roughly 196 GB. | Used by LLaVA-Video, VideoLLaMA3, VideoChat3, SmolVLM2, and many later models. | Best general VideoQA/caption anchor, but not sufficient by itself for persistent long-range memory. |
| [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) | 40K detailed GPT-4V captions and a 181K SFT mix; most videos are shorter than 2 minutes. | NeurIPS 2024; reused by Oryx, VideoLLaMA3, SmolVLM2, and others. | Useful high-quality short-video caption anchor, not a primary long-memory source. |
| [LongVALE](https://huggingface.co/datasets/ttgeng233/LongVALE) | 8,411 videos and 549 hours with dense event boundaries and omni-modal captions. | CVPR 2025. | Deprioritized for this pure-video model because many boundaries and answers depend on audio, speech, or audio-visual correlation. |

## Current Decision

The next LACT training source is TimeLens-100K rather than LongVid or NExT-QA:

1. Start with the duration-balanced, visual-only TimeLens SFT subset: 25,247 grounding events.
2. Evaluate learnability and FW-state sensitivity before scaling to the 89,108-event visual-only full manifest.
3. Keep Video-MME, MVBench, MMBench, LongVideoBench, TimeLens-Bench, and NExT-QA held out from this training.
4. Add a general VideoQA/caption anchor only after TimeLens establishes a positive controlled result.

TimeLens answers contain explicit temporal ranges rather than one-token counts or recipe-class labels, but the labels are Gemini-generated. The prepared conversion therefore filters explicit speech/audio semantics, checks every media path and decoded duration, and retains the original source revision and hashes.

## Prepared TimeLens-100K

The official snapshot is prepared at `/mnt/localssd/dataset/VideoChat3/TimeLens-100K`, pinned to revision `75e03f54a19b814de6dc8f5fceb19090625f4844`.

- All 20 official shards downloaded and extracted successfully into 19,466 MP4s. The redundant compressed shards were removed after validation; the retained extracted release is about 138 GB.
- The source JSONL contains 96,586 valid single-span events. The pure-video filter removes 7,468 explicit speech/audio-semantic queries; two duration-mismatched videos remove another ten events, leaving 89,108 full events over 19,387 videos.
- Reproducing the official seed-42 duration-balanced target-30K selection yields 25,247 events over 13,790 videos because the longest duration buckets are smaller than their 3,333-event quota.
- `scripts/prepare_timelens_100k.py` emits the balanced and full VideoChat3 JSONL/manifests plus `timelens_100k_conversion_summary.json`.
- The default recipe matches TimeLens at 2 FPS, 64-448 frames, and a 14,680,064 total-pixel budget, while rounding frames to four for LACT. Every balanced sample fits the 8K context; mean/P95/max lengths are `3,454/5,305/5,662` tokens.
- A real 498.9-second sample successfully decoded 448 frames and produced matching cache/runtime lengths of 4,766 tokens with 18 supervised answer tokens.
- Use `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_FW_train_timelens.sh` after creating the next numbered experiment record and setting `WANDB_NAME`.

## Retired VideoChat-Flash LongVid Subset

The rejected candidate remains locally available at `/mnt/localssd/dataset/VideoChat3/VideoChat-Flash-Training-Data`, pinned to source revision `be87f5516a709be079cec8b727dd2287bf2dd70f`.

- Four released QA files contain 5,870 rows over 5,478 per-dataset unique videos; every media reference resolves to a non-empty frame directory.
- The released media is an extracted JPG sequence. The official VideoChat-Flash Stage 3 recipe reads it as `img`, treats non-TVQA frame directories as 1 FPS, samples 64-512 frames, and rounds the sampled length down to a multiple of four.
- `scripts/prepare_videochat_flash_longvid.py` validates the source, converts it to VideoChat3 JSONL, and writes `VideoChatFlash_LongVid_VideoChat3.json`.
- The generated manifest selects `img2`, 1 FPS, 64-512 frames, and `video_frame_multiple=4`. VideoChat3 uses lexicographic ordering for the released zero-padded names such as `00001.jpg`.
- The explicit launcher is `xtuner-videochat3/training_scripts/stage3/VideoChat3_4B_LACT_VE_train_longvid.sh`. It requires `WANDB_NAME` so a numbered experiment must be recorded before training starts.
