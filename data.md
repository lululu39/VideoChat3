# Video Training Data Search

Last reviewed: 2026-08-30.

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
| [VideoChat-Flash LongVid subset](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/tree/main/longvid_subset) | About 197 GB. Contains 2K Ego4D-HCAP event-understanding samples plus HTStep event counting (2K), event relationship (1K), and event understanding (1K). | Released with VideoChat-Flash and the InternVideo2.5 training stack. | First choice: directly hosted, manageable, and explicitly supervises cross-event relations and counting. Generated annotations require a quality audit. |
| [Vript](https://huggingface.co/datasets/Mutonix/Vript) | Long-video media is about 716 GB; videos average about 6 minutes, reach 3 hours, and total about 1.3K hours. Rich scene-level and full-video captions. | Used by VideoChat3, VideoChat-Flash, and SmolVLM2; NeurIPS 2024 Datasets and Benchmarks. | Strong visual-only long-video source. Primarily captioning rather than QA, so use a selected shard set or derive cross-event QA. |
| [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) | About 600 GB; 43,751 videos, 3,425 hours, average 4.7 minutes, with scene splits, narrative descriptions, and QA. | Used in the SmolVLM2 and InternVL2.5 data mixtures. | Good mid/long narrative base and permissive CC-BY source videos. Some labels use speech transcripts, so filter audio/speech-dependent supervision. |
| [Oryx MovieNet long-form data](https://huggingface.co/datasets/THUdyh/Oryx-SFT-Data) | The two long-form artifacts are about 7.87 GB and 8.28 GB; Oryx recommends mixing about 30K examples. Movie sequences average roughly 45 minutes. | Used by Oryx, ICLR 2025. | Cheap long-context activation set for indexed-frame captioning and frame-difference retrieval. Patch/keyframe format needs conversion and the tasks are synthetic. |
| [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | Full release is about 1.28 TB and contains 1.3M video-language instructions over videos up to 3 minutes. The 1-3 minute academic and ActivityNet portions are roughly 196 GB. | Used by LLaVA-Video, VideoLLaMA3, VideoChat3, SmolVLM2, and many later models. | Best general VideoQA/caption anchor, but not sufficient by itself for persistent long-range memory. |
| [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) | 40K detailed GPT-4V captions and a 181K SFT mix; most videos are shorter than 2 minutes. | NeurIPS 2024; reused by Oryx, VideoLLaMA3, SmolVLM2, and others. | Useful high-quality short-video caption anchor, not a primary long-memory source. |
| [LongVALE](https://huggingface.co/datasets/ttgeng233/LongVALE) | 8,411 videos and 549 hours with dense event boundaries and omni-modal captions. | CVPR 2025. | Deprioritized for this pure-video model because many boundaries and answers depend on audio, speech, or audio-visual correlation. |

## Recommended Mix

The most practical first mixture under the current 926 GB free-space budget is:

1. VideoChat-Flash `longvid_subset` as the main explicit long-event supervision.
2. A selected 200-300 GB portion of Vript for visual narrative and scene-to-scene continuity.
3. A selected 100-200 GB LLaVA-Video academic/ActivityNet subset for general VideoQA and captioning stability.
4. Optionally add the roughly 16 GB Oryx MovieNet long-form set for long-distance retrieval pressure.

This follows the successful short-to-long recipe used by VideoChat3 and VideoChat-Flash while keeping the data visual, influential, and small enough to audit before training.
