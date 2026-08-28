# Project Memo

## Rules

- Keep this memo concise: retain only stable, code-backed conclusions; merge or delete stale detail instead of appending a diary.
- Treat executable model/data paths as authoritative over README wording and experiment names.
- When comparing with `/data/yibo/VideoMamba`, inspect only VideoViT and VideoLACT; do not use Mamba or VideoMARS as references.
- Store checkpoints under `/mnt/localssd/VideoChat3` and datasets under `/mnt/localssd/dataset/VideoChat3`; create these project-named directories when absent and never place downloaded artifacts in Git.

## Environment

- The canonical environment is the root uv project: uv 0.12.1, CPython 3.12.13, `.venv`, and committed `uv.lock`. Reproduce it only with `uv sync --frozen`; do not substitute ad-hoc pip or Conda installs.
- Root `pyproject.toml` installs local XTuner and VLMEvalKit editable sources. The locked GPU ABI is torch 2.8.0 + torchvision 0.23.0 + FlashAttention 2.8.3 on CUDA 12.8; system prerequisites are a compatible NVIDIA driver and FFmpeg/`ffprobe`.
- Python 3.12 compatibility uses `decord2==3.4.0` under the normal `import decord` API, headless OpenCV, and setuptools 80.9.0 for mmengine's legacy `pkg_resources` import.
- Petrel/Ceph is optional external infrastructure and is not in the public lock; local or mounted media works without it, while `s3://` data still requires the private client and credentials.
- Run commands after `source .venv/bin/activate` or through `uv run --frozen`. When dependencies change, update package metadata, regenerate `uv.lock`, and verify a clean `uv sync --frozen`; never edit the lock manually.

## Final Checkpoint

- The final Stage 3 checkpoint is `MCG-NJU/VideoChat3-4B`, pinned at revision `37fa901ec5913f84bc31108ebc1e60ad1903634c`, and lives at `/mnt/localssd/VideoChat3/VideoChat3-4B`.
- Reproduce it with `install -d /mnt/localssd/VideoChat3/VideoChat3-4B`, then run `uv run --frozen python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MCG-NJU/VideoChat3-4B", revision="37fa901ec5913f84bc31108ebc1e60ad1903634c", local_dir="/mnt/localssd/VideoChat3/VideoChat3-4B")'`.
- Verify the 28 official file names/sizes, the three safetensors shards against `model.safetensors.index.json`, and local `AutoConfig`/`AutoProcessor` loading; SHA rechecking is unnecessary.

## Stage 3 Data

- The filtered lightweight snapshot is `lmwang/VideoChat3-Stage3-Training-Data` revision `a7def4abd394697856be9cd6276efa98a27f23df` at `/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data`.
- Download with `snapshot_download` while ignoring `data/image/**`, `videochat3_data_annotations/image/**`, and `videochat3_data_annotations/text/**`. Molmo2 is only an external top-level reference and has no files to exclude in this repo.
- The retained download was 57 files / 535.123 GiB: 28 Video tar shards, 4 Motion-Video tar shards, annotations/manifests, and top metadata. The released loader cannot read tar members directly.
- Run `uv run --frozen python scripts/extract_stage3_lightweight.py --delete-tars` to validate, extract, decode-smoke, record recovery state, and permanently delete one verified shard at a time. It writes `.extraction_state.json`, `media/`, and `VideoChat3_Stage3_Training_Data_local.json` under the dataset root and is safe to resume.
- Current extraction is complete: 32/32 shards, 148,649 media items / 570,408,255,436 source bytes, and no tar files remain. All 30,966 references in the 17 local annotations exist; real `VLMJsonlDataset` decoding/tokenization passed for all eight media roots.

## VideoChat3 Model

- `VideoChat3ForConditionalGeneration` is `I3D-ViT -> patch merger -> multimodal projector -> Qwen3 LM`. Projected visual features replace image/video placeholder embeddings before the LM forward.
- Default vision shape is MoonViT-compatible: width 1152, 27 blocks, 16 heads, patch 14, `temporal_patch_size=1`, and `temporal_merge_size=4`. Stage 0 initializes from MoonViT-SO-400M; Stage 1 uses the I3D-ViT checkpoint. The init utility only prefixes compatible ViT keys, so source names/shapes must already match.
- Video patch embedding is per-frame 2D Conv2d, not a 4-frame Conv3d tubelet. Learned spatial position plus learned 4-slot temporal position is added; 2D RoPE has no temporal coordinate and repeats across frames.
- A video's `grid_thw` is split into independent clips of at most 4 frames. Inside each clip, all `t*h*w` patch tokens perform joint full spatial-temporal self-attention. A non-multiple-of-4 tail is one shorter clip.
- All clips from the batch are concatenated into one packed vision forward. FlashAttention uses `cu_seqlens` to evaluate them as independent sequences in one batched call; there is no cross-clip vision attention, KV cache, or recurrent state. Thus this is batched clip processing, not stateful 4-frame streaming.
- After every clip, `patch_merger` averages the temporal axis and groups each 2x2 spatial neighborhood. A full 4-frame clip therefore gets 4x temporal and 4x spatial compression (16x total) before projection. Cross-clip reasoning happens only later in the LM, aided by per-clip timestamp tokens.

## LACT Vision Encoder

- `VideoChat3VisionLACTModel` is a separate encoder selected by `VideoChat3LACTVisionConfig`; the original encoder remains available. Each layer is the unchanged packed 4-frame attention, then VideoLACT fast-weight memory, then the original MLP.
- Fast state is private to one layer and one input video. Each 4-frame clip applies the old state and then updates it for the next clip; state resets at video boundaries and the unused final update is skipped. A short tail is one final group.
- The fast-weight SwiGLU, prediction error, token-wise softplus rates, FP32 master weights, and Muon/Newton-Schulz update match VideoLACT. Private Q/K/V/O projections share-init from the loaded attention weights and a zero-initialized memory gate makes the initial output exactly match pretrained VideoChat3.
- The implementation fuses the three within-layer gradient GEMMs and Muon batches. It keeps the FSDP-compatible layer-major schedule; cross-layer grouped updates are intentionally not used with separately sharded vision blocks.
- `VideoChat3LACTDense4BConfig` loads the original 4B checkpoint with missing LACT weights initialized as above, freezes the LM/projector, and trains the entire vision encoder. Planned data is Stage 3 LV + OL, optionally a small still-undecided Academic2M mix.
- LACT builds and HF saves use an independent `videochat3_lact` model with a `videochat3_lact_vision` encoder. They inherit all non-vision behavior, preserve original key names/processor assets, export custom `auto_map` code, and support strict XTuner resume plus VLMEvalKit/Transformers loading; `hf_interval` can remain enabled.
- The retained initialization checkpoint is `/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init` (9.1 GiB). `initialization_validation.json` confirms all 734 original tensors are bitwise equal, all 270 added tensors are FW-only, and 27/27 layers satisfy Q/K/V/O share-init plus zero gate.
- `scripts/benchmark_videochat3_lact.py` records isolated BF16/FlashAttention H100 timings. For an 8-frame, 16x16 patch-grid video, LACT vision is about 5.5x slower, but full-VLM latency falls from 2.78x at 641 tokens to 2.00x/1.30x/1.04x at 2,177/8,321/32,897 tokens; see `benchmark_h100_long_context.json`. Zero gate does not bypass FW apply/update compute.
- Launch Stage 3 vision-only training with `training_scripts/stage3/VideoChat3_4B_LACT_VE_train_stage3.sh`. It uses the retained LACT init plus the 17-dataset local lightweight manifest and logs rank-zero metrics to the public `LVSM-Experiment/videochat3` W&B project as `vc3-4b-lact-fw4-ve-s3-lite-v1`, with JSONL retained on every rank.
- The LACT training launcher pins `WANDB_BASE_URL=https://api.wandb.ai`; it ignores the machine's Adobe-internal `WANDB_API_KEY` and uses the public netrc login, or `WANDB_PUBLIC_API_KEY` when explicitly supplied. Keep the run ID stable for auto-resume and change both name/ID for a new experiment version.

## VideoMamba ViT/LACT Comparison

- Image and video VideoViT share the same pre-norm attention and slow SwiGLU MLP parameter layout. VideoViT adds Conv3d tubelet embedding and temporal position; it uses one global CLS plus all tubelet patch tokens in full-sequence spatial-temporal attention, so it is not streaming.
- VideoLACT uses one `CLS + spatial patches` sequence per embedded tubelet. `kernel_size` is the temporal tubelet size (CLI default 1 for LACT), while `fw_update_group_size=g` groups `g` consecutive tubelets into both one attention window and one fast-weight chunk.
- Every LACT layer is strictly `grouped window attention -> fast-weight SwiGLU memory -> slow SwiGLU MLP`. Attention is local to a group; persistent fast weights carry information to later temporal groups.
- LACT is apply-then-update: group `i` is computed with the old per-sample, per-layer fast state, then its prediction error updates `w0/w1/w2` for group `i+1`. The final update is skipped because no later group consumes it; prediction uses the last tubelet CLS.
- The memory update learns positive per-token rates with `softplus`, computes the three SwiGLU weight updates, applies batched Muon/Newton-Schulz zeroth-power normalization, updates FP32 master weights, and re-normalizes the runtime fast weights. This inner update remains differentiable to the outer supervised loss.
- `fw_update_layer_group_size=1` uses an optimized layer-major scan; values above 1 use chunk-major execution and batch independent layer updates. These are different topological schedules of the same per-layer recurrent dependency, not different model semantics.
- Image VideoViT checkpoints initialize both video models: 2D patch kernels are inflated into Conv3d, and attention/slow-MLP names match. LACT defaults to private memory Q/K/V/O projections; `share_init` copies them from attention and uses a zero memory gate so step 0 preserves the pretrained attention function.
- The current F64 LACT recipe uses `kernel_size=1`, `g=8`, and layer-update group 4: 64 tubelets become 8 temporal groups with 7 effective recurrent updates per layer.
