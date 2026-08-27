# Project Memo

## Rules

- Keep this memo concise: retain only stable, code-backed conclusions; merge or delete stale detail instead of appending a diary.
- Treat executable model/data paths as authoritative over README wording and experiment names.
- When comparing with `/data/yibo/VideoMamba`, inspect only VideoViT and VideoLACT; do not use Mamba or VideoMARS as references.

## Environment

- The canonical environment is the root uv project: uv 0.12.1, CPython 3.12.13, `.venv`, and committed `uv.lock`. Reproduce it only with `uv sync --frozen`; do not substitute ad-hoc pip or Conda installs.
- Root `pyproject.toml` installs local XTuner and VLMEvalKit editable sources. The locked GPU ABI is torch 2.8.0 + torchvision 0.23.0 + FlashAttention 2.8.3 on CUDA 12.8; system prerequisites are a compatible NVIDIA driver and FFmpeg/`ffprobe`.
- Python 3.12 compatibility uses `decord2==3.4.0` under the normal `import decord` API, headless OpenCV, and setuptools 80.9.0 for mmengine's legacy `pkg_resources` import.
- Petrel/Ceph is optional external infrastructure and is not in the public lock; local or mounted media works without it, while `s3://` data still requires the private client and credentials.
- Run commands after `source .venv/bin/activate` or through `uv run --frozen`. When dependencies change, update package metadata, regenerate `uv.lock`, and verify a clean `uv sync --frozen`; never edit the lock manually.

## VideoChat3 Model

- `VideoChat3ForConditionalGeneration` is `I3D-ViT -> patch merger -> multimodal projector -> Qwen3 LM`. Projected visual features replace image/video placeholder embeddings before the LM forward.
- Default vision shape is MoonViT-compatible: width 1152, 27 blocks, 16 heads, patch 14, `temporal_patch_size=1`, and `temporal_merge_size=4`. Stage 0 initializes from MoonViT-SO-400M; Stage 1 uses the I3D-ViT checkpoint. The init utility only prefixes compatible ViT keys, so source names/shapes must already match.
- Video patch embedding is per-frame 2D Conv2d, not a 4-frame Conv3d tubelet. Learned spatial position plus learned 4-slot temporal position is added; 2D RoPE has no temporal coordinate and repeats across frames.
- A video's `grid_thw` is split into independent clips of at most 4 frames. Inside each clip, all `t*h*w` patch tokens perform joint full spatial-temporal self-attention. A non-multiple-of-4 tail is one shorter clip.
- All clips from the batch are concatenated into one packed vision forward. FlashAttention uses `cu_seqlens` to evaluate them as independent sequences in one batched call; there is no cross-clip vision attention, KV cache, or recurrent state. Thus this is batched clip processing, not stateful 4-frame streaming.
- After every clip, `patch_merger` averages the temporal axis and groups each 2x2 spatial neighborhood. A full 4-frame clip therefore gets 4x temporal and 4x spatial compression (16x total) before projection. Cross-clip reasoning happens only later in the LM, aided by per-clip timestamp tokens.

## Planned LACT Integration

- Start from a pretrained VideoChat3 VLM and preserve its existing 4-frame joint spatial-temporal attention windows. Add a LACT-style fast-weight memory residual after each vision attention block so consecutive 4-frame clips communicate through recurrent fast state.
- Use one apply-then-update step per 4-frame clip and a zero-initialized memory gate. At initialization the memory residual is exactly zero, preserving the pretrained VideoChat3 function.
- Train on Stage 3 LV and OL data, potentially mixed with a small, still-undecided amount of Academic2M. Freeze the LM, multimodal projector, and every non-vision module; train the vision encoder, including the new fast-weight parameters.

## VideoMamba ViT/LACT Comparison

- Image and video VideoViT share the same pre-norm attention and slow SwiGLU MLP parameter layout. VideoViT adds Conv3d tubelet embedding and temporal position; it uses one global CLS plus all tubelet patch tokens in full-sequence spatial-temporal attention, so it is not streaming.
- VideoLACT uses one `CLS + spatial patches` sequence per embedded tubelet. `kernel_size` is the temporal tubelet size (CLI default 1 for LACT), while `fw_update_group_size=g` groups `g` consecutive tubelets into both one attention window and one fast-weight chunk.
- Every LACT layer is strictly `grouped window attention -> fast-weight SwiGLU memory -> slow SwiGLU MLP`. Attention is local to a group; persistent fast weights carry information to later temporal groups.
- LACT is apply-then-update: group `i` is computed with the old per-sample, per-layer fast state, then its prediction error updates `w0/w1/w2` for group `i+1`. The final update is skipped because no later group consumes it; prediction uses the last tubelet CLS.
- The memory update learns positive per-token rates with `softplus`, computes the three SwiGLU weight updates, applies batched Muon/Newton-Schulz zeroth-power normalization, updates FP32 master weights, and re-normalizes the runtime fast weights. This inner update remains differentiable to the outer supervised loss.
- `fw_update_layer_group_size=1` uses an optimized layer-major scan; values above 1 use chunk-major execution and batch independent layer updates. These are different topological schedules of the same per-layer recurrent dependency, not different model semantics.
- Image VideoViT checkpoints initialize both video models: 2D patch kernels are inflated into Conv3d, and attention/slow-MLP names match. LACT defaults to private memory Q/K/V/O projections; `share_init` copies them from attention and uses a zero memory gate so step 0 preserves the pretrained attention function.
- The current F64 LACT recipe uses `kernel_size=1`, `g=8`, and layer-update group 4: 64 tubelets become 8 temporal groups with 7 effective recurrent updates per layer.
