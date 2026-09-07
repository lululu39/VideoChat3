# Selecting visual output tokens

The Base and LACT vision configs expose `macro_temporal_compression_mode` and
`macro_temporal_compression_factor`. Selection runs after the complete vision
encoder and the existing temporal/2x2 spatial patch merger. It does not shorten
the input video or change the LACT memory recurrence.

For four chunks with 64 merged spatial tokens each and factor 4:

| Mode | Visual output | Chunk timestamps |
|---|---|---|
| `select_last` | All 64 tokens from chunk 4 | Chunk 4 only |
| `chunk_select_last` | Last 16 original tokens from each of chunks 1–4 | All four |
| `chunk_select_uniform` | 16 evenly spaced original tokens from each chunk | All four |
| `video_last` | All spatial tokens from the video's final chunk | Final chunk only |
| Learned queries (`spatial_quarter`) | 16 learned queries from each chunk | All four |

`chunk_select_last` instantiates no query parameters and inserts no query tokens
into attention or memory. It preserves the original order of the selected
spatial tokens. With factor `R` in `1, 2, 4, 8`, each chunk retains
`max(1, floor(S/R))` of its `S` merged spatial tokens. Factor 1 is an exact no-op.
Short temporal tails retain their own timestamp and token allocation. For
non-divisible `S`, rounding matches the query budget at R4; exact equality to a
full chunk over four chunks applies when `S` is divisible by four. Images are
treated as one chunk, with matching image-placeholder compression.

`chunk_select_uniform` uses the same budget and timestamps, but samples the
flattened post-merger spatial-token order instead of its tail. For `K > 1`,
indices are `floor(i * (S - 1) / (K - 1))`, for `i = 0 .. K-1`, including both
endpoints without randomness. For `K = 1`, the index is `floor(S/2)`. For
`S=64, K=16` the indices are `0,4,8,12,16,21,25,29,33,37,42,46,50,54,58,63`.
This is uniform sampling of the flattened sequence, not a separate 2D lattice.
Select it with `VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE=chunk_select_uniform`;
v33 changes only these selection indices relative to v32.

Configure training with:

```bash
export GPU_EXCLUSIVE=0
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE=chunk_select_last
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=4
export VIDEOCHAT3_LACT_CHUNK_QUERY=0
export VIDEOCHAT3_LACT_CHUNK_QUERY_MODE=single
unset VIDEOCHAT3_CHUNK_QUERY VIDEOCHAT3_CHUNK_QUERY_MODE
```

Or set the same mode/factor on `VideoChat3VisionConfig` or
`VideoChat3LACTVisionConfig`, keeping both query flags disabled. Query modes and
post-encoder selection are mutually exclusive. The selected mode/factor are
saved in HF model and processor configs, so Transformers/VLMEvalKit inference
uses the same token counts and timestamps. When manually changing a loaded
checkpoint's mode, update both its model vision config and processor settings.

v32's launcher uses this mode with v26's parallel Linear16+Delta, joint
ViT/FW/projector training, frame budget, optimizer and 4K packing. It rebuilds a
mode-specific token-count cache; expected counts and packing match v26's
`spatial_quarter` queries, including short tails. This comparison removes the
whole learned-query interface, including query participation in attention;
it is not solely a replacement of a final pooling operation.
