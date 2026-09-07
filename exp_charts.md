# TimeLens Experiment Chart

本表汇总所有使用 TimeLens-100K 训练的 canonical 编号实验 `v9-v31`，并加入未训练 Base R4 作为固定参考。除注明外，训练数据均为 seed-42 的 12,624-row random-half manifest，评测均为同一 TimeLens-Bench native generation/scoring 协议（2 FPS、最多 448 帧、224px/14,680,064 total pixels）。内部系统 smoke 合并在对应版本中，不单列为学习实验。

记号：`V` = 原始 ViT，`F` = LACT FW（含 gate；query 实验也含 query bank），`P` = projector；`GN` 是训练日志中的全局 pre-clip gradient norm，格式为 `mean / median / max`；CE 是最后 20 步均值（短暂停实验显示最后一步及 step）；分数均为百分数，`—` 表示没有对应 checkpoint 或没有执行 TimeLens-Bench。`Base R4 ref` 是未训练的 `/mnt/localssd/VideoChat3/VideoChat3-4B-R4-mean-init`，不是 v10 checkpoint。

| Version | Status | Trainable | LM-facing visual output | Run 特点 | Steps / pack | CE | GN mean / median / max | Gate | Charades mIoU | ActivityNet mIoU | QVHighlights mIoU |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Base R4 ref | Reference | 无训练 | 每 4 chunks 做 mean，保留 1 个完整空间网格 | 未训练固定基线；后续 Base-vs-LACT 表的 R4 reference | — | — | — | — | `31.48` | `33.59` | `50.42` |
| v9 | Stopped, no ckpt | F | R1：所有 chunk 空间 token | 旧 SwiGLU+Muon、FW-only、NS5+state clip；TimeLens 首个训练诊断，速度慢且 loss 基本不降 | `87/417`, 8K | `0.438@87` | `1.059 / 0.823 / 4.721` | — | — | — | — |
| v10 | Complete; core eval only | V+P | R4 mean | Base ViT+projector 训练；没有 FW；未对 v10 checkpoint 跑 TimeLens-Bench，core eval 反而低于未训练 R4 | `93/93`, 8K | `0.318` | `3.201 / 2.576 / 12.600` | — | — | — | — |
| v11 | Stopped, no ckpt | F+P | R4 select：每 4 chunks 保留末 chunk 空间网格 | 旧 SwiGLU+Muon；正确但约 12 小时、系统成本过高；引出 Linear+Delta | `15/455`, 2K | — | `约 3.35-3.55（smoke）` | — | — | — | — |
| v12 | Complete | F+P | R4 select | Linear16+Delta、serial、gate=0、无 FW ratio clip；首次兼顾可训练性与完整时间覆盖 | `93/93`, 8K | `0.331` | `0.749 / 0.509 / 3.077` | `1.60e-4` | `26.63` | `27.98` | `39.53` |
| v13 | Complete | F+P | video-last：整段视频只留末 chunk 网格 | v12 改成 final-chunk-only；验证 recurrent state 不能替代直接视觉 token | `114/114`, 1K | `0.390` | `0.471 / 0.209 / 3.744` | `2.65e-4` | `10.41` | `7.19` | `3.82` |
| v14 | Complete | V+P | Base video-last | 无 FW 的匹配 final-chunk control；说明主要问题是输出瓶颈 | `114/114`, 1K | `0.386` | `0.467 / 0.213 / 8.287` | — | `11.80` | `7.74` | `3.06` |
| v15 | Complete | V+P | Base R1：所有视觉 token | 全 token 上界；当前三个子集最高结果 | `417/417`, 8K | `0.233` | `9.190 / 7.780 / 44.750` | — | `41.39` | `43.57` | `55.00` |
| v16 | Stopped, no ckpt | F+P | video-last | serial Linear16+Delta；加入 FW Q/K 3D RoPE，zero tanh gate 仍关闭分支 | `25/114`, 1K | `0.414@25` | `1.407 / 1.324 / 3.709` | — | — | — | — |
| v17 | Complete | F+P | video-last | serial、3D RoPE、linear gate init=0.5；强行打开 FW residual | `114/114`, 1K | `0.389` | `0.405 / 0.201 / 2.958` | `0.5`（FP32 Δ RMS `1.76e-4`） | `9.21` | `5.57` | `3.68` |
| v18 | Complete | F+P | video-last | v17 去掉 3D RoPE；Charades 改善但整体仍受 final-chunk 瓶颈限制 | `114/114`, 1K | `0.387` | `0.427 / 0.227 / 2.754` | `0.5`（FP32 Δ RMS `1.77e-4`） | `13.22` | `7.05` | `2.92` |
| v19 | Complete | F+P | video-last | parallel、3D RoPE、gate=0；首次 parallel topology | `114/114`, 1K | `0.390` | `0.476 / 0.216 / 3.762` | `2.60e-4` | `10.54` | `7.25` | `3.91` |
| v20 | Stopped, no ckpt | F+P | video-last | v19 的 cosine 改为 warmup 后 constant peak LR；无有效早期差异 | `25/114`, 1K | `0.414@25` | `1.404 / 1.324 / 3.746` | — | — | — | — |
| v21 | Stopped, no ckpt | F+P | video-last | 回到 cosine；gate LR 单独放大 100x；改善与 noise 同量级 | `27/114`, 1K | `0.403@27` | `1.216 / 0.550 / 3.599` | — | — | — | — |
| v22 | Stopped, no ckpt | F+P | video-last | v21 + gate init=0.5；branch 初始活跃但 trajectory 仍无价值 | `15/114`, 1K | `0.410@15` | `1.606 / 0.881 / 4.782` | init `0.5` | — | — | — |
| v23 | Complete | F+P | 每 chunk 1 个 learned query | parallel、3D RoPE、gate=0；query read FW、不写 FW；恢复时间覆盖但容量太低 | `413/413`, 1K | `0.372` | `0.672 / 0.585 / 2.534` | `3.29e-4` | `12.51` | `8.48` | `6.60` |
| v24 | Complete | F+P | 每 chunk `floor(S/4)` queries | parallel、Base-R4-budget queries（224px 为 16/chunk）；ViT 冻结 | `276/276`, 4K | `0.365` | `3.750 / 2.758 / 46.042` | `2.92e-4` | `15.60` | `11.30` | `7.78` |
| v25 | Complete | F+P | 每 chunk `floor(S/4)` queries | v24 的 bitwise-matched serial topology control；ViT 冻结时 serial/parallel 差异很小 | `276/276`, 4K | `0.365` | `4.910 / 4.069 / 32.316` | `2.90e-4` | `15.90` | `12.21` | `7.85` |
| v26 | Complete | V+F+P | 每 chunk `floor(S/4)` queries | parallel；在 v24 上解冻 ViT，形成成功的 ViT-query-FW 联合适配 | `276/276`, 4K | `0.261` | `7.110 / 4.380 / 164.890` | `2.03e-4` | **`36.98`** | **`37.55`** | **`48.75`** |
| v27 | Complete | V+P | 每 chunk `floor(S/4)` queries | 真 Base+query，无任何 FW/memory/gate；测试 ViT adaptation 单独是否足够 | `276/276`, 4K | `0.389` | `3.154 / 0.164 / 80.866` | — | `14.07` | `7.13` | `4.82` |
| v28 | Complete | V+F+P | 每 chunk `floor(S/4)` queries | v26 的 bitwise-matched serial control；修复 loader 后 step-1 CE 精确匹配 v26 | `276/276`, 4K | `0.391` | `30.249 / 0.195 / 634.806` | `2.05e-4` | `15.37` | `6.96` | `6.15` |
| v29 | Stopped, no ckpt | V+F+P | video-last | CPU offload 修复 OOM 后运行至 step 15；用户停止并转向 v26 复现，无原生评测 | `15/114`, 1K | `0.408611@15` | `1.783 / 2.277 / 3.236` | — | — | — | — |
| v30 | Stopped; DCP only | V+F+P | 每 chunk `floor(S/4)` queries | v26 同 seed 复现至 step 132；末20步 CE `0.2944` vs 同区间 v26 `0.2831`，用户接受训练趋势后停止，无原生评测 | `132/276`, 4K | `0.311770@132` | `9.182 / 4.056 / 75.855` | — | — | — | — |
| v31 | Prepared | V+F+P | 每 chunk 1 个 learned query | v23 唯一模型/优化变化为解冻 ViT，验证单 query 压缩的联合适配 | `0/413`, 1K | — | — | init `0` | — | — | — |

## 从表格可以归纳的规律

1. **直接视觉覆盖是前提，FW 不能凭空补回被删掉的观察。** v13/v14 的 video-last mIoU 都只有约 `3-12`；把 gate 设为 `0.5`（v17/v18）、加入 3D RoPE（v16/v17）、改 parallel（v19）都无法修复。最终 chunk 之前的信息即使经过 recurrent state，也不足以代替直接交给 LLM 的视觉 token。

2. **更多 token 有用，但“数量相同”不等于“信息等价”。** 每 chunk 1 query 的 v23 比 video-last 略好；把 query 数提高到 Base-R4 budget 的 v24 又提高 `1.18-3.09` mIoU，但仍远低于 Base R4。Learned query 必须学会提取空间/事件信息，不能仅靠匹配 token count。

3. **ViT-only 和 FW-only 都没用，二者结合且使用 parallel 才有用。** 冻结 ViT 的 parallel v24 为 `15.60/11.30/7.78`；移除 FW、只训 ViT/query/projector 的 v27 为 `14.07/7.13/4.82`；两者结合的 parallel v26 跃升到 `36.98/37.55/48.75`。这不是简单相加，而是强协同：ViT 学会生成适合 query/FW 的表示，FW 提供跨 chunk 通道。

4. **这种协同严格依赖 parallel topology。** ViT 冻结时 v24/v25 的 parallel/serial 几乎持平；ViT 一起训练后，bitwise-matched serial v28 相对 parallel v26 损失 `21.61/30.59/42.60` mIoU，并退化到接近无 FW 的 v27。有效结构是让 window attention 与 FW 同时读取稳定的 pre-attention layer input；serial FW 读写不断变化的 post-attention 表示，在 joint training 中出现大幅早期 clipping，随后落入低梯度、高 loss 解。

5. **Gate 数值大小不能衡量 FW 是否有用。** v26/v28 gate RMS 几乎一样（`2.03e-4/2.05e-4`），性能却相差 `22-43` 点；v17/v18 的有效 gate 约为 `0.5` 仍然很差；100x gate LR（v21）与 gate=0.5+100x LR（v22）也没有形成有意义趋势。应看 FW 输出尺度、信息内容和端到端行为，而不是 gate 单独大小。

6. **3D RoPE、scheduler 和单独放大 gate LR 都不是主解。** v17 对比 v18 表明 3D RoPE 在 final-chunk serial 条件下甚至伤害 Charades/ActivityNet；v20 的 constant peak LR 与 v19 早期几乎一致；v21/v22 没有证明 gate LR 或初始化能够突破输出瓶颈。

7. **Grad norm 是优化状态指标，不是能力指标。** v12 的 GN mean 只有 `0.749` 但显著好于许多后续 final-chunk run；成功的 v26 mean 为 `7.110`；失败的 v28 mean 高达 `30.249`、max `634.806`，但 median 只有 `0.195`。高 norm 可能来自有效密集监督，也可能是早期不稳定后被 global clip 扭曲，不能单独作为 FW 学习成功的证据。

8. **训练 CE 只有在出现大幅分离时才有诊断价值。** v13-v25 多数失败 run 的末段 CE 都聚集在 `0.36-0.41`，无法可靠排序；v26 降到 `0.261`，同时 native eval 大幅提升，而 matched v27/v28 仍为 `0.389/0.391`。因此显著的 CE 下降支持成功，但细小 CE 差异通常不足以预测 benchmark。

9. **当前质量排序说明仍有压缩损失。** v26 已超过 Base R4 reference 的 Charades/ActivityNet mIoU `5.50/3.96` 点，并只落后 QVHighlights `1.67` 点；但仍低于全视觉 token v15 的 `4.41/6.02/6.25` 点。Parallel ViT+FW+query 是目前最有效的压缩方案，但尚未达到 full-token 上界。

详细的训练设置、checkpoint、R1@0.3/0.5/0.7、artifact 路径及逐实验结论见 [`exp_results.md`](exp_results.md)。
