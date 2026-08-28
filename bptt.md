# LACT 长序列 BPTT 梯度稳定性

## 1. Forward 状态更新

对第 $t$ 个 4-frame group，定义：

- $S_{t-1}$：更新前的 fast-weight state。
- $G_t$：当前 group 产生的 inner gradient。
- $U_t$：Muon/Newton-Schulz 生成的更新方向。
- $S_t$：供下一个 group 使用的 state。

Forward recurrence 为：

$$
\begin{aligned}
G_t &= \operatorname{InnerGrad}(S_{t-1}, x_t), \\
U_t &= \operatorname{NS5}(G_t), \\
S_t &= \operatorname{Normalize}(S_{t-1} + U_t).
\end{aligned}
$$

纯文本：`G_t = InnerGrad(S_(t-1), x_t)`，`U_t = NS5(G_t)`，
`S_t = Normalize(S_(t-1) + U_t)`。

当前 group 使用旧 state 产生输出：

$$
y_t = \operatorname{Apply}(S_{t-1}, x_t).
$$

所有 group 的 $y_t$ 都会送入语言模型并共同影响最终监督 loss。

## 2. BPTT 中的连乘从哪里产生

记到达 state $S_t$ 的梯度为：

$$
a_t = \frac{\partial L}{\partial S_t}.
$$

反向到前一个 state 时，当前输出与未来 state 的两条梯度分支相加：

$$
\begin{aligned}
a_{t-1}
&=
\underbrace{
\left(\frac{\partial y_t}{\partial S_{t-1}}\right)^\top
\frac{\partial L}{\partial y_t}
}_{\text{当前 group 的直接梯度}}
\\
&\quad+
\underbrace{
\left(\frac{\partial S_t}{\partial S_{t-1}}\right)^\top a_t
}_{\text{从未来 state 返回的梯度}}.
\end{aligned}
$$

展开未来分支后会出现 state-transition Jacobian 的连乘：

$$
J_1^\top J_2^\top \cdots J_T^\top a_T,
\qquad
J_t = \frac{\partial S_t}{\partial S_{t-1}}.
$$

纯文本：梯度在分支处相加，在连续 state-transition 边上相乘。因此每步略大于
1 的放大系数也可能随 FW update 数量指数增长。

## 3. NS5 为什么会放大 backward

每次 FW update 都包含下面的 VJP：

$$
dG_t = J_{\operatorname{NS5}}(G_t)^\top dU_t.
$$

NS5 首先归一化输入：

$$
X_0 = \frac{G_t}{\lVert G_t\rVert_F + \epsilon}.
$$

其导数尺度包含近似项：

$$
\frac{1}{\lVert G_t\rVert_F + \epsilon}.
$$

当 $G_t$ 很小时，NS5 forward 仍然有界，但 backward gain 可能非常大。定义实际
VJP 方向上的 amplification ratio：

$$
r_t =
\frac{
\left\lVert J_{\operatorname{NS5}}(G_t)^\top dU_t \right\rVert_F
}{
\lVert dU_t\rVert_F
}.
$$

稳定化之前，长 BPTT 路径可能包含 $\prod_{t=1}^{T}r_t$。受控实验中，8 次
update 只有轻微增长，16 次开始快速增长，32 次发生严重爆炸。

## 4. Bounded NS5 backward

模型选项为：

```python
clip_ns_grad_ratio = True  # 默认启用
```

NS5 forward 完全不变：

$$
U_t = \operatorname{NS5}(G_t).
$$

Backward 首先重新计算精确 VJP：

$$
dG_{\mathrm{exact}}
=
J_{\operatorname{NS5}}(G_t)^\top dU_t.
$$

对每个独立 NS5 矩阵计算：

$$
r =
\frac{\lVert dG_{\mathrm{exact}}\rVert_F}
     {\lVert dU_t\rVert_F}.
$$

固定最大 amplification：

$$
\rho = 1.
$$

最终返回的梯度为：

$$
dG =
\begin{cases}
dG_{\mathrm{exact}}, & r \le \rho, \\[6pt]
dG_{\mathrm{exact}}\dfrac{\rho}{r}, & r > \rho.
\end{cases}
$$

等价形式为：

$$
dG = dG_{\mathrm{exact}}
\min\left(
1,
\frac{\rho\lVert dU_t\rVert_F}
     {\lVert dG_{\mathrm{exact}}\rVert_F}
\right).
$$

因此在浮点误差范围内严格满足：

$$
\boxed{\lVert dG\rVert_F \le \lVert dU_t\rVert_F}.
$$

实现按 batched NS5 中的每个矩阵独立约束，包括每个 video、FW head 以及
`w0`、`w1`、`w2` 的独立更新矩阵。

## 5. 它如何消除已观察到的指数放大

修改前，长 BPTT 路径中的 NS5 部分可能满足：

$$
r_1r_2\cdots r_T \gg 1.
$$

修改后：

$$
r_t^{\mathrm{clipped}} \le 1
\quad\Longrightarrow\quad
\prod_{t=1}^{T}r_t^{\mathrm{clipped}} \le 1.
$$

因此 NS5 不再向时间方向的 Jacobian product 提供指数放大。每个 group 的直接梯度
仍会进入 recurrence，跨 group 的长期梯度也仍然非零。这不同于 detach 或 TBPTT；
后两者会把跨边界梯度直接设为零。

## 6. 保持不变与发生改变的语义

以下行为保持不变：

- NS5 forward 数值。
- FW apply/update 数值。
- Fast/master state trajectory。
- Vision encoder 输出。
- Activation checkpoint 重算值。
- Inference 行为。
- $r\le1$ 时的精确 backward。

当 $r>1$ 时，只缩放 $dG_{\mathrm{exact}}$ 的 magnitude，方向保持不变。
`lr_proj` 和 value/update projections 仍然收到非零梯度。

代价是：触发 clipping 后，返回的是有偏 surrogate gradient，而不是原始 NS5
forward 的精确导数。此外，该保证只作用于 NS5 局部 VJP，并不能证明完整 transition
满足：

$$
\lVert dS_{t-1}\rVert_F \le \lVert dS_t\rVert_F.
$$

因为 inner-gradient 构造和 state normalization 仍位于被约束的 NS5 VJP 之外。

使用以下配置可恢复原始精确 NS5 backward：

```python
clip_ns_grad_ratio = False
```

## 7. 受控验证

固定微型模型使用相同输入，并设置 memory gate 为 $5\times10^{-5}$：

| Backward 模式 | 8 groups | 16 groups | 32 groups |
|---|---:|---:|---:|
| 原始 NS5 | $8.42\times10^{-3}$ | $8.15\times10^{-1}$ | $1.11\times10^{4}$ |
| NS5 ratio clipping | $3.45\times10^{-4}$ | $4.15\times10^{-4}$ | $4.15\times10^{-4}$ |
| NS steps = 0 | $3.29\times10^{-4}$ | $3.34\times10^{-4}$ | $3.76\times10^{-4}$ |

稳定化后的梯度接近没有 NS amplification 时的自然尺度，并且从 8 到 32 次 update
保持平坦。真实 VLM 实验仍需继续监控完整 state-transition ratio、FW/base-ViT
gradient norm、memory gate 增长以及有效 FW residual 大小。
