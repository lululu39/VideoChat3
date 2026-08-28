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

先看一条没有其他分支的纯链。若：

$$
b_{t-1}=J_t^\top b_t,
\qquad
q_t=\frac{\lVert b_{t-1}\rVert}{\lVert b_t\rVert},
$$

那么相邻范数会消去，得到严格等式：

$$
\frac{\lVert b_0\rVert}{\lVert b_T\rVert}
=
\prod_{t=1}^{T}q_t.
$$

例如每一步在其实际输入梯度方向上都放大 $1.2$ 倍，经过 32 步就是
$1.2^{32}\approx341$ 倍。这就是 BPTT 中“每步小幅放大，长链后爆炸”的来源。

LACT 的完整 transition 不是只有 NS5。记 InnerGrad、NS5 和 Normalize 的 Jacobian
分别为 $H_t$、$P_t$ 和 $N_t$，则：

$$
J_t
=
N_t\left(I+P_tH_t\right),
$$

对应的 future backward 为：

$$
J_t^\top b_t
=
\left(I+H_t^\top P_t^\top\right)N_t^\top b_t.
$$

这里的 $I$ 是 residual state 分支，$H_t^\top P_t^\top$ 是经过 FW update 的分支。
把连续多个括号展开后会得到许多路径；连续选择 FW update 分支的路径会包含多个
$P_t^\top$，也就是多个 NS5 backward。局部 NS5 amplification ratio $r_t$ 的乘积
描述的是这些路径中的 NS5 因素，而不是完整 $J_t$ 的全部增益。

纯文本：完整 transition 的 $q_t$ 沿一条纯 future chain 连乘是严格等式；NS5 的
$r_t$ 只是在展开后某条路径中的局部因素。实际 BPTT 还会加上各 group 的直接 loss
梯度，因此最终梯度不是单独一个乘积。

## 3. NS5 为什么会放大 backward

先看一次 fast-weight update 中 NS5 这个局部算子。设传给 NS5 的原始更新矩阵为
$G_t$，NS5 输出的归一化更新矩阵为 $U_t$：

$$
U_t = \operatorname{NS5}(G_t),
\qquad
G_t,U_t\in\mathbb{R}^{m\times n}.
$$

为了把矩阵到矩阵的导数写清楚，先将两个矩阵展平成向量：

$$
g_t=\operatorname{vec}(G_t),
\qquad
u_t=\operatorname{vec}(U_t).
$$

NS5 在 $G_t$ 处的 Jacobian 定义为：

$$
J_{\operatorname{NS5}}(G_t)
=
\frac{\partial u_t}{\partial g_t}
\in\mathbb{R}^{mn\times mn}.
$$

也就是说，它的第 $(i,j)$ 个元素为：

$$
\left[J_{\operatorname{NS5}}(G_t)\right]_{ij}
=
\frac{\partial (u_t)_i}{\partial (g_t)_j}.
$$

设最终训练 loss 为 $\mathcal L$。从 NS5 后面的计算传回来的上游梯度，以及继续传给
NS5 前面计算的梯度，分别记为：

$$
\bar u_t=\frac{\partial\mathcal L}{\partial u_t},
\qquad
\bar g_t=\frac{\partial\mathcal L}{\partial g_t}.
$$

根据 reverse-mode 链式法则：

$$
\frac{\partial\mathcal L}{\partial(g_t)_j}
=
\sum_i
\frac{\partial(u_t)_i}{\partial(g_t)_j}
\frac{\partial\mathcal L}{\partial(u_t)_i},
$$

写成列向量形式就是：

$$
\boxed{
\bar g_t
=
J_{\operatorname{NS5}}(G_t)^\top\bar u_t
}.
$$

这就是 reverse-mode 的 vector-Jacobian product（VJP）的列向量写法。转置来自上面
分量形式中的指标顺序，而不是额外引入的算子。将 $\bar u_t$ 和 $\bar g_t$ reshape 回矩阵后，分别写成
$\bar U_t$ 和 $\bar G_t$。实现中的 `grad_update` 对应 $\bar U_t$，原始精确
backward 得到的 `exact_grad` 对应 $\bar G_t$。

直观地说，VJP 回答的是：“已经知道 loss 希望 NS5 输出 $U_t$ 怎样变化，那么 loss
希望 NS5 输入 $G_t$ 怎样变化？”自动微分不需要构造巨大的 $mn\times mn$ Jacobian，
只计算当前上游梯度所需的乘积。在 PyTorch 记法中，概念上相当于：

```python
grad_G = torch.autograd.grad(
    outputs=U,
    inputs=G,
    grad_outputs=grad_U,
)
```

数学文献常把行向量形式 $\bar u_t^\top J$ 直接称为 VJP；本文使用等价的列向量形式
$J^\top\bar u_t$。

这个 VJP 可能放大的原因来自 NS5 的第一步。NS5 首先归一化输入：

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
\frac{\lVert\bar G_t\rVert_F}{\lVert\bar U_t\rVert_F}
=
\frac{
\left\lVert J_{\operatorname{NS5}}(G_t)^\top\bar u_t\right\rVert_2
}{\lVert\bar u_t\rVert_2}.
$$

这里矩阵的 Frobenius norm 等于其向量化结果的 Euclidean norm，所以两个比值完全
相同。这个 $r_t$ 不是整个 Jacobian 的最大奇异值，而是当前训练梯度方向上真正发生的
局部放大倍数。

稳定化之前，展开后的长 BPTT 路径可能包含多个大于 1 的 $r_t$ 因素。受控实验中，
8 次 update 只有轻微增长，16 次开始快速增长，32 次发生严重爆炸。

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
\bar G_{\mathrm{exact}}
=
\operatorname{reshape}\!\left(
J_{\operatorname{NS5}}(G_t)^\top\bar u_t
\right).
$$

对每个独立 NS5 矩阵计算：

$$
r =
\frac{\lVert\bar G_{\mathrm{exact}}\rVert_F}
     {\lVert\bar U_t\rVert_F}.
$$

固定最大 amplification：

$$
\rho = 1.
$$

最终返回的梯度为：

$$
\bar G =
\begin{cases}
\bar G_{\mathrm{exact}}, & r \le \rho, \\[6pt]
\bar G_{\mathrm{exact}}\dfrac{\rho}{r}, & r > \rho.
\end{cases}
$$

等价形式为：

$$
\bar G = \bar G_{\mathrm{exact}}
\min\left(
1,
\frac{\rho\lVert\bar U_t\rVert_F}
     {\lVert\bar G_{\mathrm{exact}}\rVert_F}
\right).
$$

因此在浮点误差范围内严格满足：

$$
\boxed{\lVert\bar G\rVert_F \le \lVert\bar U_t\rVert_F}.
$$

实现按 batched NS5 中的每个矩阵独立约束，包括每个 video、FW head 以及
`w0`、`w1`、`w2` 的独立更新矩阵。

## 5. 它如何消除已观察到的指数放大

修改前，展开后某条长 BPTT 路径中的 NS5 因素可能满足：

$$
r_1r_2\cdots r_T \gg 1.
$$

修改后：

$$
r_t^{\mathrm{clipped}} \le 1
\quad\Longrightarrow\quad
\prod_{t=1}^{T}r_t^{\mathrm{clipped}} \le 1.
$$

因此 NS5 局部 VJP 本身不再沿时间路径提供大于 1 的连乘因素。每个 group 的直接梯度
仍会进入 recurrence，跨 group 的长期梯度也仍然非零。这不同于 detach 或 TBPTT；
后两者会把跨边界梯度直接设为零。这个结论不代表完整 transition 一定 non-expansive：
$H_t$、$N_t$、residual 分支及不同分支相加仍可能改变完整梯度范数。

## 6. 保持不变与发生改变的语义

以下行为保持不变：

- NS5 forward 数值。
- FW apply/update 数值。
- Fast/master state trajectory。
- Vision encoder 输出。
- Activation checkpoint 重算值。
- Inference 行为。
- $r\le1$ 时的精确 backward。

当 $r>1$ 时，只缩放 $\bar G_{\mathrm{exact}}$ 的 magnitude，方向保持不变。
`lr_proj` 和 value/update projections 仍然收到非零梯度。

代价是：触发 clipping 后，返回的是有偏 surrogate gradient，而不是原始 NS5
forward 的精确导数。此外，该保证只作用于 NS5 局部 VJP，并不能证明完整 transition
满足：

$$
\lVert\bar S_{t-1}\rVert_F \le \lVert\bar S_t\rVert_F.
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
