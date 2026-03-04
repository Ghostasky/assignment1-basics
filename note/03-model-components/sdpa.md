# SDPA（run_scaled_dot_product_attention）学习笔记

## 1. 这个模块做什么
- SDPA（Scaled Dot-Product Attention，缩放点积注意力）是注意力层的核心算子。
- 白话：每个 token 会用自己的 query 去给全句 token 打分，然后按分数加权汇总 value，得到新的上下文表示。

## 2. 在 Transformer 哪一层
- 它在 `Multi-Head Self-Attention` 的每个 head 内部。
- 一层 MHA 会并行算多个 head 的 SDPA，再拼接并线性投影。

## 3. 输入输出形状（先单头，再多头）
先约定术语：
- `B`：batch size，一次并行处理多少句子。
- `T_q`：query 个数。
- `T_k`：key/value 个数。
- `d_k`：query/key 向量维度。
- `d_v`：value 向量维度。

单头（3 维）：
- `Q`：`(B, T_q, d_k)`
- `K`：`(B, T_k, d_k)`
- `V`：`(B, T_k, d_v)`
- `mask`：`(B, T_q, T_k)`（或可广播到这个形状）
- 输出 `O`：`(B, T_q, d_v)`

多头（4 维）：
- `Q`：`(B, H, T_q, d_k)`
- `K`：`(B, H, T_k, d_k)`
- `V`：`(B, H, T_k, d_v)`
- `mask`：`(B, H, T_q, T_k)`（或可广播到这个形状）
- 输出 `O`：`(B, H, T_q, d_v)`

关键提醒（容易混）：
- `vocab_size` 不在 SDPA 这几个张量维度里。
- `vocab_size` 出现在 embedding 表和最终 logits 维度中。

## 4. 数学公式 / 核心逻辑
关键公式：
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$
其中 \(M\) 是 mask 对应的加性偏置（被屏蔽位置可视作 \(-\infty\)）。

核心步骤（与代码顺序一致）：
1. 打分：`scores = Q @ K.transpose(-1, -2)`。
2. 缩放：`scores = scores / sqrt(d_k)`。
3. 掩码：被屏蔽位置填 `-inf`（常见写法 `masked_fill(~mask, -inf)`）。
4. 归一化：`attn = softmax(scores, dim=-1)`。
5. 聚合：`out = attn @ V`。

为什么这样设计：
- 用 `QK^T` 比相关性：让“当前 token（query）”决定看谁（keys）。
- 除 `sqrt(d_k)`：防止分数过大导致 softmax 过尖，训练不稳。
- softmax：把分数变成概率分布，便于解释和训练。
- 乘 `V`：把“看谁”的决策转成“拿什么信息”。

## 5. 用 `I like easy class` 的完整例子
句子 token：`[I, like, easy, class]`，设 `B=1`。

### 5.1 单头视角（3 维）
设 `d_k=d_v=2`，只看 query=`easy` 这一行。

为什么这里取 2：
- 这是教学简化，方便手算，不是模型硬性要求。
- 真实实现中常见 `d_k=d_v=d_head=d_model/H`，通常会比 2 大很多。

输入：
$$
q_{easy}=[1,1]
$$
$$
K=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
0.5&1
\end{bmatrix},\quad
V=
\begin{bmatrix}
1&0\\
0&2\\
2&1\\
1&3
\end{bmatrix}
$$
因果 mask（`easy` 不能看未来 `class`）：
$$
[\mathrm{T},\mathrm{T},\mathrm{T},\mathrm{F}]
$$

步骤 1：点积打分
$$
s=q_{easy}K^\top=[1,1,2,1.5]
$$
步骤 2：缩放（\(d_k=2\)）
$$
\hat{s}=\frac{s}{\sqrt{2}}=[0.707,0.707,1.414,1.061]
$$
步骤 3：应用 mask
$$
\hat{s}_{masked}=[0.707,0.707,1.414,-\infty]
$$
步骤 4：softmax 得注意力权重
$$
\alpha\approx[0.248,0.248,0.504,0]
$$
步骤 5：加权求和输出
$$
o_{easy}=\alpha V\approx[1.256,1.000]
$$

完整链路：
`输入(Q/K/V)` -> `打分` -> `缩放+mask` -> `softmax 权重` -> `最终输出向量`。

### 5.2 多头视角（4 维）
设 `H=2`，每个头 `d_k=d_v=2`。

形状变成：
- `Q, K, V`：`(1, 2, 4, 2)`
- `mask`：`(1, 2, 4, 4)`
- `out`：`(1, 2, 4, 2)`

直觉：
- head 0 可能更关注语法关系（如主谓搭配）。
- head 1 可能更关注语义关系（如修饰词和被修饰词）。
- 两个头各自产生 `(4,2)` 输出，再在 MHA 里拼接成 `(4,4)`（再经输出投影回 `d_model`）。

## 6. 3D/4D 到底怎么算（你最该记住的规则）
一句话规则：
- `Q @ K.transpose(-1, -2)` 只在最后两维做矩阵乘法；
- 前面的维度（如 `B`、`H`）是批维/分组维，各组独立并行计算。

3D（单头）可写成：
$$
\text{scores}[b] = Q[b]\,K[b]^\top
$$
其中：
- `Q[b]` 形状 `(T_q, d_k)`
- `K[b]^T` 形状 `(d_k, T_k)`
- `scores[b]` 形状 `(T_q, T_k)`

4D（多头）可写成：
$$
\text{scores}[b,h] = Q[b,h]\,K[b,h]^\top
$$
其中：
- `Q[b,h]` 形状 `(T_q, d_k)`
- `K[b,h]^T` 形状 `(d_k, T_k)`
- `scores[b,h]` 形状 `(T_q, T_k)`

朴素 for 循环视角：
```python
# 3D
for b in range(B):
    scores[b] = Q[b] @ K[b].transpose(-1, -2)

# 4D
for b in range(B):
    for h in range(H):
        scores[b, h] = Q[b, h] @ K[b, h].transpose(-1, -2)
```

## 7. 形状推导清单（按计算顺序）
你提到的重点是：`Q` 和 `K` 相乘后，再乘 `V`，每一步到底是什么维度。这里给完整版本。

### 7.1 三维（单头）
已知：
$$
Q \in \mathbb{R}^{B\times T_q\times d_k},\quad
K \in \mathbb{R}^{B\times T_k\times d_k},\quad
V \in \mathbb{R}^{B\times T_k\times d_v}
$$

第 1 步：转置 `K` 最后两维
$$
K^\top = K.transpose(-1,-2) \in \mathbb{R}^{B\times d_k\times T_k}
$$

第 2 步：打分
$$
S = QK^\top \in \mathbb{R}^{B\times T_q\times T_k}
$$

第 3 步：缩放与 softmax（形状不变）
$$
\tilde{S} = \frac{S}{\sqrt{d_k}} \in \mathbb{R}^{B\times T_q\times T_k}
$$
$$
A = \mathrm{softmax}(\tilde{S},\ dim=-1) \in \mathbb{R}^{B\times T_q\times T_k}
$$

第 4 步：加权求和（乘 `V`）
$$
O = AV \in \mathbb{R}^{B\times T_q\times d_v}
$$

### 7.2 四维（多头）
已知：
$$
Q \in \mathbb{R}^{B\times H\times T_q\times d_k},\quad
K \in \mathbb{R}^{B\times H\times T_k\times d_k},\quad
V \in \mathbb{R}^{B\times H\times T_k\times d_v}
$$

第 1 步：转置 `K` 最后两维
$$
K^\top = K.transpose(-1,-2) \in \mathbb{R}^{B\times H\times d_k\times T_k}
$$

第 2 步：打分
$$
S = QK^\top \in \mathbb{R}^{B\times H\times T_q\times T_k}
$$

第 3 步：缩放与 softmax（形状不变）
$$
\tilde{S} = \frac{S}{\sqrt{d_k}} \in \mathbb{R}^{B\times H\times T_q\times T_k}
$$
$$
A = \mathrm{softmax}(\tilde{S},\ dim=-1) \in \mathbb{R}^{B\times H\times T_q\times T_k}
$$

第 4 步：加权求和（乘 `V`）
$$
O = AV \in \mathbb{R}^{B\times H\times T_q\times d_v}
$$

一句话总结：
- `Q @ K^T` 后，最后一维从 `d_k` 变成 `T_k`（得到注意力分数）。
- `A @ V` 后，最后一维从 `T_k` 变成 `d_v`（得到上下文向量）。

## 8. 作业对应函数
- `tests/adapters.py::run_scaled_dot_product_attention`

## 9. 最小伪代码
```python
def run_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2)
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)

    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out
```

## 10. 对应测试命令
```bash
uv run pytest tests/test_model.py::test_scaled_dot_product_attention -q
uv run pytest tests/test_model.py::test_4d_scaled_dot_product_attention -q
uv run pytest tests/test_model.py -q
```

## 11. 常见坑（重点）
- 把 `K` 转置写错，必须是 `transpose(-1, -2)`。
- `softmax` 维度写错，必须在 keys 维（`dim=-1`）。
- `mask` 语义写反，导致可见/不可见位置颠倒。
- 忘了除 `sqrt(d_k)`，数值会偏差明显。
- 只支持 3 维不支持 4 维，`test_4d_scaled_dot_product_attention` 会失败。

## 12. 这个教学例子的边界（防误学）
- 这里用 `d_k=2` 只是为了手算直观，真实模型通常更大。
- 这里用了因果 mask；测试里的 `mask` 只是布尔张量，不一定严格因果结构。
- 真实 tokenizer 下 token 不一定等于“一个单词”。
