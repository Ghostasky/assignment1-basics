# RoPE（run_rope）学习笔记

## 1. 这个模块做什么
- RoPE（Rotary Positional Embedding，旋转位置编码）是给 `Q/K` 注入位置信息的方法。
- 白话：把向量里每两个数当作二维坐标 `(x, y)`，再按 token 的位置旋转一个角度。位置不同，旋转不同。

## 2. 在 Transformer 哪一层
- 在 Self-Attention 内部，且在 `QK^T` 之前。
- 顺序是：先 `Q,K` 线性投影 -> 对 `Q,K` 做 RoPE -> 再进 SDPA。

## 3. 输入输出形状
先约定：
- `B`：batch size。
- `T`：sequence length。
- `d_k`：每个 query/key 向量维度。

函数 `run_rope`：
- `in_query_or_key`：`(B, T, d_k)`（或更一般 `(..., T, d_k)`）
- `token_positions`：`(B, T)`（或可广播到 `(..., T)`）
- 输出：`(B, T, d_k)`（与输入同形状）

关键点：
- RoPE 只改数值，不改 shape。
- `d_k` 必须是偶数（因为要两两配对旋转）。

## 4. 数学公式 / 核心逻辑
把最后一维按两两分组：
- 第 `i` 组是 `(x_{2i}, x_{2i+1})`，其中 $i=0,\dots,\frac{d_k}{2}-1$。

第 `i` 组频率：
$$
\omega_i=\theta^{-\frac{2i}{d_k}}
$$

token 位置为 `p` 时，对应角度：
$$
\phi_{p,i}=p\cdot \omega_i
$$

旋转公式（二维旋转）：
$$
x'_{2i}=x_{2i}\cos\phi_{p,i}-x_{2i+1}\sin\phi_{p,i}
$$
$$
x'_{2i+1}=x_{2i}\sin\phi_{p,i}+x_{2i+1}\cos\phi_{p,i}
$$

## 5. “位置信息”到底是什么意思
- 绝对位置：token 在第几个位置（例如 `easy` 在第 3 个）。
- 相对位置：两个 token 间隔多少（例如相差 1、2、3 个位置）。

RoPE 的关键是“相对位置”会进入注意力分数。  
因为 `Q` 在位置 `p` 旋转、`K` 在位置 `q` 旋转，做点积时会自然出现与 $(p-q)$ 相关的项。  
白话：不只是“内容像不像”，还会考虑“离得远不远、前后关系”。

## 6. 从二维到多维（新手友好版）
### 6.1 二维先理解（最重要）
若向量只有两维 `(x, y)`，位置 `p` 对应角度 $\phi$，旋转后：
$$
x'=x\cos\phi-y\sin\phi,\quad y'=x\sin\phi+y\cos\phi
$$

这就是你在平面几何里学过的旋转。

### 6.2 多维怎么做
多维不是一次转完，而是“分很多个二维组分别转”：
- `(x_0,x_1)` 一组
- `(x_2,x_3)` 一组
- ...

每组角度不同（频率不同），最后再按原顺序拼回去。

## 7. 数值例子（完整链路：输入 -> 中间 -> 输出）
设：
- `d_k=4`
- `theta=10000`
- token 位置 `p=3`
- 输入向量 `x=[1, 0, 2, 0]`

步骤 1：分组
- 第 0 组：`(1, 0)`
- 第 1 组：`(2, 0)`

步骤 2：频率
$$
\omega_0=\theta^0=1,\quad \omega_1=\theta^{-1/2}=0.01
$$

步骤 3：角度
$$
\phi_{3,0}=3\times1=3,\quad \phi_{3,1}=3\times0.01=0.03
$$

步骤 4：旋转
- 第 0 组：
$$
x'_0=1\cos3-0\sin3\approx -0.9900,\quad
x'_1=1\sin3+0\cos3\approx 0.1411
$$
- 第 1 组：
$$
x'_2=2\cos0.03-0\sin0.03\approx 1.9991,\quad
x'_3=2\sin0.03+0\cos0.03\approx 0.0600
$$

步骤 5：最终输出
$$
x' \approx [-0.9900,\ 0.1411,\ 1.9991,\ 0.0600]
$$

## 8. 作业对应函数
- `tests/adapters.py::run_rope`

## 9. 最小伪代码（先 for 循环视角）
```python
for i in range(d_k // 2):
    w = theta ** (-(2 * i) / d_k)
    phi = pos * w
    a = x[2 * i]
    b = x[2 * i + 1]
    y[2 * i] = a * cos(phi) - b * sin(phi)
    y[2 * i + 1] = a * sin(phi) + b * cos(phi)
```

再映射到 PyTorch：
```python
x_even = x[..., 0::2]
x_odd = x[..., 1::2]
angles = token_positions[..., None] * inv_freq
out_even = x_even * cos - x_odd * sin
out_odd = x_even * sin + x_odd * cos
out[..., 0::2] = out_even
out[..., 1::2] = out_odd
```

## 10. 对应测试命令
```bash
uv run pytest tests/test_model.py::test_rope -q
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -q
```

## 11. 常见坑
- `d_k` 写成奇数，无法两两分组。
- `token_positions` 形状不对，导致和 `inv_freq` 不能广播。
- 把 `0::2` 和 `1::2` 切错维度（必须是最后一维）。
- 错把 RoPE 用到 `V` 上（一般只对 `Q/K` 用）。
