# swiglu_ffn

## 1. 这个模块做什么
- SwiGLU 是 Transformer 里的 FFN 变体：先升维到 `d_ff`，做门控，再降回 `d_model`。
- 直觉：一条支路产生“门值”，另一条支路提供“信息”，两者逐元素相乘后再输出。

## 2. 输入输出形状
- 输入：`x`，形状 `(..., d_model)`
- 中间：
  - `a = x @ W1^T`，形状 `(..., d_ff)`
  - `b = x @ W3^T`，形状 `(..., d_ff)`
  - `h = SiLU(a) * b`，形状 `(..., d_ff)`
- 输出：`y = h @ W2^T`，形状 `(..., d_model)`

## 3. 数学公式 / 核心逻辑
关键公式：
$$
\mathrm{SwiGLU}(x)=\left(\mathrm{SiLU}(xW_1^\top)\odot(xW_3^\top)\right)W_2^\top
$$

其中：
$$
\mathrm{SiLU}(z)=z\cdot \sigma(z)
$$

核心步骤（与代码顺序一致）：
1. 上投影两次：得到 `a` 和 `b`
2. 对 `a` 做 SiLU 得到门值 `g`
3. 门控逐元素乘：`h = g * b`
4. 下投影：`y = h @ W2^T`

## 4. 常见疑问（学习记录）
- 问：ReLU-FFN 和 SwiGLU-FFN 的结构差异是什么？
  - 答：ReLU-FFN 是单支路：`x -> W1 -> ReLU -> W2`。
  - 答：SwiGLU 是双支路：`x -> W1 -> SiLU` 与 `x -> W3`，两路逐元素相乘后再过 `W2`。
- 问：什么是“门控逐元素乘”？
  - 答：同位置乘同位置，不是矩阵乘法。
  - 例子：`g=[0.2, 0.8, -0.5]`，`b=[10, 3, 4]`，则 `h=g*b=[2, 2.4, -2]`。
- 问：为什么叫“门控”？
  - 答：`g` 像阀门；`g` 接近 0 会抑制该维度，`g` 较大时放行/放大，`g` 为负时可反向该维度信号。

## 5. 作业对应函数
- `tests/adapters.py::run_swiglu`

## 6. 最小代码（可直接写进 run_swiglu）
```python
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    a = in_features @ w1_weight.T
    b = in_features @ w3_weight.T
    h = run_silu(a) * b
    out = h @ w2_weight.T
    return out
```

## 7. 按实现顺序的数字例子（完整链路）
设 `d_model=2`，`d_ff=3`，输入：
$$
x=[1,-2]
$$

设：
$$
W_1=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
W_3=\begin{bmatrix}1&-1\\2&0\\0&1\end{bmatrix},\quad
W_2=\begin{bmatrix}1&0&-1\\0.5&1&0\end{bmatrix}
$$

步骤1：上投影
$$
a=xW_1^\top=[1,-2,-1],\quad b=xW_3^\top=[3,2,-2]
$$

步骤2：激活
$$
\mathrm{SiLU}(a)\approx[0.7311,-0.2384,-0.2689]
$$

步骤3：门控逐元素乘
$$
h=\mathrm{SiLU}(a)\odot b\approx[2.1933,-0.4768,0.5378]
$$

步骤4：下投影
$$
y=hW_2^\top\approx[1.6555,0.6199]
$$

最终输出就是这个 `d_model=2` 向量。

## 8. 可手写实现步骤（不改代码）
1. `a = in_features @ w1_weight.T`
2. `b = in_features @ w3_weight.T`
3. `g = run_silu(a)`（或 `torch.nn.functional.silu(a)`）
4. `h = g * b`
5. `out = h @ w2_weight.T`
6. `return out`

## 9. 对应测试命令
```bash
uv run pytest tests/test_model.py::test_swiglu -q
uv run pytest tests/test_model.py -k "silu or swiglu" -q
uv run pytest -q
```

## 10. 常见坑
- 忘记 `.T`（权重是 `d_out x d_in` 存的）。
- 把“逐元素乘”误写成矩阵乘法。
- 先后顺序写错：必须先 `SiLU(a)`，再和 `b` 相乘，最后过 `W2`。
