# rmsnorm

## 1. 这个模块做什么
- RMSNorm（Root Mean Square Normalization）会在最后一维上，把向量按均方根（RMS）缩放到稳定尺度，再乘可学习权重。
- 在本作业的 Pre-Norm Transformer 中，它放在 Attention/FFN 之前，作用是稳定训练而不改变张量形状。

## 2. 输入输出形状
- 输入：
  - `in_features`: `(..., d_model)`，前导维可以是 `(batch, seq)` 或任意形状。
  - `weights`: `(d_model,)`，逐通道缩放参数。
- 输出：
  - `out`: `(..., d_model)`，与 `in_features` 形状一致。

术语速记（容易混）：
- `batch`：一次并行处理多少条句子/样本。
- `seq_len`：每条句子里有多少个 token。
- `d_model`：每个 token 的向量维度（embedding/hidden size）。

为什么叫“最后一维”：
- 若 `x.shape = (batch, seq_len, d_model)`，`d_model` 在最右侧。
- PyTorch 写 `dim=-1` 就是“最右边那一维”，等价于这里的 `d_model` 维。

## 3. 数学公式 / 核心逻辑
- 关键公式：
$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\quad
y_i=\frac{x_i}{\mathrm{RMS}(x)}\cdot w_i
$$
- 核心步骤：
  1. 对最后一维平方并求均值：`mean_sq = (x ** 2).mean(dim=-1, keepdim=True)`。
  2. 加 `eps` 后开方得到分母：`rms = sqrt(mean_sq + eps)`。
  3. 归一化：`x_hat = x / rms`。
  4. 乘缩放权重：`out = x_hat * weights`（广播到前导维）。

为什么这样设计：
- 只控制向量“大小”（scale），不做去均值，计算更简单。
- 在深层 Transformer 中常比 Post-Norm 更稳（本作业采用 Pre-Norm 结构）。
- `eps` 用于避免分母接近 0 的数值不稳定。

通俗例子（`i like easy class`）：
- 假设 `batch=1, seq_len=4, d_model=4`。
- 句子 token 为 `[i, like, easy, class]`，输入形状是 `(1, 4, 4)`。
- 其中 `like` 这个 token 的向量可写成 `[1.0, 2.0, 2.0, 1.0]`。
- `RMSNorm(dim=-1)` 不是只看最后那个 `1.0`，而是把这 4 个数作为一个整体算 RMS，再整体缩放。

## 4. 作业对应函数
- `tests/adapters.py::run_rmsnorm`
- 在 block 中通常对应两处：
  - `h1 = x + SelfAttention(RMSNorm(x))`
  - `h2 = h1 + FFN(RMSNorm(h1))`

## 5. 最小伪代码
```python
def run_rmsnorm(d_model, eps, weights, in_features):
    mean_sq = (in_features ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    normalized = in_features / rms
    out = normalized * weights
    return out
```

## 6. 对应测试命令
```bash
uv run pytest tests/test_model.py::test_rmsnorm -q
uv run pytest tests/test_model.py -q
```

## 7. 常见坑
- 把 RMSNorm 误写成 LayerNorm：LayerNorm 会减均值，RMSNorm 不减均值。
- `mean` 的维度写错：必须在最后一维 `dim=-1`，且 `keepdim=True`。
- 忘记乘 `weights`，或 `weights` 形状不对导致广播错误。
- `eps` 放错位置（应在开方前加到均值项上）。
