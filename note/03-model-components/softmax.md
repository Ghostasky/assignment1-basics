# softmax（`run_softmax`）

## 1. 它是什么
- `softmax` 把一组实数（logits）变成一组概率分布。
- 直观上：把“每个类别的原始分数”变成“每个类别的相对概率”。

## 2. 在 Transformer 哪一层会用到
- 最典型在 Attention 里：
  - 先算注意力分数 `scores = QK^T / sqrt(d_k)`。
  - 再对 `scores` 的最后一维做 `softmax`，得到注意力权重。

## 3. 公式与 shape
- 公式（对某个维度 `dim`）：
  - `softmax(x_i) = exp(x_i) / sum_j exp(x_j)`
- 输入输出 shape：
  - 输入：`in_features`，任意 shape（例如 `[B, T, V]` 或 `[B, H, T, T]`）。
  - 输出：shape 与输入完全相同。
  - 归一化发生在 `dim` 上：该维每个切片的和为 1。

### 一个直观例子（`[B, T, V]`）
- 设 `x.shape = [2, 2, 3]`，把它看成一个三维长方体：
  - `B=2`：两块样本切片
  - `T=2`：每块里两行
  - `V=3`：每行三个分数
- 输入：
  - `x[0] = [[1, 2, 3], [2, 4, 0]]`
  - `x[1] = [[0, 1, 0], [3, 3, 3]]`
- 若执行 `softmax(x, dim=2)`（沿 `V` 归一化），输出约为：
  - `y[0] = [[0.0900, 0.2447, 0.6652], [0.1173, 0.8668, 0.0159]]`
  - `y[1] = [[0.2119, 0.5761, 0.2119], [0.3333, 0.3333, 0.3333]]`
- 每个 `(b, t)` 位置上的 3 个值之和都为 1。

## 4. 为什么要做“数值稳定”
- 问题：`exp(很大数)` 会溢出成 `inf`。
- 稳定写法：先减去该维最大值再 `exp`：
  - `x_stable = x - max(x, dim, keepdim=True)`
  - 再做 `exp(x_stable) / sum(exp(x_stable))`
- 为什么这样不改变结果：
  - 分子分母都乘了同一个常数 `exp(-max)`，比例不变。
- 这也是测试里 `x` 和 `x + 100` 结果应一致的原因。

## 5. `run_softmax` 实现要点（手写）
- 只在指定 `dim` 上归一化，不要写死最后一维。
- `max` 和 `sum` 要 `keepdim=True`，否则广播可能出错。
- 返回张量 dtype 与输入兼容（通常沿用输入 dtype）。

### `keepdim=True` 是什么
- 含义：按某个维度做 `max/sum` 后，保留这个维度（长度变成 1），而不是删除这个维度。
- 例子：`x.shape=[2,3,4]`，对 `dim=2` 求和
  - `keepdim=False` -> 结果 `shape=[2,3]`
  - `keepdim=True` -> 结果 `shape=[2,3,1]`
- 在 softmax 里的作用：
  - `max_vals = x.max(dim=dim, keepdim=True).values`
  - `denom = exp_vals.sum(dim=dim, keepdim=True)`
  - 这样 `x - max_vals`、`exp_vals / denom` 都能自动按正确维度广播，shape 对齐更稳定。

## 6. 常见错误
- 忘记减最大值，导致大输入溢出。
- `dim` 写错（例如固定 `-1`，但测试传别的 dim 会错）。
- `keepdim=False` 导致除法广播错位。
- 用 in-place 改写输入，影响后续计算图。

## 7. 最小测试命令
```bash
uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q
```

## 8. 通过标准
- 与 `torch.nn.functional.softmax` 数值一致（`atol=1e-6`）。
- 对平移输入（如 `x + 100`）输出不变（容差内）。
