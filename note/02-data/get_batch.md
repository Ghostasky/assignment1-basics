# get_batch

## 1. 这个模块做什么
- `get_batch` 用来从一条长 token 序列中，随机切出一批训练样本。
- 每个样本都返回一对 `(x, y)`：`x` 是上下文输入，`y` 是 `x` 整体右移一位后的 next-token 标签。

## 2. 输入输出形状
- 输入：
  - `dataset`: 形状 `(N,)` 的 1D 整数数组（token ids）
  - `batch_size = B`
  - `context_length = T`
- 输出：
  - `x`: 形状 `(B, T)`
  - `y`: 形状 `(B, T)`
  - 且逐元素满足 `y[i, j] = dataset[s_i + j + 1]`，`x[i, j] = dataset[s_i + j]`

## 3. 数学公式 / 核心逻辑
- 关键公式：
  - 先采样起点 `s_i ~ Uniform({0, 1, ..., N - T - 1})`
  - `x_i = dataset[s_i : s_i + T]`
  - `y_i = dataset[s_i + 1 : s_i + 1 + T]`
- 核心步骤：
  1. 计算合法起点个数 `N - T`
  2. 均匀随机采样 `B` 个起点（可重复）
  3. 按每个起点切片得到 `x_i` 与 `y_i`
  4. 堆叠成 `(B, T)`，转 `torch.long` 并放到指定 `device`

为什么这样设计：
- 随机起点：提高样本多样性，减少训练只记住固定位置模式。
- `y` 右移一位：语言模型目标是“给前文预测下一个 token”。
- 批量返回 `(B, T)`：便于并行计算，训练效率高。

## 4. 作业对应函数
- `tests/adapters.py` 中的 `run_get_batch`

## 5. 最小伪代码
```python
num_start = len(dataset) - context_length
starts = uniform_random_ints(low=0, high=num_start, size=batch_size)

x = []
y = []
for s in starts:
    x.append(dataset[s : s + context_length])
    y.append(dataset[s + 1 : s + 1 + context_length])

x = torch.tensor(x, dtype=torch.long, device=device)
y = torch.tensor(y, dtype=torch.long, device=device)
return x, y
```

## 6. 对应测试命令
```bash
uv run pytest tests/test_data.py::test_get_batch -q
```

## 7. 常见坑
- 起点上界写错：应是 `< len(dataset) - context_length`，否则 `y` 会越界。
- 忘记右移一位：把 `y` 写成和 `x` 一样，语义就错了。
- 忘记 `dtype=torch.long`：token id 不是浮点数。
- 忘记使用 `device`：会导致测试里非法设备检查失效。

## 8. 数值例子（完整链路）
已知：
- `dataset = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- `B = 2, T = 4`
- 采样到起点 `s = [1, 4]`

步骤1：构造输入 `x`
- `x[0] = dataset[1:5] = [11, 12, 13, 14]`
- `x[1] = dataset[4:8] = [14, 15, 16, 17]`

步骤2：构造标签 `y`（右移一位）
- `y[0] = dataset[2:6] = [12, 13, 14, 15]`
- `y[1] = dataset[5:9] = [15, 16, 17, 18]`

步骤3：堆叠
- `x.shape = y.shape = (2, 4)`
- 且对应位置满足“`y` 比 `x` 晚一个 token”。
