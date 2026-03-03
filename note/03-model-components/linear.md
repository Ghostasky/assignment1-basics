# Linear（run_linear）学习笔记

## 1. 定义
- `Linear`（线性层）在这里是不带 `bias` 的线性变换。
- 白话：把输入向量和每个“输出通道”的权重做点积，得到新向量。

公式：

$y = xW^T$

- `x` 形状：`[..., d_in]`
- `W` 形状：`[d_out, d_in]`
- `y` 形状：`[..., d_out]`

## 2. 在 Transformer 哪一层
- 几乎所有子模块都会用到线性层：
- 注意力里的 `q_proj/k_proj/v_proj/o_proj`
- FFN/SwiGLU 里的 `w1/w2/w3`

## 3. 输入输出形状（本作业场景）
- `in_features`：`[batch, seq, d_model]`
- `weights`：`[d_ff, d_model]`
- 输出：`[batch, seq, d_ff]`

## 4. 为什么这样设计
- 为什么 `weights` 存成 `[d_out, d_in]`：
- 每一行代表一个输出维度的参数，语义清晰，和 PyTorch `nn.Linear.weight` 一致。
- 为什么要 `weights.T`：
- `in_features` 最后一维是 `d_in`，要和 `weights` 的 `d_in` 对齐后做矩阵乘。
- 为什么 FFN 先 `d_model -> d_ff`：
- 先升维再做非线性，能提高表达能力；再降回 `d_model` 方便残差相加。

## 5. 朴素数组视角（完整数值链路）
设：
- `d_in=3, d_out=2`
- 输入 `x=[1,2,3]`
- 权重
  - 第 1 行：`[1,0,-1]`
  - 第 2 行：`[2,1,0]`

步骤 1：算第 1 个输出维度
- `y0 = 1*1 + 2*0 + 3*(-1) = -2`

步骤 2：算第 2 个输出维度
- `y1 = 1*2 + 2*1 + 3*0 = 4`

步骤 3：拼接输出
- `y=[-2,4]`

完整链路：`输入 [1,2,3] -> 中间点积结果 [-2,4] -> 最终输出 [-2,4]`

## 6. PyTorch 实现要点
- 推荐写法：
- `out = torch.matmul(in_features, weights.T)`
- 等价写法：
- `out = in_features @ weights.T`

## 7. 常见坑
- 忘记转置 `weights`，导致维度对不上。
- 把顺序写成 `weights @ in_features`。
- 误以为这里要加 `bias`（本函数参数没有 `bias`）。
- 把 `d_in`、`d_out` 含义搞反。

## 8. 最小测试命令
```bash
uv run pytest tests/test_model.py::test_linear -q
uv run pytest tests/test_model.py -q
```
