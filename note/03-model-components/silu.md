# SiLU（run_silu）学习笔记

## 1. 这个模块做什么
- `SiLU` 是一个逐元素激活函数（也叫 Swish）。
- 在本作业里，它用于 Transformer 的 FFN（更具体是 SwiGLU 的门控部分）。

## 2. 输入输出形状
- 输入：任意形状张量（例如模型里常见的 `[batch_size, seq_len, d_model]`）。
- 输出：与输入完全同形状的张量。

补充：`[batch_size, seq_len, d_model]` 的含义
- `batch_size`：一批有多少条样本。
- `seq_len`：每条样本有多少个 token。
- `d_model`：每个 token 的特征维度（向量长度）。

## 3. 数学公式 / 核心逻辑
- 关键公式：$SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))$
- 核心步骤：
1. 对输入 `x` 计算 `sigmoid(x)`。
2. 与原输入逐元素相乘，得到输出。

## 4. 作业对应函数
- `tests/adapters.py` 中的 `run_silu(in_features)`。

## 5. 最小伪代码
```python
import torch

def run_silu(in_features):
    return in_features * torch.sigmoid(in_features)
    # 或等价写法：torch.nn.functional.silu(in_features)
```

## 6. 对应测试命令
```bash
uv run pytest tests/test_model.py::test_silu_matches_pytorch -q
```

## 7. 常见坑
- 忘记“逐元素”操作，误写成矩阵乘法。
- 返回类型或设备不一致（应直接沿用输入 tensor 的 dtype/device）。
- 把 SiLU 写成 ReLU/GELU，导致数值对不上测试。

## 8. 为什么这么做（重点）
- 为什么 FFN 里要有“激活”？
  - 如果没有激活，`Linear -> Linear -> Linear` 仍然等价于一个线性层，深度就失去意义。
  - 激活函数提供非线性，模型才能表达复杂模式。

- 为什么是 `Linear(d_model -> d_ff) -> 激活 -> Linear(d_ff -> d_model)`？
  - 先升维到 `d_ff`：给模型更大特征空间。
  - 中间非线性激活：让模型有非线性表达能力。
  - 再降回 `d_model`：便于和残差分支相加，保持 block 进出维度一致。

- 为什么 SiLU 常用于现代 Transformer（尤其 SwiGLU）？
  - 比 ReLU 更平滑，负区间也保留信息（不是一刀切为 0）。
  - 与门控结构（SwiGLU）搭配时，通常在稳定性和效果上更好。

## 9. 在 Transformer 里的具体位置
- 不在 Attention 子层里。
- 在每层 Transformer Block 的 FFN 子层里，典型 pre-norm 流程：
1. `x = x + Attention(RMSNorm(x))`
2. `x = x + FFN(RMSNorm(x))`
- 其中 FFN 若用 SwiGLU，常见形式是：
  - `u = xW1`
  - `v = xW3`
  - `g = SiLU(u) * v`
  - `out = gW2`
