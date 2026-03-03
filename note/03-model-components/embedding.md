# Embedding（run_embedding）学习笔记

## 1. 定义
- `Embedding` 是一个可学习的查表层：输入 token id，输出对应向量。
- 白话：token id 只是“编号”，Embedding 把编号翻译成模型能计算的语义向量。

## 2. 在 Transformer 哪一层
- 在最前面的输入层，用来把离散 token 序列转换成连续向量序列。
- 转换后张量才会送入后续的注意力层和前馈层。

## 3. 输入输出形状（本作业场景）
- `weights`：`[vocab_size, d_model]`
- `token_ids`：`[batch, seq]`（更一般是 `[...]`）
- 输出：`[batch, seq, d_model]`（更一般是 `[..., d_model]`）

## 4. 为什么这样设计
- 为什么要 Embedding：token id 是离散编号，没有可直接比较的语义距离；向量空间能表达“相似词更接近”的关系。
- 为什么是查表：它和 one-hot 乘矩阵数学等价，但查表更省内存和计算。
- 为什么输出最后一维是 `d_model`：后续 Transformer 子层都约定最后一维是模型维度，便于残差连接和模块拼接。

## 5. 朴素数组视角（完整数值链路）
设：
- `vocab_size=5, d_model=3`
- `weights` 为：
  - id=0 -> `[0.1, 0.2, 0.3]`
  - id=1 -> `[1.0, 1.1, 1.2]`
  - id=2 -> `[2.0, 2.1, 2.2]`
  - id=3 -> `[3.0, 3.1, 3.2]`
  - id=4 -> `[4.0, 4.1, 4.2]`
- `token_ids=[3, 1, 3, 0]`

步骤 1：读第 1 个 id `3`，取 `weights[3]`，得到 `[3.0, 3.1, 3.2]`  
步骤 2：读第 2 个 id `1`，取 `weights[1]`，得到 `[1.0, 1.1, 1.2]`  
步骤 3：读第 3 个 id `3`，取 `weights[3]`，得到 `[3.0, 3.1, 3.2]`  
步骤 4：读第 4 个 id `0`，取 `weights[0]`，得到 `[0.1, 0.2, 0.3]`  
步骤 5：拼接成最终输出：
`[[3.0, 3.1, 3.2], [1.0, 1.1, 1.2], [3.0, 3.1, 3.2], [0.1, 0.2, 0.3]]`

完整链路：`输入 token_ids -> 中间逐位置查表 -> 最终 embedding 向量序列`

## 6. 作业对应函数
- [`tests/adapters.py:39`](../../tests/adapters.py) 的 `run_embedding`

## 7. 最小伪代码
```python
def run_embedding(vocab_size, d_model, weights, token_ids):
    # 可选：形状检查
    # assert weights.shape == (vocab_size, d_model)

    # 高级索引查表
    out = weights[token_ids]
    return out
```

## 8. PyTorch 写法对照
- 推荐：`out = weights[token_ids]`
- 等价：`out = torch.nn.functional.embedding(token_ids, weights)`

## 9. 常见坑
- 把 Embedding 写成矩阵乘法并错误转置，导致维度不匹配。
- `token_ids` 不是整数类型（应为 `LongTensor`/整数索引）。
- id 越界（必须满足 `0 <= token_id < vocab_size`）。
- 误加 `softmax` 或非线性；Embedding 这里只做查表。

## 10. 最小测试命令
```bash
uv run pytest tests/test_model.py::test_embedding -q
uv run pytest tests/test_model.py -q
uv run pytest -q
```
