# CS336 A1 Basics：我的学习型实现项目

我把这个仓库当成一份系统学习 `Transformer` 基础组件的实践记录。  
我的目标不只是把测试跑绿，而是把每个模块的原理、输入输出形状、设计理由和常见坑都搞清楚，并沉淀成可复习的笔记。

作业原始说明在这里：
- [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## 我在这个项目里要完成什么

- 我会通过实现 `tests/adapters.py` 中的函数，补齐 A1 的基础组件链路。
- 我会按“先理解、再实现、再测试”的节奏推进。
- 我会把关键结论写进 `note/`，方便后续复盘。

## 我如何组织这个仓库

- `tests/adapters.py`  
  我把它当作测试适配层：这里负责把测试接到我的实现。

- `tests/`  
  这里是官方测试集。我会按模块逐个击破，而不是一开始就全量硬跑。

- `cs336_basics/`  
  这里放我的实现代码。我会按模块拆分文件，避免把逻辑都堆在一个地方。

- `note/`  
  这里放我的学习笔记，重点记录“为什么这样实现”和“我踩过哪些坑”。

## 我的笔记目录索引（`note/`）

- `note/00-roadmap/`：学习路线与验收清单
- `note/01-foundations/`：数学与张量基础
- `note/02-data/`：数据采样（`get_batch`）
- `note/03-model-components/`：模型组件（Embedding、RMSNorm、RoPE、MHA、SwiGLU 等）
- `note/04-model-assembly/`：Transformer Block / LM 组装
- `note/05-optimization/`：AdamW、学习率调度、梯度裁剪
- `note/06-tokenizer/`：Tokenizer 与 BPE 训练
- `note/07-serialization/`：Checkpoint 保存与加载
- `note/08-debug/`：报错与排查记录
- `note/09-torch-syntax/`：PyTorch 语法和类型标注补充

我通常从这两份开始看：
- [note/00-roadmap/学习与实现路线.md](./note/00-roadmap/学习与实现路线.md)
- [note/00-roadmap/测试与验收清单.md](./note/00-roadmap/测试与验收清单.md)

## 我常用的运行方式

我使用 `uv` 管理环境和执行命令：

```bash
uv run <python_file_path>
uv run pytest -q
```

项目初期出现大量 `NotImplementedError` 是正常现象，我会按模块逐步把失败清掉。

## 我的实现顺序（固定节奏）

1. 先完成基础数学函数：`silu / softmax / cross_entropy / gradient_clipping`
2. 再完成数据采样：`get_batch`
3. 再完成模型底层：`linear / embedding / rmsnorm / sdpa / rope / swiglu`
4. 再组装模型：`mha / block / transformer_lm`
5. 再完成训练工程：`adamw / lr schedule / checkpoint`
6. 最后做 tokenizer 与 `train_bpe`

我的原则是：每完成一个小模块，就跑对应最小测试，尽量不把问题堆到最后。

## 我当前进度

### 1) 基础数学组件

- [x] `silu`
- [x] `softmax`
- [x] `cross_entropy`
- [x] `gradient_clipping`
- 完成定义：`uv run pytest tests/test_nn_utils.py -q`

### 2) 数据组件

- [x] `get_batch`
- 完成定义：`uv run pytest tests/test_data.py -q`

### 3) 模型底层组件

- [x] `linear`
- [x] `embedding`
- [ ] `rmsnorm`
- [ ] `scaled_dot_product_attention`
- [ ] `rope`
- [ ] `swiglu`
- 完成定义：`uv run pytest tests/test_model.py -q`

### 4) 模型组装

- [ ] `multihead_self_attention`
- [ ] `multihead_self_attention_with_rope`
- [ ] `transformer_block`
- [ ] `transformer_lm`
- 完成定义：`uv run pytest tests/test_model.py -q`

### 5) 训练工程组件

- [ ] `adamw`
- [ ] `lr_cosine_schedule`
- [ ] `save_checkpoint / load_checkpoint`
- 完成定义：
  - `uv run pytest tests/test_optimizer.py -q`
  - `uv run pytest tests/test_serialization.py -q`

### 6) Tokenizer 组件

- [ ] `get_tokenizer`
- [ ] `train_bpe`
- 完成定义：
  - `uv run pytest tests/test_tokenizer.py -q`
  - `uv run pytest tests/test_train_bpe.py -q`

### 7) 总体验收

- [ ] 全量测试通过（目标：`46 passed, 2 skipped`）
- 完成定义：`uv run pytest -q`

## 数据下载（按需）

如果我要做训练相关实验，会先下载 TinyStories 和 OWT 子集：

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
