from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算一个批量输入经过线性变换后的输出。

    Args:
        in_dim (int): 输入维度大小
        out_dim (int): 输出维度大小
        weights (Float[Tensor, "d_out d_in"]): 要使用的线性层权重
        in_features (Float[Tensor, "... d_in"]): 要应用该函数的输入张量

    Returns:
        Float[Tensor, "... d_out"]: 你的线性模块变换后的输出。
    """
    out = in_features @ weights.T
    
    # 或者：out = torch.matmul(in_features, weights.transpose(-1, -2))
    return out

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层权重，返回一批 token id 对应的嵌入向量。

    Args:
        vocab_size (int): 词表中的嵌入数量
        d_model (int): 嵌入维度大小
        weights (Float[Tensor, "vocab_size d_model"]): 用于查表的嵌入矩阵
        token_ids (Int[Tensor, "..."]): 需要从 Embedding 层中取出的 token id 集合

    Returns:
        Float[Tensor, "... d_model"]: 由你的 Embedding 层返回的一批嵌入向量。
    """
    out = weights[token_ids]
    return out
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定一个 SwiGLU 网络的权重，返回你的实现使用这些权重时的输出。

    Args:
        d_model (int): 前馈层输入与输出的维度。
        d_ff (int): SwiGLU 内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重
        w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重
        w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重
        in_features (Float[Tensor, "... d_model"]): 前馈层输入的嵌入。

    Returns:
        Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入。
    """
    # 示例：
    # 如果你的 state dict 键名一致，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    a = in_features @ w1_weight.T
    b = in_features @ w3_weight.T
    h = run_silu(a) * b
    out = h @ w2_weight.T
    return out
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定 key (K)、query (Q) 和 value (V) 张量，返回你的
    缩放点积注意力实现输出。

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query 张量
        K (Float[Tensor, " ... keys d_k"]): Key 张量
        V (Float[Tensor, " ... values d_v"]): Value 张量
        mask (Bool[Tensor, " ... queries keys"] | None): 掩码张量
    Returns:
        Float[Tensor, " ... queries d_v"]: SDPA 的输出
    """
    d_k = Q.shape[-1] 
    scores = Q @ K.transpose(-1, -2)
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)

    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定一个朴素、非批处理多头注意力实现中的 Q/K/V 投影权重，
    返回优化后的批处理实现输出。该实现应在一次矩阵乘法中完成
    所有头的 key、query 和 value 投影。
    此函数不应使用 RoPE。
    参见 Vaswani et al., 2017 第 3.2.2 节。

    Args:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中头的数量。
        max_seq_len (int): 若你的实现会预缓存，表示可预缓存的最大序列长度。
        q_proj_weight (Float[Tensor, "d_k d_in"]): Q 投影权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): K 投影权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): V 投影权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 要运行该实现的输入张量。

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: 使用给定 QKV 投影权重和输入特征运行
        你的优化批处理多头注意力实现后的输出张量。
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定一个朴素、非批处理多头注意力实现中的 Q/K/V 投影权重，
    返回优化后的批处理实现输出。该实现应在一次矩阵乘法中完成
    所有头的 key、query 和 value 投影。
    该版本 MHA 需要包含 RoPE。
    此时 RoPE 的嵌入维度必须等于单头嵌入维度（d_model // num_heads）。
    参见 Vaswani et al., 2017 第 3.2.2 节。

    Args:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中头的数量。
        max_seq_len (int): 若你的实现会预缓存，表示可预缓存的最大序列长度。
        theta (float): RoPE 参数。
        q_proj_weight (Float[Tensor, "d_k d_in"]): Q 投影权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): K 投影权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): V 投影权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 要运行该实现的输入张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选的 token 位置张量

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: 使用给定 QKV 投影权重和输入特征运行
        你的优化批处理多头注意力实现后的输出张量。
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定输入张量执行 RoPE。

    Args:
        d_k (int): query 或 key 张量的嵌入维度大小。
        theta (float): RoPE 参数。
        max_seq_len (int): 若你的实现会预缓存，表示可预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 要执行 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的 token 位置张量
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: 应用 RoPE 后的张量。
    """
  # RoPE 需要两两分组，所以 d_k 必须是偶数
    if d_k % 2 != 0:
        raise ValueError(f"d_k must be even for RoPE, got {d_k}")

    x = in_query_or_key
    device = x.device
    dtype = x.dtype
    half = d_k // 2

    # 频率: [theta^0, theta^(-2/d_k), theta^(-4/d_k), ...]
    i = torch.arange(half, device=device, dtype=dtype)
    inv_freq = theta ** (-2 * i / d_k)  # (half,)

    # 角度: (..., seq_len, half)
    angles = token_positions.to(dtype=dtype).unsqueeze(-1) * inv_freq

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 两两拆分
    x_even = x[..., 0::2]  # (..., seq_len, half)
    x_odd = x[..., 1::2]   # (..., seq_len, half)

    # 旋转
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # 交错拼回原维度
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定一个 pre-norm Transformer block 的权重和输入特征，
    返回运行该 Transformer block 的输出。

    此函数应使用 RoPE。
    视你的实现而定，你可能只需将相关参数传给 TransformerBlock 构造函数，
    或者需要初始化你自己的 RoPE 类并传入。

    Args:
        d_model (int): Transformer block 输入的维度。
        num_heads (int): 多头注意力中的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈层内部维度。
        max_seq_len (int): 若你的实现会预缓存，表示可预缓存的最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的 state dict。
            字典键包括：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的 query 投影。
                形状为 (d_model, d_model)。
                这些行按形状为 (num_heads, d_k) 的矩阵顺序排列，
                即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的 key 投影。
                形状为 (d_model, d_model)。
                这些行按形状为 (num_heads, d_k) 的矩阵顺序排列，
                即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的 value 投影。
                形状为 (d_model, d_model)。
                这些行按形状为 (num_heads, d_v) 的矩阵顺序排列，
                即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影权重。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer block 中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                FFN 中第一层线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w2.weight`
                FFN 中第二层线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w3.weight`
                FFN 中第三层线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ln2.weight`
                Transformer block 中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            要运行该实现的输入张量。

    Returns:
        Float[Tensor, "batch sequence_length d_model"]: 在使用 RoPE 的情况下，
        对输入特征运行 Transformer block 后的输出张量。
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入索引，
    返回对输入索引执行一次前向传播后的输出。

    此函数应使用 RoPE。

    Args:
        vocab_size (int): 要预测的输出词表中唯一项的数量。
        context_length (int): 一次最多处理的 token 数。
        d_model (int): 模型嵌入和子层输出的维度。
        num_layers (int): Transformer 层数。
        num_heads (int): 多头注意力中的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈层内部维度（第 3.3 节）。
        rope_theta (float): RoPE 的 $\\Theta$ 参数。
        weights (dict[str, Tensor]):
            参考实现的 state dict。{num_layers} 表示
            `0` 到 `num_layers - 1` 之间的层索引。
            字典键包括：
            - `token_embeddings.weight`
                token 嵌入矩阵。形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 个注意力头的 query 投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                这些行按形状为 (num_heads, d_k) 的矩阵顺序排列，
                即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 个注意力头的 key 投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                这些行按形状为 (num_heads, d_k) 的矩阵顺序排列，
                即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 个注意力头的 value 投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                这些行按形状为 (num_heads, d_v) 的矩阵顺序排列，
                即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影权重。
                形状为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer block 中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN 中第一层线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN 中第二层线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN 中第三层线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ln2.weight`
                Transformer block 中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ln_final.weight`
                应用于最后一个 Transformer block 输出的 RMSNorm 仿射变换权重。
                形状为 (d_model,)。
            - `lm_head.weight`
                语言模型输出嵌入权重。
                形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]): 运行语言模型的输入索引张量。
            形状为 (batch_size, sequence_length)，其中 `sequence_length` 至多为 `context_length`。

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 每个 token 的预测未归一化 next-word 分布张量。
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，
    返回对输入特征执行 RMSNorm 的输出。

    Args:
        d_model (int): RMSNorm 输入维度。
        eps: (float): 为数值稳定性加入分母的值。
        weights (Float[Tensor, "d_model"]): RMSNorm 权重。
        in_features (Float[Tensor, "... d_model"]): 要执行 RMSNorm 的输入特征。
            可以有任意前导维度。

    Returns:
        Float[Tensor,"... d_model"]: 与 `in_features` 形状相同的张量，
        表示 `in_features` 经 RMSNorm 后的输出。
    """
    mean_sq = (in_features ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    normalized = in_features / rms
    out = normalized * weights
    return out
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入张量，返回对每个元素应用 SiLU 后的输出。

    Args:
        in_features(Float[Tensor, "..."]): 要执行 SiLU 的输入特征，形状任意。

    Returns:
        Float[Tensor,"..."]: 与 `in_features` 形状相同的张量，
        表示逐元素应用 SiLU 后的输出。
    """
    return in_features * torch.sigmoid(in_features)
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（一个 1D 整数 numpy 数组）、批大小和上下文长度，
    从数据集中采样语言模型输入序列及其对应标签。

    Args:
        dataset (np.array): 数据集中 token id 的 1D 整数 numpy 数组。
        batch_size (int): 期望采样的 batch 大小。
        context_length (int): 每个样本的期望上下文长度。
        device (str): PyTorch 设备字符串（如 'cpu' 或 'cuda:0'），
            指定采样输入序列和标签应放置的设备。

    Returns:
        由两个形状为 (batch_size, context_length) 的 torch.LongTensor 组成的元组。
        第一个是采样得到的输入序列，第二个是对应的语言模型标签。
    """
    num_starting_indices = len(dataset) - context_length
    assert num_starting_indices > 0, "dataset length must be greater than context_length"

    data = torch.as_tensor(dataset, dtype=torch.long, device=device)
    # 均匀采样每个样本的起点（有放回采样）。
    starts = torch.randint(low=0, high=num_starting_indices, size=(batch_size,), device=device)
    offsets = torch.arange(context_length, device=device)

    x = data[starts.unsqueeze(1) + offsets.unsqueeze(0)]
    y = data[starts.unsqueeze(1) + offsets.unsqueeze(0) + 1]
    return x, y


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回在指定 `dim` 上应用 softmax 后的输出。

    Args:
        in_features (Float[Tensor, "..."]): 要做 softmax 的输入特征，形状任意。
        dim (int): 对 `in_features` 执行 softmax 的维度。

    Returns:
        Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，
        表示在指定 `dim` 上归一化后的 softmax 输出。
    """
    max_value = in_features.max(dim=dim, keepdim=True).values
    stable = in_features - max_value
    exp_vals = torch.exp(stable)
    denom = exp_vals.sum(dim=dim, keepdim=True)
    return exp_vals / denom


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入张量和目标标签，计算样本平均交叉熵损失。

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): `inputs[i][j]`
            表示第 i 个样本对第 j 类的未归一化 logit。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，
            每个值是正确类别索引，且必须在 0 到 `num_classes - 1` 之间。

    Returns:
        Float[Tensor, ""]: 所有样本上的平均交叉熵损失。
    """
    # 先在类别维做稳定 log-softmax，得到每个类别的对数概率。
    log_probs = torch.log_softmax(inputs, dim=1)
    # 从每个样本中取出真实类别对应的 log-prob，形状 [B]。
    gold_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    # 负对数似然并在 batch 维求平均，返回标量 loss。
    return (-gold_log_probs).mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，将其合并后的梯度裁剪到不超过 max_l2_norm 的 l2 范数。

    Args:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数集合。
        max_l2_norm (float): 最大 l2 范数（正数）。

    参数梯度（parameter.grad）应被原地修改。
    """
    grads: list[Tensor] = []
    for p in parameters:
        grad = p.grad
        if grad is not None:
            grads.append(grad)

    if not grads:
        return

    # 计算所有梯度拼接后的全局 L2 范数。
    total_norm_sq = torch.zeros((), device=grads[0].device)
    for grad in grads:
        total_norm_sq = total_norm_sq + grad.pow(2).sum()
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm <= max_l2_norm:
        return

    # 与 torch.nn.utils.clip_grad_norm_ 保持一致，分母加 1e-6 防止除零。
    scale = max_l2_norm / (total_norm + 1e-6)
    for grad in grads:
        grad.mul_(scale)


def get_adamw_cls() -> Any:
    """
    返回一个实现 AdamW 的 torch.optim.Optimizer 类。
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定带线性 warmup 的余弦学习率衰减调度参数和迭代步数，
    返回该迭代对应的学习率。

    Args:
        it (int): 要查询学习率的迭代步。
        max_learning_rate (float): alpha_max，调度中的最大学习率。
        min_learning_rate (float): alpha_min，调度中的最小/最终学习率。
        warmup_iters (int): T_w，线性 warmup 的迭代步数。
        cosine_cycle_iters (int): T_c，余弦退火阶段的迭代步数。

    Returns:
        指定调度下给定迭代步的学习率。
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代步数，将它们序列化保存到磁盘。

    Args:
        model (torch.nn.Module): 需要序列化状态的模型。
        optimizer (torch.optim.Optimizer): 需要序列化状态的优化器。
        iteration (int): 需要序列化的值，表示已完成的训练迭代数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 序列化输出目标，
            可以是路径或类文件对象。
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定一个已序列化的 checkpoint（路径或类文件对象），
    将序列化状态恢复到给定模型和优化器。
    返回 checkpoint 中先前保存的迭代步数。

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 已序列化 checkpoint 的路径或类文件对象。
        model (torch.nn.Module): 要恢复状态的模型。
        optimizer (torch.optim.Optimizer): 要恢复状态的优化器。
    Returns:
        int: 之前序列化保存的迭代步数。
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词表、merge 列表和 special token 列表，
    返回使用这些配置的 BPE tokenizer。

    Args:
        vocab (dict[int, bytes]): tokenizer 词表，映射关系为
            int（词表中的 token id）到 bytes（token 的字节表示）。
        merges (list[tuple[bytes, bytes]]): BPE merges。
            每一项是 bytes 二元组（<token1>, <token2>），
            表示 <token1> 与 <token2> 进行了合并。
            merges 按创建顺序排列。
        special_tokens (list[str] | None): tokenizer 的特殊 token 字符串列表。
            这些字符串不会被拆成多个 token，并且始终作为单个 token 保留。

    Returns:
        使用给定 vocab、merges 和 special tokens 的 BPE tokenizer。
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料路径，训练一个 BPE tokenizer，
    并输出其词表和 merges。

    Args:
        input_path (str | os.PathLike): BPE tokenizer 训练数据路径。
        vocab_size (int): tokenizer 词表总大小（包含 special tokens）。
        special_tokens (list[str]): 要加入 tokenizer 词表的特殊 token 字符串列表。
            这些字符串不会被拆成多个 token，并始终作为单个 token 保留。
            如果这些 special tokens 出现在 `input_path` 中，它们会被当作普通字符串处理。

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练得到的 tokenizer 词表，映射关系为
                int（词表中的 token id）到 bytes（token 的字节表示）。
            merges:
                BPE merges。每一项是 bytes 二元组（<token1>, <token2>），
                表示 <token1> 与 <token2> 进行了合并。
                merges 按创建顺序排列。
    """
    raise NotImplementedError
