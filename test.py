#!/usr/bin/env python3
import torch
from block_sparse_attn.block_sparse_attn_interface import (
    block_sparse_attn_func,
    block_sparse_attn_func_pairs,
    build_tile_pairs_from_mask64,
)

def tile_pairs_to_mask(row_ptr: torch.Tensor,
                       data: torch.Tensor,
                       *,
                       batch_size: int,
                       num_heads_sparse: int,
                       q_blocks: int,
                       k_blocks: int) -> torch.Tensor:
    """
    根据 CSR 格式的 tile pair（每个条目 = [col_parent, child_off0, child_off1]）
    还原成 128×128 的布尔掩码，方便与旧接口对照验证。
    """
    mask = torch.zeros(
        batch_size,
        num_heads_sparse,
        q_blocks,
        k_blocks,
        dtype=torch.bool,
        device=row_ptr.device,
    )

    row_ptr_cpu = row_ptr.detach().to("cpu")
    data_cpu = data.detach().to("cpu")
    row_ptr_view = row_ptr_cpu.view(batch_size * num_heads_sparse, q_blocks + 1)
    data_view = data_cpu.view(batch_size * num_heads_sparse, data_cpu.shape[-2], 3)

    for flat_idx in range(batch_size * num_heads_sparse):
        b = flat_idx // num_heads_sparse
        h = flat_idx % num_heads_sparse
        row_ptr_slice = row_ptr_view[flat_idx]
        data_slice = data_view[flat_idx]
        for q_blk in range(q_blocks):
            start = int(row_ptr_slice[q_blk].item())
            end = int(row_ptr_slice[q_blk + 1].item())
            for p in range(start, end):
                col_parent = int(data_slice[p, 0].item())
                if col_parent == -1:
                    continue
                mask[b, h, q_blk, col_parent] = True
    return mask

def pairs_reference_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    row_ptr: torch.Tensor,
    pair_data: torch.Tensor,
    *,
    is_causal: bool,
    softmax_scale: float | None = None,
):
    """
    CPU reference for pair semantics: For each row (token), aggregate all tokens
    from the two 64-wide segments of every pair in its Q-block row, then compute
    softmax over that set and P@V.

    Shapes:
      q: [Lq, H, D], k/v: [Lk, H, D]
      row_ptr: [B, Hs, Q_blocks+1]
      pair_data: [B, Hs, pair_cap, 3] (col_parent, off0, off1)
    """
    device = q.device
    Lq, H, D = q.shape
    Lk = k.shape[0]
    bs = 128
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)
    B = cu_q.numel() - 1
    Hs = row_ptr.shape[1]
    Qb = row_ptr.shape[2] - 1
    pair_cap = pair_data.shape[-2]

    out = torch.zeros_like(q, dtype=q.dtype, device=device)
    lse = torch.full((B, H, Lq // B), float("inf"), device=device, dtype=torch.float32)

    # Flatten views for easier scalar loops on CPU
    row_ptr_cpu = row_ptr.detach().to("cpu").view(B * Hs, Qb + 1)
    pair_cpu = pair_data.detach().to("cpu").view(B * Hs, pair_cap, 3)
    # Use float32 on CPU for stable reference numerics
    q_cpu = q.detach().to("cpu", dtype=torch.float32)
    k_cpu = k.detach().to("cpu", dtype=torch.float32)
    v_cpu = v.detach().to("cpu", dtype=torch.float32)

    def seg_base(parent: int, off: int) -> int:
        parent_delta = 1 if off >= 128 else 0
        rel = off % 128  # 0 or 64
        return (parent + parent_delta) * 128 + rel

    for b in range(B):
        q_start = int(cu_q[b].item())
        q_end = int(cu_q[b + 1].item())
        k_start = int(cu_k[b].item())
        k_end = int(cu_k[b + 1].item())
        Lq_b = q_end - q_start
        Lk_b = k_end - k_start
        for h in range(H):
            hs = h  # assume all heads are blocksparse
            flat_idx = b * Hs + hs
            for t in range(Lq_b):
                q_token = q_start + t
                q_blk = (t // bs)
                start = int(row_ptr_cpu[flat_idx, q_blk].item())
                end = int(row_ptr_cpu[flat_idx, q_blk + 1].item())
                if end <= start:
                    continue
                tokens = []
                for p in range(start, end):
                    col_parent = int(pair_cpu[flat_idx, p, 0].item())
                    off0 = int(pair_cpu[flat_idx, p, 1].item())
                    off1 = int(pair_cpu[flat_idx, p, 2].item())
                    if col_parent < 0:
                        continue
                    base0 = seg_base(col_parent, off0)
                    base1 = seg_base(col_parent, off1)
                    for s in range(64):
                        tok = base0 + s
                        if tok < k_end:
                            tokens.append(tok)
                    for s in range(64):
                        tok = base1 + s
                        if tok < k_end:
                            tokens.append(tok)
                if not tokens:
                    continue
                # Remove causal-violating tokens
                if is_causal:
                    tokens = [tt for tt in tokens if tt <= q_token]
                    if not tokens:
                        continue
                # Compute softmax over selected tokens
                qv = q_cpu[q_token, h]
                scores = []
                for tok in tokens:
                    kv = k_cpu[tok, h]
                    scores.append(float((qv * kv).sum().item()))
                import math
                scores = [s * softmax_scale for s in scores]
                m = max(scores)
                exps = [math.exp(s - m) for s in scores]
                ssum = sum(exps)
                # Accumulate P@V
                acc = torch.zeros(D, dtype=torch.float32)
                for w, tok in zip(exps, tokens):
                    acc += (w / ssum) * v_cpu[tok, h].to(torch.float32)
                out[q_token, h] = acc.to(out.dtype)
    return out

def run_case(
    *,
    B: int,
    H: int,
    Q_blocks: int,
    K_blocks: int,
    head_dim: int,
    dtype=torch.float16,
    device="cuda",
):
    """
    构造一组小规模的张量、配对 CSR，分别调用 pairs API 和旧 API，对比误差。
    """
    block_size = 128
    seq_q = Q_blocks * block_size
    seq_k = K_blocks * block_size

    # Toy 输入（可改成 deterministic random seed 以复现）
    q = torch.randn(seq_q, H, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_k, H, head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_k, H, head_dim, device=device, dtype=dtype)

    # cu_seqlens 按 batch 展开
    cu_q = torch.arange(0, (B + 1) * seq_q // B, seq_q // B, device=device, dtype=torch.int32)
    cu_k = torch.arange(0, (B + 1) * seq_k // B, seq_k // B, device=device, dtype=torch.int32)

    # 所有 head 均使用 Block-Sparse 路径
    # Use distinct positive ids per head so that CSR heads map uniquely
    head_mask_type = torch.arange(1, H + 1, device=device, dtype=torch.int32)

    # 构造一个简单的 pair CSR：每行 1~2 个 pair，child 偏移信息暂未真正下沉到 CUDA
    # row_ptr 形状: [B, H_sparse, Q_blocks + 1]
    row_ptr = torch.tensor(
        [[[0, 1, 2] for _ in range(H)] for _ in range(B)],  # 每个 q_block 1 个 pair
        device=device,
        dtype=torch.int32,
    )

    # data 形状: [B, H_sparse, pair_capacity, 3]
    # “col_parent, child_off0, child_off1”，这里 child_off* 保留 64 子块偏移（0 / 64 / 128 / 192）
    pair_capacity = Q_blocks  # 每个 q_block 最多 1 个 pair
    pair_data = torch.full(
        (B, H, pair_capacity, 3),
        fill_value=-1,
        device=device,
        dtype=torch.int32,
    )

    # 手工指定一些 pair：故意让两个 head 的模式不同，覆盖 1100 / 0011 & 1010 等路径
    # head 0: q_block0 -> col0，q_block1 -> col2
    pair_data[0, 0, 0] = torch.tensor([0, 0, 64], device=device, dtype=torch.int32)   # pattern ~1100
    pair_data[0, 0, 1] = torch.tensor([2, 0, 64], device=device, dtype=torch.int32)   # pattern ~0011

    # head 1: q_block0 -> col1，q_block1 -> col1（同一列，重复算两次）
    pair_data[0, 1, 0] = torch.tensor([1, 0, 64], device=device, dtype=torch.int32)
    pair_data[0, 1, 1] = torch.tensor([1, 64, 192], device=device, dtype=torch.int32)  # child 偏移不同

    # 通过 CSR 恢复旧的 128×128 掩码，用来喂原始 kernel 做对照
    mask = tile_pairs_to_mask(
        row_ptr,
        pair_data,
        batch_size=B,
        num_heads_sparse=H,  # 此处 sparse head = 全部 head
        q_blocks=Q_blocks,
        k_blocks=K_blocks,
    )

    # 旧接口（128 blockmask）结果
    out_mask = block_sparse_attn_func(
        q, k, v,
        cu_q, cu_k,
        head_mask_type,
        None,                # streaming_info
        mask,
        max_seqlen_q_=seq_q,
        max_seqlen_k_=seq_k,
        p_dropout=0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )

    # 新接口（pair-list）结果 —— 触发 64 专用内核
    out_pairs = block_sparse_attn_func_pairs(
        q, k, v,
        cu_q, cu_k,
        head_mask_type,
        None,
        row_ptr,
        pair_data,
        max_seqlen_q_=seq_q,
        max_seqlen_k_=seq_k,
        p_dropout=0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )

    max_abs_diff = (out_pairs - out_mask).abs().max().item()
    print(f"Output diff (abs max): {max_abs_diff:.3e}")
    print(f"pairs shape: {tuple(out_pairs.shape)}, mask shape: {tuple(out_mask.shape)}")

    # Reference check vs pairs semantics
    out_ref = pairs_reference_out(q, k, v, cu_q, cu_k, row_ptr, pair_data, is_causal=False)
    diff_ref = (out_pairs - out_ref).abs().max().item()
    print(f"[pairs vs ref] Output diff (abs max): {diff_ref:.3e}")

    # 逐 head 打印误差，便于定位多头映射问题
    with torch.no_grad():
        diffs_per_head = (out_pairs - out_ref).abs().amax(dim=0)  # [H, D] -> reduce along tokens
        diffs_per_head = diffs_per_head.amax(dim=-1).tolist()
        print("[pairs vs ref] per-head diffs:", [f"{d:.3e}" for d in diffs_per_head])

if __name__ == "__main__":
    torch.manual_seed(0)
    # 基础对比（手写 pair 列表，覆盖不同列父块）
    run_case(B=1, H=2, Q_blocks=2, K_blocks=3, head_dim=64, dtype=torch.float16, device="cuda")

    # 64 粒度 mask -> pair-list 构造与对比
    B, H, Qb, Kb = 1, 1, 2, 4  # 注意 Kb 需为偶数（按单元格=2个128）
    bs = 128
    Lq, Lk, D = Qb*bs, Kb*bs, 64
    device = "cuda"

    q = torch.randn(Lq, H, D, device=device, dtype=torch.float16)
    k = torch.randn(Lk, H, D, device=device, dtype=torch.float16)
    v = torch.randn(Lk, H, D, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, Lq], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, Lk], device=device, dtype=torch.int32)
    head_mask_type = torch.ones(H, device=device, dtype=torch.int32)

    # 构造 64 粒度列掩码: [B, H, Qb, Kb*2]
    mask64 = torch.zeros(B, H, Qb, Kb*2, device=device, dtype=torch.bool)
    # 在 (q=0) 的第一单元格(两块128,共4个64)设置 0110（跨父块配对）
    # cell0 覆盖 k_parent=0,1 -> 4个64索引0..3
    mask64[0,0,0, 1] = True  # idx=1 -> off=64
    mask64[0,0,0, 2] = True  # idx=2 -> off=128
    # 在 (q=1) 的第二单元格设置 1010（跨父块配对）
    # cell1 覆盖 k_parent=2,3 -> in last dim indices [4,5,6,7]
    mask64[0,0,1, 4] = True  # idx=0 -> off=0 (but relative to cell base col_parent=2)
    mask64[0,0,1, 6] = True  # idx=2 -> off=128

    row_ptr, pair_data = build_tile_pairs_from_mask64(mask64)

    # 用于旧核对照的 128 掩码（注意：只在列父块维度上标 True）
    mask = tile_pairs_to_mask(row_ptr, pair_data, batch_size=B, num_heads_sparse=H, q_blocks=Qb, k_blocks=Kb)

    out_mask = block_sparse_attn_func(
        q, k, v,
        cu_q, cu_k,
        head_mask_type,
        None,
        mask,
        max_seqlen_q_=Lq, max_seqlen_k_=Lk,
        p_dropout=0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )

    out_pairs = block_sparse_attn_func_pairs(
        q, k, v,
        cu_q, cu_k,
        head_mask_type,
        None,
        row_ptr, pair_data,
        max_seqlen_q_=Lq, max_seqlen_k_=Lk,
        p_dropout=0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    diff = (out_pairs - out_mask).abs().max().item()
    print(f"[mask64->pairs] Output diff (abs max): {diff:.3e}")
    out_ref = pairs_reference_out(q, k, v, cu_q, cu_k, row_ptr, pair_data, is_causal=False)
    diff_ref = (out_pairs - out_ref).abs().max().item()
    print(f"[mask64->pairs] Pairs vs ref (abs max): {diff_ref:.3e}")

    # 追加更多 pattern 覆盖：0000（无对）、1111（两对）
    B, H, Qb, Kb = 1, 1, 2, 4
    Lq, Lk, D = Qb*bs, Kb*bs, 64
    q = torch.randn(Lq, H, D, device=device, dtype=torch.float16)
    k = torch.randn(Lk, H, D, device=device, dtype=torch.float16)
    v = torch.randn(Lk, H, D, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, Lq], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, Lk], device=device, dtype=torch.int32)
    head_mask_type = torch.ones(H, device=device, dtype=torch.int32)

    # 0000 + 1111：q=0 -> 0000（无 pair），q=1 -> 1111（两对）
    mask64 = torch.zeros(B, H, Qb, Kb*2, device=device, dtype=torch.bool)
    # q=1 第二行，两个父块（2,3）的4个64全部置1
    mask64[0,0,1, 4:8] = True

    row_ptr, pair_data = build_tile_pairs_from_mask64(mask64)
    mask = tile_pairs_to_mask(row_ptr, pair_data, batch_size=B, num_heads_sparse=H, q_blocks=Qb, k_blocks=Kb)

    out_mask = block_sparse_attn_func(
        q, k, v,
        cu_q, cu_k,
        head_mask_type,
        None,
        mask,
        max_seqlen_q_=Lq, max_seqlen_k_=Lk,
        p_dropout=0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )

    out_pairs = block_sparse_attn_func_pairs(
        q, k, v,
        cu_q, cu_k,
        head_mask_type,
        None,
        row_ptr, pair_data,
        max_seqlen_q_=Lq, max_seqlen_k_=Lk,
        p_dropout=0.0,
        deterministic=True,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    diff2 = (out_pairs - out_mask).abs().max().item()
    print(f"[0000/1111] Output diff (abs max): {diff2:.3e}")
    out_ref = pairs_reference_out(q, k, v, cu_q, cu_k, row_ptr, pair_data, is_causal=False)
    diff2_ref = (out_pairs - out_ref).abs().max().item()
    print(f"[0000/1111] Pairs vs ref (abs max): {diff2_ref:.3e}")

    # 调试：统计第三组（0000/1111）中 q=1 行每个 pair 的 token 数，期望每对约 128（末尾可被 Lk 截断）
    if True:
        flat_idx = 0
        q_blk = 1
        row_ptr_cpu = row_ptr.detach().to("cpu").view(1, 1, -1)
        pair_cpu = pair_data.detach().to("cpu").view(1, 1, -1, 3)
        start = int(row_ptr_cpu[0, 0, q_blk].item())
        end = int(row_ptr_cpu[0, 0, q_blk + 1].item())
        def seg_base(parent: int, off: int) -> int:
            parent_delta = 1 if off >= 128 else 0
            rel = off % 128
            return (parent + parent_delta) * 128 + rel
        tok_counts = []
        for p in range(start, end):
            col_parent = int(pair_cpu[0, 0, p, 0].item())
            off0 = int(pair_cpu[0, 0, p, 1].item())
            off1 = int(pair_cpu[0, 0, p, 2].item())
            tokens = list(range(seg_base(col_parent, off0), seg_base(col_parent, off0) + 64)) 
            tokens += list(range(seg_base(col_parent, off1), seg_base(col_parent, off1) + 64))
            tok_counts.append(len(tokens))
        print(f"[0000/1111] q_blk=1 pair token counts: {tok_counts}")
