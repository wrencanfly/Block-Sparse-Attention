# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_blocksparse_attn_interface.py

import block_sparse_attn_cuda
import torch
import torch.nn as nn

__all__ = [
    'block_sparse_attn_func',
    'block_sparse_attn_func_pairs',
]

# Python-side replacement for C++-style TORCH_CHECK used below
def TORCH_CHECK(cond, msg: str):
    try:
        import torch as _torch
        if isinstance(cond, _torch.Tensor):
            cond = cond.item() if cond.numel() == 1 else bool(cond)
    except Exception:
        pass
    if not bool(cond):
        raise RuntimeError(msg)



def convert_blockmask(blockmask, causal):
    """Convert from the 0-1 format to the format used by the CUDA code.
    0 means the block is skipped.
    nonzero means the block is not skipped.
    Argument:
        blockmask: (row, col): a 0-1 tensor
    Return:
        blockmask_converted: (col, row), dtype torch.int32: for each column, it contains the row
            indices of the nonzero blocks, padded with -1 to reach length @row.
            The indices are multiplied by 4, with the smallest bit used to encode whether
            it is the first nonzero in its row, and the 2nd smallest bit to encode whether it is
            the last nonzero in its row..
    """
    assert not causal
    nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=0, stable=True, descending=True)
    nonzero_unsorted_rowidx = nonzero_sorted_rowidx.argsort(dim=0)
    last_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True).indices[:, -1]
    last_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), last_nonzero_col_per_row
    ]
    first_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True, descending=True).indices[:, 0]
    first_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), first_nonzero_col_per_row
    ]
    nonzero_idx = nonzero_sorted_rowidx * 4
    nonzero_idx[last_nonzero_col_per_row_after_sort, last_nonzero_col_per_row] += 2
    nonzero_idx[first_nonzero_col_per_row_after_sort, first_nonzero_col_per_row] += 1
    nonzero_idx[nonzero_val == 0] = -1
    return nonzero_idx.T.contiguous().to(dtype=torch.int32)


def convert_blockmask_row_reverse(blockmask, causal=False):
    # assert not causal
    # nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-1, stable=True, descending=False)
    
    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    # print("nonzero_idx: ", nonzero_idx)
    nonzero_idx = torch.flip(nonzero_idx, dims=[-1])
    # print("nonzero_idx: ", nonzero_idx)
    
    return nonzero_idx.contiguous().to(dtype=torch.int32)


def convert_blockmask_col_reverse(blockmask, causal=False):
    # assert not causal
    # nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=-2, stable=True, descending=False)
    
    nonzero_idx = nonzero_sorted_rowidx
    nonzero_idx[nonzero_val == 0] = -1
    nonzero_idx = torch.flip(nonzero_idx, dims=[-2])
    nonzero_idx = torch.transpose(nonzero_idx, -1, -2)
    
    return nonzero_idx.contiguous().to(dtype=torch.int32)


def replace_ones_with_count(tensor):
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num

def build_tile_pairs_from_mask64(mask64: torch.Tensor,
                                 *,
                                 reorder_mode: str = "compact"):
    """Build CSR tile-pair list from a 64-granularity K-side mask.

    Assumptions:
    - Q 仍按 128 粒度分块（行块数 = Q_blocks）。
    - K 在 64 粒度给出掩码：形状 [B, H_sparse, Q_blocks, K_blocks * 2]。
      其中每个 128 K-block 被拆成 2 个 64 子段（列侧更细）。
    - 我们按“单元格=相邻两个 128 K-block（共4个64）”在 host 侧生成 pair 列表：
      每条 pair = (col_parent, off0, off1)，off∈{0,64,128,192}（以单元格左起的 128 为基准）。

    返回：tile_pairs_row_ptr [B, H_sparse, Q_blocks+1], tile_pairs_data [B, H_sparse, pair_cap, 3]
    注：pair_cap 取各行最大 pair 数；不足位置以 (-1,-1,-1) 填充。
    """
    TORCH_CHECK(mask64.dtype in (torch.bool, torch.uint8, torch.int32), "mask64 must be bool/uint8/int32")
    if mask64.dtype != torch.bool:
        mask64 = mask64 != 0
    TORCH_CHECK(mask64.dim() == 4, "mask64 must be [B, H_sparse, Q_blocks, K_blocks*2]")
    B, Hs, Qb, K2 = mask64.shape
    TORCH_CHECK(K2 % 2 == 0, "last dim must be even (2* K_blocks)")
    Kb = K2 // 2
    TORCH_CHECK(Kb % 2 == 0, "K_blocks must be even to form 256-cells (2x128)")

    # Prepare lists on CPU for scalar-friendly loops
    m_cpu = mask64.detach().to('cpu')
    # For each (b, hs), count total pairs across all rows to size the data buffer.
    total_pairs_per_head = torch.zeros((B, Hs), dtype=torch.int64)
    for b in range(B):
        for hs in range(Hs):
            total = 0
            for q in range(Qb):
                row = m_cpu[b, hs, q]
                for c in range(0, Kb, 2):
                    bits = [
                        bool(row[c*2 + 0].item()),
                        bool(row[c*2 + 1].item()),
                        bool(row[(c+1)*2 + 0].item()),
                        bool(row[(c+1)*2 + 1].item()),
                    ]
                    cnt = sum(bits)
                    total += cnt // 2  # compact-left pairing count
            total_pairs_per_head[b, hs] = total

    pair_cap = int(total_pairs_per_head.max().item())
    pair_cap = max(1, pair_cap)
    data = torch.full((B, Hs, pair_cap, 3), -1, dtype=torch.int32, device=mask64.device)
    row_ptr = torch.zeros((B, Hs, Qb+1), dtype=torch.int32, device=mask64.device)

    # Fill row_ptr and data with a running cursor per (b,hs)
    for b in range(B):
        for hs in range(Hs):
            cursor = 0
            for q in range(Qb):
                row_ptr[b, hs, q] = cursor
                row = m_cpu[b, hs, q]
                for c in range(0, Kb, 2):
                    col_parent = c
                    bits = [
                        bool(row[c*2 + 0].item()),
                        bool(row[c*2 + 1].item()),
                        bool(row[(c+1)*2 + 0].item()),
                        bool(row[(c+1)*2 + 1].item()),
                    ]
                    idx = [i for i, v in enumerate(bits) if v]
                    def off_of(i: int) -> int:
                        return [0, 64, 128, 192][i]
                    while len(idx) >= 2:
                        if cursor < pair_cap:
                            data[b, hs, cursor, 0] = col_parent
                            data[b, hs, cursor, 1] = off_of(idx[0])
                            data[b, hs, cursor, 2] = off_of(idx[1])
                        cursor += 1
                        idx = idx[2:]
            row_ptr[b, hs, Qb] = cursor
    return row_ptr, data

def _validate_tile_pairs(tile_pairs_row_ptr: torch.Tensor, tile_pairs_data: torch.Tensor, *, device: torch.device):
    """Validate CSR-style tile pair metadata for 64-block pairing."""
    TORCH_CHECK(tile_pairs_row_ptr.dtype == torch.int32, "tile_pairs_row_ptr must be torch.int32")
    TORCH_CHECK(tile_pairs_data.dtype == torch.int32, "tile_pairs_data must be torch.int32")
    TORCH_CHECK(tile_pairs_row_ptr.is_cuda and tile_pairs_data.is_cuda, "tile pair tensors must reside on CUDA")
    TORCH_CHECK(tile_pairs_row_ptr.device == device and tile_pairs_data.device == device, "tile pair tensors must align with q/k/v device")
    TORCH_CHECK(tile_pairs_row_ptr.dim() >= 1, "tile_pairs_row_ptr must have >=1 dimension")
    TORCH_CHECK(tile_pairs_data.dim() >= 2, "tile_pairs_data must have >=2 dimensions")
    TORCH_CHECK(tile_pairs_data.shape[-1] in (3, 4), "tile_pairs_data last dim must be 3 or 4 (col_parent, child_off0, child_off1[, reserved])")
    TORCH_CHECK(tile_pairs_row_ptr.shape[:-1] == tile_pairs_data.shape[:-2], "tile pair metadata batch dims must align")
    # CSR monotonicity check
    diffs = tile_pairs_row_ptr[..., 1:] - tile_pairs_row_ptr[..., :-1]
    TORCH_CHECK((diffs >= 0).all().item(), "tile_pairs_row_ptr must be non-decreasing")



def _block_sparse_attn_forward(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    row_blockmask,
    tile_pairs_row_ptr,
    tile_pairs_data,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    is_causal,
    exact_streaming,
    return_softmax,
    window_size_left,
    window_size_right
):
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = block_sparse_attn_cuda.fwd_block(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        row_blockmask,
        tile_pairs_row_ptr,
        tile_pairs_data,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_streaming,
        return_softmax,
        window_size_left,
        window_size_right, 
        None
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _block_sparse_attn_backward(
    dout,
    q, k, v,
    out,
    softmax_lse,
    dq, dk, dv,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    col_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    deterministic,
    rng_state=None,
):
    dq, dk, dv, softmax_d = block_sparse_attn_cuda.bwd_block(
        dout,
        q, k, v,
        out,
        softmax_lse,
        dq, dk, dv,
        cu_seqlens_q, cu_seqlens_k,
        m_block_dim, n_block_dim,
        head_mask_type,
        streaming_info,
        col_blockmask,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
        None, rng_state
    )
    return dq, dk, dv, softmax_d


class BlockSparseAttnFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                m_block_dim, n_block_dim,
                head_mask_type,
                streaming_info,
                base_blockmask,
                tile_pairs_row_ptr,
                tile_pairs_data,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_softmax,
                window_size_left,
                window_size_right, deterministic=False):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None
        
        if exact_streaming:
            assert streaming_info is not None
            assert is_causal
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _block_sparse_attn_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            row_blockmask,
            tile_pairs_row_ptr,
            tile_pairs_data,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_softmax=False,
            window_size_left=window_size_left,
            window_size_right=window_size_right
        )
        ctx.save_for_backward(q, k, v,
                              out, S_dmask, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k,
                              head_mask_type,
                              streaming_info,
                              base_blockmask,
                              rng_state)
        # ctx.is_blocksparse = is_blocksparse
        ctx.m_block_dim = m_block_dim
        ctx.n_block_dim = n_block_dim
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.max_seqlen_q_ = max_seqlen_q_
        ctx.max_seqlen_k_ = max_seqlen_k_
        ctx.p_dropout = p_dropout
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.exact_streaming = exact_streaming
        ctx.deterministic = deterministic
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, S_dmask, softmax_lse, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # S_dmask is None, temporarily use another tensor just to get it running
        if base_blockmask is not None:
            col_blockmask = convert_blockmask_col_reverse(base_blockmask, ctx.is_causal)
        else:
            col_blockmask = None
            
        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"
            
        _block_sparse_attn_backward(
            dout,
            q, k, v,
            out,
            softmax_lse,
            dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k,
            ctx.m_block_dim, ctx.n_block_dim,
            head_mask_type,
            streaming_info,
            col_blockmask,
            ctx.max_seqlen_q_, ctx.max_seqlen_k_,
            ctx.p_dropout,
            ctx.softmax_scale,
            True,  # zero_tensors
            ctx.is_causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            rng_state=rng_state
        )
        return (dq, dk, dv) + (None,) * 19


# We duplicate code to return both the output and the softmax for testing
# Returning both makes backward a bit slower, so we want to keep using the other version for speed.
class BlockSparseAttnFunWithS(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                m_block_dim, n_block_dim,
                head_mask_type,
                streaming_info,
                base_blockmask,
                tile_pairs_row_ptr,
                tile_pairs_data,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_softmax,
                window_size_left,
                window_size_right,
                deterministic=False):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None
            
        if exact_streaming:
            assert streaming_info is not None
            print("is_causal: ", is_causal)
            assert is_causal
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _block_sparse_attn_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            row_blockmask,
            tile_pairs_row_ptr,
            tile_pairs_data,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_softmax=return_softmax and p_dropout > 0,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        
        ctx.save_for_backward(q, k, v,
                              out, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k,
                              head_mask_type,
                              streaming_info,
                              base_blockmask,
                              rng_state)
        # ctx.is_blocksparse = is_blocksparse
        ctx.m_block_dim = m_block_dim
        ctx.n_block_dim = n_block_dim
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.max_seqlen_q_ = max_seqlen_q_
        ctx.max_seqlen_k_ = max_seqlen_k_
        ctx.p_dropout = p_dropout
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.exact_streaming = exact_streaming
        ctx.deterministic = deterministic
        return out, softmax_lse, S_dmask

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        
        # S_dmask is None, temporarily use another tensor just to get it running
        if base_blockmask is not None:
            col_blockmask = convert_blockmask_col_reverse(base_blockmask, ctx.is_causal)
        else:
            col_blockmask = None
        
        assert not ctx.exact_streaming, "Exact streaming not supported in backward pass"
        
        dq, dk, dv, _ = _block_sparse_attn_backward(
            dout,
            q, k, v,
            out,
            softmax_lse,
            dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k,
            ctx.m_block_dim, ctx.n_block_dim,
            head_mask_type,
            streaming_info,
            col_blockmask,
            ctx.max_seqlen_q_, ctx.max_seqlen_k_,
            ctx.p_dropout,
            ctx.softmax_scale,
            True,  # zero_tensors
            ctx.is_causal,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.deterministic,
            rng_state=rng_state
        )
        
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        return (dq, dk, dv) + (None,) * 19


def block_sparse_attn_func_pairs(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    tile_pairs_row_ptr,
    tile_pairs_data,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
):
    """Experimental path accepting 64-block pair metadata (CSR) and internally forming 128 mask."""
    TORCH_CHECK(cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32")
    TORCH_CHECK(cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32")
    _orig_head_mask_type = head_mask_type
    blocksparse_head_num = int((head_mask_type > 0).sum().item())
    _validate_tile_pairs(tile_pairs_row_ptr, tile_pairs_data, device=q.device)
    if exact_streaming:
        raise NotImplementedError("Exact streaming not supported for pair-list API yet")
    if return_attn_probs:
        raise NotImplementedError("Returning attn probs not supported for pair-list API yet")

    B = cu_seqlens_q.numel() - 1
    TORCH_CHECK(B > 0, "empty batch not supported")
    q_block_num = (max_seqlen_q_ + 127) // 128
    k_block_num = (max_seqlen_k_ + 127) // 128

    # Convert pair metadata to CPU for easier scalar iteration (host pre-processing).
    row_ptr_cpu = tile_pairs_row_ptr.detach().to('cpu')
    data_cpu = tile_pairs_data.detach().to('cpu')

    # Expect row_ptr shape [..., q_block_num + 1].
    TORCH_CHECK(row_ptr_cpu.shape[-1] == q_block_num + 1,
                "tile_pairs_row_ptr last dim must be q_block_num + 1")
    flat_rows = row_ptr_cpu.numel() // (q_block_num + 1)
    TORCH_CHECK(flat_rows == B * blocksparse_head_num,
                "tile pair row_ptr batch dims must be [B, head_sparse, q_block_num+1]")
    row_ptr_view = row_ptr_cpu.view(flat_rows, q_block_num + 1)

    # Data expected shape [B, head_sparse, pair_cap, 3] or flattened equivalent.
    TORCH_CHECK(data_cpu.dim() >= 3,
                "tile_pairs_data must have at least 3 dims (batch, head, pair, 3)")
    pair_cap = data_cpu.shape[-2]
    data_view = data_cpu.view(flat_rows, pair_cap, 3)

    # Build a global-offset CSR for CUDA: each (b,hs) head slice occupies a
    # contiguous range of length pair_cap in the flattened pairs array.
    # row_ptr_global[flat_idx, q] = row_ptr_local[flat_idx, q] + flat_idx * pair_cap
    row_ptr_view = row_ptr_cpu.view(flat_rows, q_block_num + 1)
    row_ptr_global_cpu = torch.empty_like(row_ptr_view)
    for flat_idx in range(flat_rows):
        base = flat_idx * pair_cap
        row_ptr_global_cpu[flat_idx] = row_ptr_view[flat_idx] + base
    row_ptr_global = row_ptr_global_cpu.view_as(row_ptr_cpu).to(device=q.device, dtype=torch.int32)

    base_blockmask = torch.zeros(
        (B, blocksparse_head_num, q_block_num, k_block_num),
        device=q.device,
        dtype=torch.bool,
    )

    for flat_idx in range(flat_rows):
        b = flat_idx // blocksparse_head_num
        hs = flat_idx % blocksparse_head_num
        row_ptr = row_ptr_view[flat_idx]
        data_row = data_view[flat_idx]
        prev = int(row_ptr[0].item())
        TORCH_CHECK(prev == 0, "CSR row_ptr must start with 0")
        for q_blk in range(q_block_num):
            start = int(row_ptr[q_blk].item())
            end = int(row_ptr[q_blk + 1].item())
            if end > pair_cap:
                raise RuntimeError("tile_pairs_row_ptr points beyond tile_pairs_data capacity")
            for p in range(start, end):
                col_parent = int(data_row[p, 0].item())
                TORCH_CHECK(0 <= col_parent < k_block_num, "col_parent out of range")
                base_blockmask[b, hs, q_blk, col_parent] = True

    return block_sparse_attn_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        _orig_head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_, max_seqlen_k_,
        p_dropout,
        deterministic=deterministic,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        exact_streaming=exact_streaming,
        return_attn_probs=return_attn_probs,
        tile_pairs_row_ptr=row_ptr_global,
        tile_pairs_data=tile_pairs_data,
    )




def block_sparse_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
    tile_pairs_row_ptr=None,
    tile_pairs_data=None,
):
    _orig_head_mask_type = head_mask_type
    blocksparse_head_num = int((head_mask_type > 0).sum().item())
    if base_blockmask is not None:
        assert base_blockmask.shape[1] == blocksparse_head_num
    
    """dropout_p should be set to 0.0 during evaluation"""
    # print("is_causal0: ", is_causal)
    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,
                head_mask_type,
                streaming_info,
                base_blockmask,
                tile_pairs_row_ptr,
                tile_pairs_data,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_attn_probs,
                -1, -1,
                deterministic
                )
    
    
def token_streaming_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q_, max_seqlen_k_,
    deterministic=False,
    softmax_scale=None,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation"""
    # print("is_causal0: ", is_causal)
    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,
                head_mask_type,
                streaming_info,
                None,
                None,
                None,
                max_seqlen_q_, max_seqlen_k_,
                0.0,
                softmax_scale,
                True,
                True,
                return_attn_probs,
                -1, -1,
                deterministic
                )
    
def block_streaming_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=True,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation"""
    # print("is_causal0: ", is_causal)
    func = BlockSparseAttnFun if not return_attn_probs else BlockSparseAttnFunWithS
    return func.apply(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                128, 128,
                head_mask_type,
                streaming_info,
                None,
                None,
                None,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                False,
                return_attn_probs,
                -1, -1,
                deterministic
                )
