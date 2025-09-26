#!/usr/bin/env python3
"""
Visualize dense vs. 128-block vs. 64-pair coverage for a single head.

Produces a side-by-side figure saved to assets/viz_video_mask.png:
  - Dense attention mask (token x token)
  - 128-block mask expanded to token grid
  - 64-pair coverage expanded to token grid (two 64 segments per pair)
"""
import os
import math
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from block_sparse_attn.block_sparse_attn_interface import build_tile_pairs_from_mask64


def build_video_like_mask(
    Lq: int,
    Lk: int,
    *,
    band: int = 256,
    long_stride: int = 256,
    long_width: int = 64,
    causal: bool = False,
    forward_stripes: bool = False,
) -> torch.Tensor:
    """Create a simple video-like attention mask at token level.
    - A diagonal local band of width `band`.
    - Additional long-range stripes every `long_stride` with width `long_width` behind current token.
    """
    M = torch.zeros(Lq, Lk, dtype=torch.bool)
    for i in range(Lq):
        j0 = max(0, i - band // 2)
        j1 = min(Lk, i + band // 2)
        if causal:
            j1 = min(j1, i + 1)
        if j0 < j1:
            M[i, j0:j1] = True
        # Long-range backward stripes
        j = i - long_stride
        while j >= 0:
            j2 = min(Lk, j + long_width)
            M[i, j:j2] = True
            j -= long_stride
        # Optional long-range forward stripes (see future frames)
        if forward_stripes and not causal:
            j = i + long_stride
            while j < Lk:
                j2 = min(Lk, j + long_width)
                M[i, j:j2] = True
                j += long_stride
    return M


def block128_from_dense(M: torch.Tensor, block: int = 128) -> torch.Tensor:
    Lq, Lk = M.shape
    Qb = (Lq + block - 1) // block
    Kb = (Lk + block - 1) // block
    B = torch.zeros(Qb, Kb, dtype=torch.bool)
    for qb in range(Qb):
        qs = qb * block
        qe = min(Lq, (qb + 1) * block)
        for kb in range(Kb):
            ks = kb * block
            ke = min(Lk, (kb + 1) * block)
            if (M[qs:qe, ks:ke]).any():
                B[qb, kb] = True
    return B


def mask64_from_dense(M: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Return 64-granularity col mask per 128 parent block: [Qb, Kb*2]."""
    Lq, Lk = M.shape
    Qb = (Lq + block - 1) // block
    Kb = (Lk + block - 1) // block
    out = torch.zeros(Qb, Kb * 2, dtype=torch.bool)
    for qb in range(Qb):
        qs = qb * block
        qe = min(Lq, (qb + 1) * block)
        for kb in range(Kb):
            ks = kb * block
            mid = min(Lk, ks + 64)
            ke = min(Lk, ks + 128)
            # segment 0: [ks, ks+64)
            if qs < qe and ks < mid:
                out[qb, kb * 2 + 0] = bool(M[qs:qe, ks:mid].any())
            # segment 1: [ks+64, ks+128)
            if qs < qe and mid < ke:
                out[qb, kb * 2 + 1] = bool(M[qs:qe, mid:ke].any())
    return out


def expand_block128_to_tokens(B: torch.Tensor, Lq: int, Lk: int, block: int = 128) -> torch.Tensor:
    Qb, Kb = B.shape
    out = torch.zeros(Lq, Lk, dtype=torch.bool)
    for qb in range(Qb):
        qs = qb * block
        qe = min(Lq, (qb + 1) * block)
        for kb in range(Kb):
            if not B[qb, kb].item():
                continue
            ks = kb * block
            ke = min(Lk, (kb + 1) * block)
            out[qs:qe, ks:ke] = True
    return out


def expand_pairs_to_tokens(row_ptr: torch.Tensor, pair_data: torch.Tensor, Lq: int, Lk: int, block: int = 128) -> torch.Tensor:
    """Convert CSR pairs [Qb+1], [pair_cap, 3] back to token coverage.
    Each pair marks two 64-wide segments.
    """
    Qb = row_ptr.shape[0] - 1
    out = torch.zeros(Lq, Lk, dtype=torch.bool)
    for qb in range(Qb):
        qs = qb * block
        qe = min(Lq, (qb + 1) * block)
        start = int(row_ptr[qb].item())
        end = int(row_ptr[qb + 1].item())
        for p in range(start, end):
            col_parent = int(pair_data[p, 0].item())
            if col_parent < 0:
                continue
            off0 = int(pair_data[p, 1].item())
            off1 = int(pair_data[p, 2].item())
            def seg_base(parent, off):
                parent_delta = 1 if off >= 128 else 0
                rel = off % 128  # 0 or 64
                return (parent + parent_delta) * 128 + rel
            for base in (seg_base(col_parent, off0), seg_base(col_parent, off1)):
                ks = base
                ke = min(Lk, base + 64)
                if qs < qe and ks < ke:
                    out[qs:qe, ks:ke] = True
    return out


def overlay_random_pairs(mask64: torch.Tensor, *, cell_prob: float = 0.3, pattern: str = "0101", seed: int | None = 0) -> torch.Tensor:
    """Optionally overlay a random 4x64 pattern per 2-parent cell onto mask64.
    mask64: [Qb, Kb*2] where each 128 parent contributes 2 entries (0,64).
    We consider cells as (parent c, c+1), affecting indices [2c,2c+1, 2(c+1),2(c+1)+1].
    pattern: one of {"0101","1010","0011","1100","1111"} to make the difference visible.
    """
    if cell_prob <= 0:
        return mask64
    if seed is not None:
        torch.manual_seed(seed)
    Qb, K2 = mask64.shape
    Kb = K2 // 2
    out = mask64.clone()
    # encode desired 4 bits per cell
    pat_maps = {
        "0101": [0,1,0,1],
        "1010": [1,0,1,0],
        "0011": [0,0,1,1],
        "1100": [1,1,0,0],
        "1111": [1,1,1,1],
    }
    pat = torch.tensor(pat_maps.get(pattern, pat_maps["0101"]), dtype=torch.bool)
    for qb in range(Qb):
        for c in range(0, Kb - 1, 2):  # each cell spans parents c and c+1
            if torch.rand(1).item() < cell_prob:
                idxs = [2*c + 0, 2*c + 1, 2*(c+1) + 0, 2*(c+1) + 1]
                out[qb, idxs] |= pat
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1024, help="sequence length (Q=K=L)")
    parser.add_argument("--band", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--longw", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--rand64", action="store_true", help="overlay random 64-segment patterns per 2-parent cell")
    parser.add_argument("--rand64-prob", type=float, default=0.3, dest="rand64_prob")
    parser.add_argument("--rand64-pattern", type=str, default="0101", choices=["0101","1010","0011","1100","1111"])
    parser.add_argument("--forward-stripes", action="store_true", help="add forward (future) long-range stripes as well")
    args = parser.parse_args()

    L = args.L
    block = 128

    # 1) Dense token-level mask
    dense = build_video_like_mask(
        L, L,
        band=args.band,
        long_stride=args.stride,
        long_width=args.longw,
        causal=args.causal,
        forward_stripes=args.forward_stripes,
    )

    # 2) 128-block mask
    B128 = block128_from_dense(dense, block=block)
    dense_from_block = expand_block128_to_tokens(B128, L, L, block=block)

    # 3) 64-pair CSR from 64-granularity col mask, then expand
    mask64 = mask64_from_dense(dense, block=block)  # [Qb, Kb*2]
    if args.rand64:
        mask64 = overlay_random_pairs(mask64, cell_prob=args.rand64_prob, pattern=args.rand64_pattern, seed=0)
    # Build pairs via helper (expects [B,H,Qb,Kb*2])
    mask64_batched = mask64[None, None, ...]  # [1,1,Qb,Kb*2]
    row_ptr, pair_data = build_tile_pairs_from_mask64(mask64_batched)
    # Drop batch/head dims for visualization
    row_ptr = row_ptr[0, 0]
    pair_data = pair_data[0, 0]
    dense_from_pairs = expand_pairs_to_tokens(row_ptr, pair_data, L, L, block=block)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(dense.cpu().numpy(), aspect='auto', interpolation='nearest', cmap='Greys')
    axes[0].set_title('Dense')
    axes[1].imshow(dense_from_block.cpu().numpy(), aspect='auto', interpolation='nearest', cmap='Greys')
    axes[1].set_title('128-Block (expanded)')
    axes[2].imshow(dense_from_pairs.cpu().numpy(), aspect='auto', interpolation='nearest', cmap='Greys')
    axes[2].set_title('64-Pairs (expanded)')
    for ax in axes:
        ax.set_xlabel('K (tokens)')
        ax.set_ylabel('Q (tokens)')

    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'viz_video_mask.png')
    plt.savefig(out_path, dpi=120)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
