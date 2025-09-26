// Pair-list forward path (64+64 -> 128) — initial implementation scaffold.
// We keep the logic in this TU to avoid touching headers for fast rebuilds.

#include <math_constants.h>

#include <ATen/cuda/CUDAContext.h>

#include "flash.h"
#include "flash_fwd_launch_template.h"  // for Flash_fwd_kernel_traits

// NOTE
// We keep all pair-list logic in this TU to avoid touching headers (fast rebuilds).
// Step 1: Add a minimal kernel that parses CSR tile-pair metadata per CTA (b, h, m_block).
// Step 2 (follow-up): Replace the fallback with real 2x64 gathers + 128 compute inside this TU.

template<typename Kernel_traits>
__global__ void flash_fwd_block_pairs_kernel_simplified(Flash_fwd_params params) {
    // This kernel currently only parses CSR row ranges, serving as a scaffold.
    // It does not write outputs; we still fallback to the legacy path for correctness.
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;

    // Guard: ensure we have pair-list metadata
    if (!params.use_pair_list || params.tile_pairs_row_ptr == nullptr || params.tile_pairs == nullptr) {
        return;
    }

    // Only consider blocksparse heads (mask_type>0). Dense/streaming will be handled by fallback path.
    const int mask_type = params.head_mask_type ? params.head_mask_type[bidh] : 0;
    if (mask_type <= 0) {
        return;
    }

    // Effective numbers
    constexpr int kBlockM = Kernel_traits::kBlockM; // rows of Q per CTA
    const int Q_blocks = (params.seqlen_q_rounded + params.m_block_dim - 1) / params.m_block_dim;
    const int head_sparse_id = mask_type - 1; // 0-based within blocksparse heads
    const int row_block_idx = (m_block * kBlockM) / params.m_block_dim; // scale if kBlockM!=m_block_dim
    const int flat_row = bidb * params.num_blocksparse_heads + head_sparse_id; // (b, hs)
    const int rp_base = flat_row * (Q_blocks + 1);
    // Row_ptr layout is [B, H_sparse, Q_blocks+1]
    const int* __restrict__ row_ptr = params.tile_pairs_row_ptr;
    const int4* __restrict__ pairs = params.tile_pairs;

    // Bounds check (defensive)
    if (flat_row < 0) return;
    const int start = row_ptr[rp_base + row_block_idx + 0];
    const int end   = row_ptr[rp_base + row_block_idx + 1];
    if (end <= start) {
        return;
    }

    // Iterate pairs to validate indexing (no compute yet)
    // We intentionally keep this empty for now to avoid side-effects.
    for (int p = start; p < end; ++p) {
        int4 e = pairs[p];
        (void)e; // suppress unused warning
    }
}

// ----------------------------------------------------------------------------
// Step 2: head_dim=64 pair kernel with real 64+64 -> 128 compute. The kernel
// iterates CSR ranges, gathers two 64-wide segments into shared memory, runs a
// per-row streaming softmax, and accumulates P@V. Dropout/alibi remain
// unsupported here; other head dims keep falling back on the legacy block mask path.
// ----------------------------------------------------------------------------

template<typename Kernel_traits>
__global__ void flash_fwd_block_pairs_kernel_hdim64(Flash_fwd_params params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const int tidx = threadIdx.x;

    if (!params.use_pair_list || params.tile_pairs_row_ptr == nullptr || params.tile_pairs == nullptr) {
        return;
    }
    const int mask_type = params.head_mask_type ? params.head_mask_type[bidh] : 0;
    if (mask_type <= 0) return;  // dense / streaming not handled here

    const int Q_blocks = (params.seqlen_q_rounded + params.m_block_dim - 1) / params.m_block_dim;
    const int head_sparse_id = mask_type - 1;
    const int row_block_idx = (m_block * kBlockM) / params.m_block_dim;
    const int flat_row = bidb * params.num_blocksparse_heads + head_sparse_id;
    const int rp_base = flat_row * (Q_blocks + 1);

    const int* __restrict__ row_ptr = params.tile_pairs_row_ptr;
    const int4* __restrict__ pairs = params.tile_pairs;
    if (flat_row < 0) return;
    const int start = row_ptr[rp_base + row_block_idx + 0];
    const int end   = row_ptr[rp_base + row_block_idx + 1];

    // Block info for variable-length handling.
    const flash::BlockInfo</*Varlen=*/true> binfo(params, bidb);
    const int q_block_row_start = m_block * kBlockM;
    if (q_block_row_start >= binfo.actual_seqlen_q) {
        return;
    }
    const int rows_in_block = min(kBlockM, binfo.actual_seqlen_q - q_block_row_start);

    const int head_k = bidh / params.h_h_k_ratio;
    auto *q_ptr = reinterpret_cast<Element *>(params.q_ptr);
    auto *k_ptr = reinterpret_cast<Element *>(params.k_ptr);
    auto *v_ptr = reinterpret_cast<Element *>(params.v_ptr);
    auto *o_ptr = reinterpret_cast<Element *>(params.o_ptr);
    auto *lse_ptr = reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr);

    const index_t q_batch_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    const index_t k_batch_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    const index_t v_batch_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    const index_t o_batch_offset = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);

    Element *q_block_ptr = q_ptr + q_batch_offset + q_block_row_start * params.q_row_stride + bidh * params.q_head_stride;
    Element *k_head_ptr = k_ptr + k_batch_offset + head_k * params.k_head_stride;
    Element *v_head_ptr = v_ptr + v_batch_offset + head_k * params.v_head_stride;
    Element *o_block_ptr = o_ptr + o_batch_offset + q_block_row_start * params.o_row_stride + bidh * params.o_head_stride;
    ElementAccum *lse_block_ptr = lse_ptr + (bidb * params.h + bidh) * params.seqlen_q + q_block_row_start;

    // If this row-block has no pairs, write zeros to O and INF to LSE (match legacy early exit).
    if (end <= start) {
        if (tidx < rows_in_block) {
            Element *o_row = o_block_ptr + tidx * params.o_row_stride;
            ElementAccum *lse_row = lse_block_ptr + tidx;
#pragma unroll
            for (int d = 0; d < kHeadDim; ++d) { o_row[d] = Element(0.f); }
            *lse_row = INFINITY;
        }
        return;
    }

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Element *smem_K = reinterpret_cast<Element *>(smem_raw);
    Element *smem_V = smem_K + kBlockN * kHeadDim;

    __shared__ int sh_col_indices[kBlockN];
    __shared__ int sh_col_count;

    // Per-row buffers (each thread handles one row when rows_in_block <= blockDim)
    float q_vec[kHeadDim];
    float acc_vec[kHeadDim];
    float row_max = -CUDART_INF_F;
    float row_sum = 0.f;

    if (tidx < rows_in_block) {
        Element *q_row_ptr = q_block_ptr + tidx * params.q_row_stride;
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            q_vec[d] = static_cast<float>(q_row_ptr[d]);
            acc_vec[d] = 0.f;
        }
        row_max = -CUDART_INF_F;
        row_sum = 0.f;
    }
    __syncthreads();

    const bool is_causal = params.is_causal;
    const float scale = params.scale_softmax;

    auto compute_token_base = [](int parent, int off)->int {
        const int parent_delta = (off >= 128) ? 1 : 0;
        const int rel = off % 128;  // 0 or 64
        return (parent + parent_delta) * 128 + rel;
    };

    // Single-pass online softmax across all pairs (row-level): # FIXME LATER!!! 
    for (int p = start; p < end; ++p) {
        const int4 meta = pairs[p];
        const int col_parent = meta.x;
        const int off0 = meta.y;
        const int off1 = meta.z;
        if (col_parent < 0) continue;

        if (tidx == 0) {
            int count = 0;
            auto append_segment = [&](int base_token) {
                for (int s = 0; s < 64 && count < kBlockN; ++s) {
                    int token = base_token + s;
                    if (token >= binfo.actual_seqlen_k) { break; }
                    sh_col_indices[count++] = token;
                }
            };
            append_segment(compute_token_base(col_parent, off0));
            append_segment(compute_token_base(col_parent, off1));
            sh_col_count = count;
        }
        __syncthreads();

        const int col_count = sh_col_count;
        if (col_count == 0) { __syncthreads(); continue; }

        // Load K for this pair to compute chunk_max
        for (int idx = tidx; idx < col_count * kHeadDim; idx += blockDim.x) {
            const int col = idx / kHeadDim;
            const int dim = idx % kHeadDim;
            const int token = sh_col_indices[col];
            const Element *k_vec = k_head_ptr + static_cast<index_t>(token) * params.k_row_stride;
            smem_K[col * kHeadDim + dim] = k_vec[dim];
        }
        __syncthreads();

        // Compute chunk_max for this pair
        float chunk_max = -CUDART_INF_F;
        if (tidx < rows_in_block) {
            const int q_token = q_block_row_start + tidx;
            for (int col = 0; col < col_count; ++col) {
                const int token = sh_col_indices[col];
                if (token >= binfo.actual_seqlen_k) { continue; }
                if (is_causal && token > q_token) { continue; }
                const Element *k_col = smem_K + col * kHeadDim;
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < kHeadDim; ++d) { dot += q_vec[d] * static_cast<float>(k_col[d]); }
                dot *= scale;
                chunk_max = fmaxf(chunk_max, dot);
            }
        }
        __syncthreads();

        // Rescale previous acc and sum to the new max
        if (tidx < rows_in_block) {
            const float new_row_max = (row_sum == 0.f && row_max == -CUDART_INF_F) ? chunk_max : fmaxf(row_max, chunk_max);
            if (row_sum > 0.f && row_max != -CUDART_INF_F) {
                const float prev_scale = expf(row_max - new_row_max);
#pragma unroll
                for (int d = 0; d < kHeadDim; ++d) { acc_vec[d] *= prev_scale; }
                row_sum *= prev_scale;
            }
            row_max = new_row_max;
        }
        __syncthreads();

        // Load V (and reuse K in smem) to accumulate contributions with new_row_max
        for (int idx = tidx; idx < col_count * kHeadDim; idx += blockDim.x) {
            const int col = idx / kHeadDim;
            const int dim = idx % kHeadDim;
            const int token = sh_col_indices[col];
            const Element *v_vec = v_head_ptr + static_cast<index_t>(token) * params.v_row_stride;
            smem_V[col * kHeadDim + dim] = v_vec[dim];
        }
        __syncthreads();

        if (tidx < rows_in_block) {
            const int q_token = q_block_row_start + tidx;
            for (int col = 0; col < col_count; ++col) {
                const int token = sh_col_indices[col];
                if (token >= binfo.actual_seqlen_k) { continue; }
                if (is_causal && token > q_token) { continue; }
                const Element *k_col = smem_K + col * kHeadDim;
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < kHeadDim; ++d) { dot += q_vec[d] * static_cast<float>(k_col[d]); }
                dot *= scale;
                const float w = expf(dot - row_max);
                row_sum += w;
                const Element *v_col = smem_V + col * kHeadDim;
#pragma unroll
                for (int d = 0; d < kHeadDim; ++d) { acc_vec[d] += w * static_cast<float>(v_col[d]); }
            }
        }
        __syncthreads();
    }

    if (tidx < rows_in_block) {
        Element *o_row = o_block_ptr + tidx * params.o_row_stride;
        ElementAccum *lse_row = lse_block_ptr + tidx;
        if (row_sum > 0.f && row_max != -CUDART_INF_F) {
            const float inv_row_sum = 1.f / row_sum;
#pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                const float val = acc_vec[d] * inv_row_sum;
                o_row[d] = static_cast<Element>(val);
            }
            *lse_row = row_max + logf(row_sum);
        } else {
#pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                o_row[d] = Element(0.f);
            }
            *lse_row = INFINITY;
        }
    }
}
template<typename T, int Headdim>
void run_mha_fwd_block_pairs(Flash_fwd_params &params, cudaStream_t stream) {
    using Traits = Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>;
    constexpr int kBlockM = Traits::kBlockM;
    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    const bool have_pairs = params.use_pair_list && params.tile_pairs_row_ptr && params.tile_pairs;
    bool launched_pairs_kernel = false;
    if (have_pairs) {
        const bool no_dropout = (params.p_dropout == 1.f);     // keep-prob == 1 -> no dropout
        const bool no_alibi = (params.alibi_slopes_ptr == nullptr);
        if (Headdim == 64 && no_dropout && no_alibi) {
            // Only need K/V for two 64-wide segments: 2 * 128cols * Headdim elements.
            const size_t smem_bytes = size_t(Traits::kBlockN) * Traits::kHeadDim * sizeof(typename Traits::Element) * 2;
            // No need to bump dyn smem limit since we request only what we use here.
            // Debug: print launch config once for small cases
            if (params.b <= 4 && params.h <= 8 && grid.x <= 16) {
                printf("[pairs64] launch grid=(%d,%d,%d) threads=%d smem=%zu bytes\n", grid.x, grid.y, grid.z, Traits::kNThreads, smem_bytes);
            }
            flash_fwd_block_pairs_kernel_hdim64<Traits><<<grid, Traits::kNThreads, smem_bytes, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            launched_pairs_kernel = true;
        } else {
            // Simplified parser uses no dynamic smem.
            if (params.b <= 4 && params.h <= 8 && grid.x <= 16) {
                printf("[pairs64] simplified grid=(%d,%d,%d) threads=%d smem=%d\n", grid.x, grid.y, grid.z, Traits::kNThreads, 0);
            }
            flash_fwd_block_pairs_kernel_simplified<Traits><<<grid, Traits::kNThreads, 0, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    if (!launched_pairs_kernel) {
        run_mha_fwd_block_<T, Headdim>(params, stream);
    }
}

// Explicit instantiations for the head dims used by FWD_BLOCK_HEADDIM_SWITCH.
#include <cutlass/numeric_types.h>

template void run_mha_fwd_block_pairs<cutlass::half_t, 32>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_block_pairs<cutlass::half_t, 64>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_block_pairs<cutlass::half_t, 128>(Flash_fwd_params &, cudaStream_t);

template void run_mha_fwd_block_pairs<cutlass::bfloat16_t, 32>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_block_pairs<cutlass::bfloat16_t, 64>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_block_pairs<cutlass::bfloat16_t, 128>(Flash_fwd_params &, cudaStream_t);
