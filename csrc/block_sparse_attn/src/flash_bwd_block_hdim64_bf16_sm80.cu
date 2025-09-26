// Copyright (c) 2023, Tri Dao.
// Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_block_<cutlass::bfloat16_t, 64>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    run_mha_bwd_block_hdim64<cutlass::bfloat16_t>(params, stream, configure);
}
