# Exact upstream function definitions; only decorators replaced by @triton.jit.
# /*
# * Copyright 2018-2020 Philippe Tillet
# * Copyright 2020-2022 OpenAI
# *
# * Permission is hereby granted, free of charge, to any person obtaining
# * a copy of this software and associated documentation files
# * (the "Software"), to deal in the Software without restriction,
# * including without limitation the rights to use, copy, modify, merge,
# * publish, distribute, sublicense, and/or sell copies of the Software,
# * and to permit persons to whom the Software is furnished to do so,
# * subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be
# * included in all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# */
# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
import triton
import triton.language as tl

@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                k = gk
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles
