import math
from typing import Union
import torch
import triton
from triton import language as tl

from flag_gems.utils.shape_utils import (
    heuristics_for_tile_size,
    heuristics_for_num_warps,
    stride_order,
)
from flag_gems.utils.tensor_wrapper import StridedBuffer
from flag_gems.utils.libentry import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.runtime import torch_device_fn


def relu_forward_wrapper_rank_1(in0: Union[torch.Tensor, StridedBuffer], /, *, out0: Union[torch.Tensor, StridedBuffer]): 
    """Generated wrapper function with Pointwise: StridedBuffer, StridedBuffer(a1!) -> StridedBuffer(a1!)"""
    assert in0.shape == out0.shape, 'operand shapes mismatch'
    # task partitioning
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    tile_sizes = heuristics_for_tile_size(256, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes))
    num_ctas = min(65536, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta==1
    grid = (num_ctas, 1, 1)
    # kernel launch
    in0_strides = in0.stride()
    in0_stride_order = (0,)
    out0_strides = out0.stride()
    out0_stride_order = (0,)
    with torch_device_fn.device(in0.device.index):
        relu_forward_kernel_rank_1[grid](
            in0, out0,
            in0_strides[0], # stride for in0
            in0_stride_order[0], # stride order for in0
            out0_strides[0], # stride for out0
            out0_stride_order[0], # stride orderfor out0
            shape[0], # task indexing space
            num_tasks, # num tasks
            tiles_per_cta=tiles_per_cta, # tiles_per_cta
            tile_size0=tile_sizes[0],
            one_tile_per_cta=one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0

@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)

@libentry()
@triton.jit
def relu_forward_kernel_rank_1(
    in0_ptr: tl.tensor, # of tl.pointer_type
    out0_ptr: tl.tensor, # of tl.pointer_type
    in0_stride0: int, # strides for in0
    in0_stride_order0: tl.constexpr, # stride order for in0
    out0_stride0: int, # strides for out0
    out0_stride_order0: tl.constexpr, # stride order for out0
    s0, # task_space
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tle.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta: # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id0 = tile_id

        # tile offsets
        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        # loads
        in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

        # compute
        out0 = relu_forward(in0)

        # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
        out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
    else: # grid-stride-loop style kernel
        num_ctas = tle.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id0 = tile_id

            # tile offsets
            offset0 = (tile_id0 * tile_size0).to(tl.int32)
            # loads
            in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

            # compute
            out0 = relu_forward(in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))

