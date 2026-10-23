# Exact original decorated FLA forward functions. MIT; see fla/LICENSE archive.
import importlib.util,sys
from pathlib import Path
import torch
import triton
import triton.language as tl

def _load_source(name,relative):
    path=Path(__file__).resolve().parent/relative
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

_env=_load_source('_exp3_fla_native_env','fla/utils/env.py')
autotune_cache_kwargs=_env.autotune_cache_kwargs

_cache=_load_source('_exp3_fla_native_cache','fla/ops/utils/cache.py')
fla_cache_autotune=_cache.fla_cache_autotune
IS_AMD=False  # CUDA-only adapter; exact original hardware.py backend predicate
NUM_WARPS_AUTOTUNE=[2,4,8,16] if IS_AMD else [4,8,16,32]

@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_y = (o_t < T)[:, None] & m_d[None, :]

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            # [BT, BD]
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # separate branches avoid a Triton compiler issue.
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            # [BT, BD]
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d, mask=m_x, other=0).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = residual + bos * D + o_t[:, None] * D + o_d[None, :]
        b_residual = tl.load(p_residual, mask=m_y, other=0.0)
        b_y += b_residual

    p_y = y + bos * D + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)

