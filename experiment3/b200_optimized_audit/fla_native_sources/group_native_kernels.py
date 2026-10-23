from __future__ import annotations
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
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [32, 64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=['D', 'NB', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def layer_norm_fwd_kernel(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_g = o_t % G
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]

    p_x = x + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = res + o_t[:, None] * D + o_d[None, :]
        b_x += tl.load(p_res, mask=m_x, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = res_out + o_t[:, None] * D + o_d[None, :]
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), mask=m_x)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = mean + o_t
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), mask=m_t)
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = rstd + o_t
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    p_y = y + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['D', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_fwd_kernel_row(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    G: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_g = i_t % G

    x += i_t * D
    y += i_t * D
    if HAS_RESIDUAL:
        res += i_t * D
    if STORE_RESIDUAL_OUT:
        res_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(res + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(res_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    tl.store(y + o_d, b_y, mask=m_d)

