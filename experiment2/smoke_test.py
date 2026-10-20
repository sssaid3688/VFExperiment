#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B200 基准 25 算子 → 昇腾 (Ascend950PR / A5) 开源适配代码冒烟测试.

目的:
  - 验证各权威开源仓拉取下来的昇腾适配代码能否在本机 (triton-ascend 3.2.2,
    torch 2.7.1 + torch_npu, CANN 9.1.0-beta.3) 直接运行;
  - 只做正确性对照 (fp32 参考, 相对误差 <= 2e-2, 与 B200 基准口径一致),
    不做性能计时;
  - 逐算子独立 try/except, 单个失败不影响后续; 失败只记录原因, 不改环境.

代码来源 (third_party/):
  FlagGems v5.0.2        —— 16 个算子 (runtime/backend/_ascend + 通用 Triton)
  Liger-Kernel v0.8.4    —— 7 个算子 (ops/backends/_ascend, 910B 调优)
  vllm v0.24.0           —— unified_attention (glue/vllm 下 shim 后导入)
  triton-ascend tutorials —— 06-fused-attention.py (FA2)

用法: ./run_smoke.sh
"""

import csv
import importlib.util
import os
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

EXP2 = Path(__file__).resolve().parent
TP = EXP2 / "third_party"
GLUE = EXP2 / "glue"

# 顺序很重要: glue 在最前, 保证 `import vllm` 命中我们的 shim 而不是真实 vllm
for p in (
    str(GLUE),
    str(TP / "FlagGems-5.0.2" / "src"),
    str(TP / "Liger-Kernel-0.8.4" / "src"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("GEMS_VENDOR", "ascend")  # flag_gems 设备检测直连 _ascend 后端

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_npu  # noqa: E402

DEV = "npu:0"
torch.npu.set_device(0)
torch.manual_seed(0)

RTOL, ATOL = 2e-2, 2e-2  # 与 B200 基准一致的正确性判据


# ---------------------------------------------------------------------------
# 缺失可选依赖打桩 (不安装任何包, 不改环境)
# ---------------------------------------------------------------------------


def _install_stub_modules():
    """为 venv 中缺失的可选依赖提供等价桩, 仅在真正缺失时介入.

    1. transformers.utils.is_torch_npu_available
       —— Liger _ascend 后端用它检测 NPU; venv 没装 transformers,
          用 torch.npu.is_available() 等价替代.
    2. flag_gems.utils.models (SQLAlchemy 持久化层)
       —— flag_gems 把 autotune 结果写 sqlite, venv 没装 sqlalchemy;
          用内存 no-op 模型替代, 不影响 kernel 运行, 只是 autotune 结果不落盘.
    """
    import importlib.util as _ilu

    def _missing(name):
        try:
            return _ilu.find_spec(name) is None
        except (ModuleNotFoundError, ValueError):
            return True

    if _missing("transformers"):
        tfm = types.ModuleType("transformers")
        tfu = types.ModuleType("transformers.utils")
        tfu.is_torch_npu_available = lambda: torch.npu.is_available()
        tfm.utils = tfu
        sys.modules.setdefault("transformers", tfm)
        sys.modules.setdefault("transformers.utils", tfu)

    if _missing("sqlalchemy"):
        class _NoDBModel:
            """PersistantModel 的内存替身: 语义与 sqlite 版一致 (put 后 get 可读回),
            只是进程退出后不落盘; 供 flag_gems autotune 的 先存后读 流程使用."""

            def __init__(self, *args, **kwargs):
                self._configs = {}
                self._benchmarks = {}

            def get_config(self, table, key):
                return self._configs.get((table, key))

            def put_config(self, table, key, config):
                self._configs[(table, key)] = config

            def get_benchmark(self, table, key, config):
                return self._benchmarks.get((table, key, str(config)))

            def put_benchmark(self, table, key, config, benchmark):
                self._benchmarks[(table, key, str(config))] = benchmark

        fgm = types.ModuleType("flag_gems.utils.models")
        fgm.PersistantModel = _NoDBModel
        fgm.SQLPersistantModel = _NoDBModel
        sys.modules.setdefault("flag_gems.utils.models", fgm)


_install_stub_modules()

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def npu_dev(t, dtype=None):
    return t.to(device=DEV, dtype=dtype) if dtype is not None else t.to(DEV)


def check_close(out, ref, rtol=RTOL, atol=ATOL, tag=""):
    a = out.detach().float().cpu()
    b = ref.detach().float().cpu()
    if a.shape != b.shape:
        raise RuntimeError(f"{tag} shape mismatch: out{tuple(a.shape)} vs ref{tuple(b.shape)}")
    diff = (a - b).abs()
    max_abs = diff.max().item() if a.numel() else 0.0
    denom = b.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item() if a.numel() else 0.0
    ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    if not ok:
        raise RuntimeError(f"{tag} correctness mismatch: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return f"max_abs={max_abs:.2e} max_rel={max_rel:.2e}"


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# 惰性导入 (导入失败不影响其它组)
# ---------------------------------------------------------------------------

_fg = {"mod": None, "err": None, "mode": ""}


def get_flag_gems():
    if _fg["err"] is not None:
        raise RuntimeError("import flag_gems failed:\n" + _fg["err"])
    if _fg["mod"] is None:
        try:
            import flag_gems

            _fg["mod"], _fg["mode"] = flag_gems, "full-import"
        except Exception:
            tb = traceback.format_exc()
            try:
                # 兜底: 跳过 flag_gems/__init__.py 的全量导入, 手工注册包路径后
                # 按需加载单个 op 模块 (不执行 ops/__init__.py 的 246 个导入)
                src = TP / "FlagGems-5.0.2" / "src" / "flag_gems"
                pkg = types.ModuleType("flag_gems")
                pkg.__path__ = [str(src)]
                sys.modules["flag_gems"] = pkg
                ops_pkg = types.ModuleType("flag_gems.ops")
                ops_pkg.__path__ = [str(src / "ops")]
                sys.modules["flag_gems.ops"] = ops_pkg

                class _FG:
                    pass

                fg = _FG()
                for name in (
                    "softmax", "silu", "mm", "gelu", "add", "mul", "sum_dim",
                    "argmax", "clamp", "cumsum", "topk", "embedding", "bmm",
                    "addmm", "mv", "fp8_matmul",
                ):
                    setattr(fg, name, importlib.import_module(f"flag_gems.ops.{name}").__dict__.get(name))
                _fg["mod"], _fg["mode"] = fg, "module-injection"
            except Exception:
                _fg["err"] = tb
                raise RuntimeError("import flag_gems failed:\n" + tb)
    return _fg["mod"]


_liger_cache = {}


def get_liger_op(opname):
    key = f"liger_kernel.ops.backends._ascend.ops.{opname}"
    if key not in _liger_cache:
        try:
            _liger_cache[key] = importlib.import_module(key)
        except Exception:
            _liger_cache[key] = traceback.format_exc()
    if isinstance(_liger_cache[key], str):
        raise RuntimeError(f"import {key} failed:\n" + _liger_cache[key])
    return _liger_cache[key]


_vllm_ua = {"fn": None, "err": None}


def get_unified_attention():
    if _vllm_ua["err"] is not None:
        raise RuntimeError("import vllm unified_attention failed:\n" + _vllm_ua["err"])
    if _vllm_ua["fn"] is None:
        try:
            from vllm.v1.attention.ops.triton_unified_attention import unified_attention

            _vllm_ua["fn"] = unified_attention
        except Exception:
            _vllm_ua["err"] = traceback.format_exc()
            raise RuntimeError("import vllm unified_attention failed:\n" + _vllm_ua["err"])
    return _vllm_ua["fn"]


_fa2 = {"mod": None, "err": None}


def get_tutorial_fa2():
    if _fa2["err"] is not None:
        raise RuntimeError("import triton-ascend tutorial 06 failed:\n" + _fa2["err"])
    if _fa2["mod"] is None:
        try:
            # 用与本机 triton-ascend 3.2.2 版本匹配的 v3.2.2 tag 教程;
            # main 分支版本基于 Triton 3.6 API (tensor_descriptor/set_allocator/
            # get_active_torch_device 等), 在 3.2.2 上编译不过
            tut = TP / "triton-ascend-v3.2.2" / "python" / "tutorials" / "06-fused-attention.py"
            try:
                import pytest  # noqa: F401
            except ImportError:
                # 教程仅用 pytest.mark.parametrize 装饰 bench 函数, 打桩即可, 不装包
                stub = types.ModuleType("pytest")
                stub.mark = types.SimpleNamespace(
                    parametrize=lambda *a, **k: (lambda f: f)
                )
                sys.modules["pytest"] = stub
            spec = importlib.util.spec_from_file_location("fa2_tutorial", tut)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _fa2["mod"] = mod
        except Exception:
            _fa2["err"] = traceback.format_exc()
            raise RuntimeError("import triton-ascend tutorial 06 failed:\n" + _fa2["err"])
    return _fa2["mod"]


# ---------------------------------------------------------------------------
# 25 个算子测试
# ---------------------------------------------------------------------------


def t01_unified_attention():
    """vllm v0.24.0 triton_unified_attention: varlen Q + paged KV + causal + GQA."""
    ua = get_unified_attention()
    B, Sq, Skv, Hq, Hkv, D = 2, 256, 256, 8, 2, 64
    bs = 64  # page size
    nblk_per_seq = Skv // bs
    total_q = B * Sq
    scale = 1.0 / (D**0.5)

    q = torch.randn(total_q, Hq, D, dtype=torch.bfloat16)
    k = torch.randn(B * nblk_per_seq * bs, Hkv, D, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)

    q_d, out_d = npu_dev(q), npu_dev(out)
    k_d = npu_dev(k).view(B * nblk_per_seq, bs, Hkv, D)  # paged KV cache
    v_d = npu_dev(v).view(B * nblk_per_seq, bs, Hkv, D)

    cu_seqlens_q = torch.arange(0, B + 1, dtype=torch.int32) * Sq
    seqused_k = torch.full((B,), Skv, dtype=torch.int32)
    block_table = torch.arange(B * nblk_per_seq, dtype=torch.int32).view(B, nblk_per_seq)

    ua(
        q_d, k_d, v_d, out_d,
        npu_dev(cu_seqlens_q), Sq, npu_dev(seqused_k), Skv,
        scale, True, (-1, -1), npu_dev(block_table),
        0.0, None, None, None,
    )
    torch.npu.synchronize()

    # fp32 参考 (逐序列, GQA 展开)
    ref = torch.empty_like(q, dtype=torch.float32)
    for i in range(B):
        qi = q[i * Sq : (i + 1) * Sq].float()
        ki = k[i * Skv : (i + 1) * Skv].float()
        vi = v[i * Skv : (i + 1) * Skv].float()
        rep = Hq // Hkv
        ki = ki.repeat_interleave(rep, dim=1)
        vi = vi.repeat_interleave(rep, dim=1)
        att = torch.einsum("lhd,shd->lhs", qi, ki) * scale
        mask = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        p = att.softmax(-1)
        ref[i * Sq : (i + 1) * Sq] = torch.einsum("lhs,shd->lhd", p, vi)
    return check_close(out_d, ref, tag="unified_attention")


def t02_flash_attention():
    """triton-ascend 官方 FA2 tutorial (v3.2.2 tag, 与本机 triton-ascend 版本匹配):
    (Z,H,N,D), causal, forward(q,k,v,causal,sm_scale)."""
    mod = get_tutorial_fa2()
    Z, H, N, D = 2, 4, 256, 64
    scale = 1.0 / (D**0.5)

    last_err = None
    for dtype in (torch.bfloat16, torch.float16):  # 优先 bf16 (B200 口径), 失败退 fp16
        q = torch.randn(Z, H, N, D, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        q_d, k_d, v_d = npu_dev(q), npu_dev(k), npu_dev(v)
        try:
            o_d = mod._attention.apply(q_d, k_d, v_d, True, scale)
            torch.npu.synchronize()
            ref = torch.empty(Z, H, N, D, dtype=torch.float32)
            for z in range(Z):
                for h in range(H):
                    att = (q[z, h].float() @ k[z, h].float().T) * scale
                    att = att.masked_fill(
                        ~torch.tril(torch.ones(N, N, dtype=torch.bool)), float("-inf")
                    )
                    ref[z, h] = att.softmax(-1) @ v[z, h].float()
            note = check_close(o_d, ref, tag=f"flash_attention[{str(dtype).split('.')[-1]}]")
            return note + f" dtype={str(dtype).split('.')[-1]}"
        except Exception:
            last_err = traceback.format_exc()
    raise RuntimeError(last_err)


# ---- FlagGems 16 算子 -------------------------------------------------------


def t03_softmax():
    g = get_flag_gems()
    x = torch.randn(256, 4096, dtype=torch.bfloat16)
    xd = npu_dev(x)
    out = g.softmax(xd, -1)
    torch.npu.synchronize()
    return check_close(out, F.softmax(x.float(), -1), tag="softmax")


def t04_silu():
    g = get_flag_gems()
    x = torch.randn(1024, 4096, dtype=torch.bfloat16)
    xd = npu_dev(x)
    out = g.silu(xd)
    torch.npu.synchronize()
    return check_close(out, F.silu(x.float()), tag="silu")


def t05_rms_norm():
    m = get_liger_op("rms_norm")
    X = torch.randn(256, 2048, dtype=torch.bfloat16)
    W = torch.randn(2048, dtype=torch.bfloat16)
    eps = 1e-6
    ref = (X.float() * torch.rsqrt(X.float().pow(2).mean(-1, keepdim=True) + eps)).to(
        torch.bfloat16
    ).float() * W.float()
    out = m.LigerRMSNormFunction.apply(npu_dev(X.clone()), npu_dev(W), eps)
    torch.npu.synchronize()
    return check_close(out, ref, tag="rms_norm")


def t06_mm():
    g = get_flag_gems()
    a = torch.randn(256, 1024, dtype=torch.bfloat16)
    b = torch.randn(1024, 512, dtype=torch.bfloat16)
    out = g.mm(npu_dev(a), npu_dev(b))
    torch.npu.synchronize()
    return check_close(out, a.float() @ b.float(), tag="mm")


def t07_layer_norm():
    m = get_liger_op("layer_norm")
    X = torch.randn(256, 2048, dtype=torch.bfloat16)
    W = torch.randn(2048, dtype=torch.bfloat16)
    Bt = torch.randn(2048, dtype=torch.bfloat16)
    out = m.LigerLayerNormFunction.apply(npu_dev(X), npu_dev(W), npu_dev(Bt), 1e-6)
    torch.npu.synchronize()
    ref = F.layer_norm(X.float(), (2048,), W.float(), Bt.float(), 1e-6)
    return check_close(out, ref, tag="layer_norm")


def t08_gelu():
    g = get_flag_gems()
    x = torch.randn(1024, 4096, dtype=torch.bfloat16)
    out = g.gelu(npu_dev(x))
    torch.npu.synchronize()
    return check_close(out, F.gelu(x.float()), tag="gelu")


def t09_add():
    g = get_flag_gems()
    a = torch.randn(1024, 4096, dtype=torch.bfloat16)
    b = torch.randn(1024, 4096, dtype=torch.bfloat16)
    out = g.add(npu_dev(a), npu_dev(b))
    torch.npu.synchronize()
    return check_close(out, a.float() + b.float(), tag="add")


def t10_mul():
    g = get_flag_gems()
    a = torch.randn(1024, 4096, dtype=torch.bfloat16)
    b = torch.randn(1024, 4096, dtype=torch.bfloat16)
    out = g.mul(npu_dev(a), npu_dev(b))
    torch.npu.synchronize()
    return check_close(out, a.float() * b.float(), tag="mul")


def t11_sum():
    g = get_flag_gems()
    x = torch.randn(512, 4096, dtype=torch.float32)  # B200 基准: sum 正确性用 fp32
    out = g.sum_dim(npu_dev(x), dim=[-1])  # flag_gems sum_dim 要求 dim 为可迭代
    torch.npu.synchronize()
    return check_close(out, x.sum(-1), tag="sum")


def t12_argmax():
    g = get_flag_gems()
    x = torch.randn(512, 4096, dtype=torch.bfloat16)
    out = g.argmax(npu_dev(x), dim=-1)
    torch.npu.synchronize()
    ref = x.float().argmax(-1)
    if out.shape != ref.shape or not torch.equal(out.cpu(), ref):
        raise RuntimeError(f"argmax mismatch: {(out.cpu() != ref).sum()} of {ref.numel()} entries")
    return "exact match"


def t13_cross_entropy():
    m = get_liger_op("cross_entropy")
    x = torch.randn(256, 1024, dtype=torch.bfloat16)
    tgt = torch.randint(0, 1024, (256,))
    res = m.LigerCrossEntropyFunction.apply(npu_dev(x), npu_dev(tgt), None)
    torch.npu.synchronize()
    loss = res[0] if isinstance(res, tuple) else res
    ref = F.cross_entropy(x.float(), tgt)
    return check_close(loss, ref, tag="cross_entropy")


def t14_fused_add_rms_norm():
    m = get_liger_op("fused_add_rms_norm")
    X = torch.randn(256, 2048, dtype=torch.bfloat16)
    R = torch.randn(256, 2048, dtype=torch.bfloat16)
    W = torch.randn(2048, dtype=torch.bfloat16)
    eps = 1e-6
    s = (X.float() + R.float())
    ref = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16).float() * W.float()
    Y, S = m.LigerFusedAddRMSNormFunction.apply(npu_dev(X.clone()), npu_dev(R.clone()), npu_dev(W), eps)
    torch.npu.synchronize()
    note = check_close(Y, ref, tag="fused_add_rms_norm.Y")
    note += "; " + check_close(S, s.to(torch.bfloat16).float(), tag=".S(residual)")
    return note


def t15_swiglu():
    m = get_liger_op("swiglu")
    a = torch.randn(256, 4096, dtype=torch.bfloat16)
    b = torch.randn(256, 4096, dtype=torch.bfloat16)
    out = m.swiglu_forward(npu_dev(a), npu_dev(b))
    torch.npu.synchronize()
    return check_close(out, F.silu(a.float()) * b.float(), tag="swiglu")


def t16_rope():
    m = get_liger_op("rope")
    B, Hq, Hkv, S, D = 1, 8, 2, 256, 128
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    t = torch.arange(S).float()
    freqs = torch.outer(t, inv)
    # Liger _ascend rope 约定: cos/sin 为半宽表 (1, S, D/2), 内核只加载前 half_hd
    cos = freqs.cos()[None]
    sin = freqs.sin()[None]
    q = torch.randn(B, Hq, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)

    q_d, k_d = npu_dev(q.clone()), npu_dev(k.clone())
    res = m.rope_forward(q_d, k_d, npu_dev(cos).to(torch.bfloat16), npu_dev(sin).to(torch.bfloat16))
    torch.npu.synchronize()
    qo = res[0] if isinstance(res, tuple) else res
    ko = res[1] if isinstance(res, tuple) and len(res) > 1 else None

    # 参考: HF NeoX 全宽风格, 全宽表 = cat([freqs, freqs]) (cos[d]==cos[d+D/2],
    # 与内核的半宽表数学等价; 不能用 repeat_interleave —— 那是 GPT-J 配对)
    cos_full = torch.cat([cos.to(torch.bfloat16).float()] * 2, dim=-1)
    sin_full = torch.cat([sin.to(torch.bfloat16).float()] * 2, dim=-1)
    ref_q = q.float() * cos_full + rotate_half(q.float()) * sin_full
    note = check_close(qo, ref_q, tag="rope.q")
    if ko is not None:
        ref_k = k.float() * cos_full + rotate_half(k.float()) * sin_full
        note += "; " + check_close(ko, ref_k, tag="rope.k")
    return note


def t17_clamp():
    g = get_flag_gems()
    x = torch.randn(1024, 4096, dtype=torch.bfloat16)
    out = g.clamp(npu_dev(x), -0.5, 0.5)
    torch.npu.synchronize()
    return check_close(out, x.float().clamp(-0.5, 0.5), tag="clamp")


def t18_cumsum():
    g = get_flag_gems()
    x = torch.randn(256, 4096, dtype=torch.float32)  # B200 基准: cumsum 正确性用 fp32
    out = g.cumsum(npu_dev(x), -1)
    torch.npu.synchronize()
    return check_close(out, x.cumsum(-1), tag="cumsum")


def t19_topk():
    g = get_flag_gems()
    x = torch.randn(256, 4096, dtype=torch.bfloat16)
    vals, idx = g.topk(npu_dev(x), 50, dim=-1)
    torch.npu.synchronize()
    ref_vals = x.float().topk(50, dim=-1).values
    sv = vals.float().cpu().sort(descending=True).values
    sr = ref_vals.sort(descending=True).values
    note = check_close(sv, sr, rtol=1e-2, atol=1e-2, tag="topk.values(sorted)")
    # 检查索引一致性: 按 argmax 抽查首元素
    if not torch.equal(idx[:, 0].cpu(), x.float().argmax(-1)):
        raise RuntimeError("topk index mismatch at k=0")
    return note + " idx[0] exact"


def t20_embedding():
    g = get_flag_gems()
    w = torch.randn(1000, 512, dtype=torch.bfloat16)
    ids = torch.randint(0, 1000, (8, 64))
    out = g.embedding(npu_dev(w), npu_dev(ids))
    torch.npu.synchronize()
    return check_close(out, F.embedding(ids, w.float()), tag="embedding")


def t21_bmm():
    g = get_flag_gems()
    a = torch.randn(16, 256, 512, dtype=torch.bfloat16)
    b = torch.randn(16, 512, 128, dtype=torch.bfloat16)
    out = g.bmm(npu_dev(a), npu_dev(b))
    torch.npu.synchronize()
    return check_close(out, torch.bmm(a.float(), b.float()), tag="bmm")


def t22_addmm():
    g = get_flag_gems()
    bias = torch.randn(512, dtype=torch.bfloat16)
    a = torch.randn(256, 1024, dtype=torch.bfloat16)
    b = torch.randn(1024, 512, dtype=torch.bfloat16)
    out = g.addmm(npu_dev(bias), npu_dev(a), npu_dev(b))
    torch.npu.synchronize()
    ref = bias.float() + a.float() @ b.float()
    return check_close(out, ref, tag="addmm")


def t23_mv():
    g = get_flag_gems()
    a = torch.randn(512, 1024, dtype=torch.bfloat16)
    v = torch.randn(1024, dtype=torch.bfloat16)
    out = g.mv(npu_dev(a), npu_dev(v))
    torch.npu.synchronize()
    return check_close(out, a.float() @ v.float(), tag="mv")


def t24_fp8_matmul():
    g = get_flag_gems()
    M, K, N = 256, 512, 512
    a = (torch.randn(M, K) * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 0.25).to(torch.float8_e4m3fn)
    a_s = torch.rand(M, K // 128) + 0.5
    b_s = torch.rand(N // 128, K // 128) + 0.5
    out = g.fp8_matmul(npu_dev(a), npu_dev(a_s), npu_dev(b), npu_dev(b_s))
    torch.npu.synchronize()
    # 参考: 用量化后的 fp8 值 × 块 scale, fp32 精确累加
    a_f = a.float() * a_s.repeat_interleave(128, dim=1)
    b_f = b.float() * b_s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    ref = a_f @ b_f.T
    return check_close(out, ref, rtol=3e-2, atol=3e-2, tag="fp8_matmul")


def t25_fused_linear_cross_entropy():
    m = get_liger_op("fused_linear_cross_entropy")
    x = torch.randn(256, 1024, dtype=torch.bfloat16)
    w = torch.randn(1000, 1024, dtype=torch.bfloat16)
    tgt = torch.randint(0, 1000, (256,))
    res = m.LigerFusedLinearCrossEntropyFunction.apply(npu_dev(x), npu_dev(w), npu_dev(tgt))
    torch.npu.synchronize()
    loss = res[0] if isinstance(res, tuple) else res
    ref = F.cross_entropy(x.float() @ w.float().T, tgt)
    return check_close(loss, ref, tag="fused_linear_cross_entropy")


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CASES = [
    (1, "unified_attention", "vllm v0.24.0", "glue/vllm (shim) + triton-ascend", t01_unified_attention),
    (2, "flash_attention", "triton-ascend tutorial", "third_party/triton-ascend-tutorials/06-fused-attention.py", t02_flash_attention),
    (3, "softmax", "FlagGems v5.0.2", "ops/softmax.py (+_ascend)", t03_softmax),
    (4, "silu", "FlagGems v5.0.2", "ops/silu.py", t04_silu),
    (5, "rms_norm", "Liger-Kernel v0.8.4", "_ascend/ops/rms_norm.py", t05_rms_norm),
    (6, "mm", "FlagGems v5.0.2", "ops/mm.py (+_ascend)", t06_mm),
    (7, "layer_norm", "Liger-Kernel v0.8.4", "_ascend/ops/layer_norm.py", t07_layer_norm),
    (8, "gelu", "FlagGems v5.0.2", "ops/gelu.py", t08_gelu),
    (9, "add", "FlagGems v5.0.2", "ops/add.py", t09_add),
    (10, "mul", "FlagGems v5.0.2", "ops/mul.py", t10_mul),
    (11, "sum", "FlagGems v5.0.2", "ops/sum.py", t11_sum),
    (12, "argmax", "FlagGems v5.0.2", "ops/argmax.py (+_ascend)", t12_argmax),
    (13, "cross_entropy", "Liger-Kernel v0.8.4", "_ascend/ops/cross_entropy.py", t13_cross_entropy),
    (14, "fused_add_rms_norm", "Liger-Kernel v0.8.4", "_ascend/ops/fused_add_rms_norm.py", t14_fused_add_rms_norm),
    (15, "swiglu", "Liger-Kernel v0.8.4", "_ascend/ops/swiglu.py", t15_swiglu),
    (16, "rope", "Liger-Kernel v0.8.4", "_ascend/ops/rope.py", t16_rope),
    (17, "clamp", "FlagGems v5.0.2", "ops/clamp.py", t17_clamp),
    (18, "cumsum", "FlagGems v5.0.2", "ops/cumsum.py (+_ascend)", t18_cumsum),
    (19, "topk", "FlagGems v5.0.2", "ops/topk.py", t19_topk),
    (20, "embedding", "FlagGems v5.0.2", "ops/embedding.py (+_ascend)", t20_embedding),
    (21, "bmm", "FlagGems v5.0.2", "ops/bmm.py (+_ascend)", t21_bmm),
    (22, "addmm", "FlagGems v5.0.2", "ops/addmm.py (+_ascend)", t22_addmm),
    (23, "mv", "FlagGems v5.0.2", "ops/mv.py", t23_mv),
    (24, "fp8_matmul", "FlagGems v5.0.2", "ops/fp8_matmul.py", t24_fp8_matmul),
    (25, "fused_linear_cross_entropy", "Liger-Kernel v0.8.4", "_ascend/ops/fused_linear_cross_entropy.py", t25_fused_linear_cross_entropy),
]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = EXP2 / "results"
    res_dir.mkdir(exist_ok=True)
    csv_path = res_dir / f"smoke_{ts}.csv"

    print("=" * 100)
    print("Ascend 25-op smoke test | dev=NPU:0")
    try:
        import triton

        print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
              f"triton={triton.__version__} (triton-ascend backend)")
    except Exception:
        print("triton version query failed")
    try:
        print("device:", torch.npu.get_device_name(0))
    except Exception:
        print("device name query failed")
    print("=" * 100, flush=True)

    n_pass = n_fail = 0
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["no", "op", "source", "impl", "status", "seconds", "note"])
        f.flush()
        for no, op, src, impl, fn in CASES:
            t0 = time.time()
            try:
                note = fn() or ""
                status = "PASS"
                n_pass += 1
            except Exception:
                note = traceback.format_exc(limit=8)[-1200:]
                status = "FAIL"
                n_fail += 1
            dt = time.time() - t0
            wr.writerow([no, op, src, impl, status, f"{dt:.1f}", note])
            f.flush()
            print(f"[{status}] #{no:02d} {op:<30s} ({dt:6.1f}s) {note.splitlines()[-1] if note else ''}", flush=True)

    print("=" * 100)
    print(f"TOTAL: {n_pass} PASS / {n_fail} FAIL  (csv: {csv_path})")
    if _fg["mode"]:
        print(f"flag_gems import mode: {_fg['mode']}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
