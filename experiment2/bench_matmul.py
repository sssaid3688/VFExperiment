#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
矩阵乘相关补充算子 B200 基准测试 (5 个, 全部 kaixin env)

1) bmm                        - FlagGems 批量矩阵乘 (注意力 QK^T/PV, head-batched)
2) addmm                      - FlagGems 偏置融合 GEMM (Linear: y = x@W + b)
3) mv                         - FlagGems 矩阵-向量乘 (GEMV, decode Linear)
4) fp8_matmul                 - FlagGems MXFP8 分块量化 GEMM (128 分块 scale)
5) fused_linear_cross_entropy - Liger GEMM+CE 融合 (LLM 训练输出层, 免 logits 物化)

运行: /root/miniconda3/envs/kaixin/bin/python bench_matmul.py [--only bmm,addmm,...]
"""

import argparse
import traceback

import torch
import torch.nn.functional as F

import flag_gems
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

from common import bench_one, save_result, clear_result, load_results, torch_version_str
import shapes as S

DEV = "cuda"
DTYPE_SUFFIX = ""  # --dtype 模式下自动重绑
DT = torch.bfloat16
FG = "FlagOpen/FlagGems v5.0.2 (Triton)"
LG = "linkedin/Liger-Kernel v0.8.4 (Triton)"

print(f"[ENV] {torch_version_str()}, GPU={torch.cuda.get_device_name(0)}, "
      f"cap={torch.cuda.get_device_capability(0)}", flush=True)


def rel_err(out, ref):
    e = (out.float() - ref).abs().max().item()
    return e, e / (ref.abs().max().item() + 1e-6)


def bench_bmm():
    op = "bmm"
    for idx, (B, M, K, N, tag) in enumerate(S.BMM_SHAPES):
        a = torch.randn(B, M, K, device=DEV, dtype=DT)
        b = torch.randn(B, K, N, device=DEV, dtype=DT)
        ref = torch.bmm(a.float(), b.float())
        out = flag_gems.bmm(a, b)
        err, rel = rel_err(out, ref)
        flops = 2 * B * M * K * N
        nbytes = 2 * (B * M * K + B * K * N + B * M * N)
        m = bench_one(lambda: flag_gems.bmm(a, b), flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"({B},{M},{K}) x ({B},{K},{N})",
               "dims": f"B={B}, M={M}, K={K}, N={N}",
               "elements": B * M * N, "flops": flops, "bytes": nbytes,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 5e-2 else f"CHECK(rel={rel:.2e})",
               "source": FG}
        row.update(m)
        save_result(op + DTYPE_SUFFIX, row)


def bench_addmm():
    op = "addmm"   # y = beta*bias + alpha*(x @ w)
    for idx, (M, K, N, tag) in enumerate(S.MM_SHAPES):
        x = torch.randn(M, K, device=DEV, dtype=DT)
        w = torch.randn(K, N, device=DEV, dtype=DT)
        bias = torch.randn(N, device=DEV, dtype=DT)
        ref = x.float() @ w.float() + bias.float()
        out = flag_gems.addmm(bias, x, w, beta=1.0, alpha=1.0)
        err, rel = rel_err(out, ref)
        flops = 2 * M * K * N + M * N
        nbytes = 2 * (M * K + K * N + M * N) + 2 * N
        m = bench_one(lambda: flag_gems.addmm(bias, x, w, beta=1.0, alpha=1.0), flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"({M},{K}) x ({K},{N}) + ({N},)",
               "dims": f"M={M}, K={K}, N={N}, bias={N}",
               "elements": M * N, "flops": flops, "bytes": nbytes,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 5e-2 else f"CHECK(rel={rel:.2e})",
               "source": FG}
        row.update(m)
        save_result(op + DTYPE_SUFFIX, row)


def bench_mv():
    op = "mv"      # y = A @ v
    for idx, (M, K, tag) in enumerate(S.MV_SHAPES):
        a = torch.randn(M, K, device=DEV, dtype=DT)
        v = torch.randn(K, device=DEV, dtype=DT)
        ref = a.float() @ v.float()
        out = flag_gems.mv(a, v)
        err, rel = rel_err(out, ref)
        flops = 2 * M * K
        nbytes = 2 * (M * K + K) + 2 * M
        m = bench_one(lambda: flag_gems.mv(a, v), flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"({M},{K}) x ({K},)",
               "dims": f"M={M}, K={K}",
               "elements": M, "flops": flops, "bytes": nbytes,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 5e-2 else f"CHECK(rel={rel:.2e})",
               "source": FG}
        row.update(m)
        save_result(op + DTYPE_SUFFIX, row)


def _quant_tokens(x):
    """MXFP8 激活量化: x[M,K] fp32 -> q[M,K] e4m3, s[M,K//128] (每 token 128 分组)"""
    M, K = x.shape
    xg = x.view(M, K // 128, 128)
    s = xg.abs().amax(-1) / 448.0
    q = (xg / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(M, K), s.contiguous()


def _quant_blocks(y):
    """MXFP8 权重量化: y[K,N] -> b[N,K] e4m3, s[N//128, K//128] (128x128 分块)"""
    yt = y.t().contiguous()          # [N, K]
    N, K = yt.shape
    yb = yt.view(N // 128, 128, K // 128, 128)
    s = yb.abs().amax(dim=(1, 3)) / 448.0
    q = (yb / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
    return q.contiguous(), s.contiguous()


def bench_fp8_matmul():
    op = "fp8_matmul"
    for idx, (M, K, N, tag) in enumerate(S.FP8_MM_SHAPES):
        x = torch.randn(M, K, device=DEV)
        y = torch.randn(K, N, device=DEV)
        aq, asc = _quant_tokens(x)
        bq, bsc = _quant_blocks(y)
        # 参考: 反量化后的 fp32 GEMM
        xd = (aq.view(M, K // 128, 128).float() * asc.unsqueeze(-1)).view(M, K)
        yd = (bq.view(N // 128, 128, K // 128, 128).float()
              * bsc[:, None, :, None]).view(N, K).t()
        ref = xd @ yd
        out = flag_gems.fp8_matmul(aq, asc, bq, bsc)
        err, rel = rel_err(out, ref)
        flops = 2 * M * K * N
        nbytes = (M * K + N * K + 4 * (M * (K // 128) + (N // 128) * (K // 128))
                  + 2 * M * N)          # fp8 1B + scale 4B + bf16 out
        m = bench_one(lambda: flag_gems.fp8_matmul(aq, asc, bq, bsc), flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"({M},{K}) x ({K},{N}) fp8-blk128",
               "dims": f"M={M}, K={K}, N={N}, quant=MXFP8-128",
               "elements": M * N, "flops": flops, "bytes": nbytes,
               "dtype": "fp8(e4m3)+scale",
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 8e-2 else f"CHECK(rel={rel:.2e})",
               "source": FG}
        row.update(m)
        save_result(op + DTYPE_SUFFIX, row)


def flce_ref_chunked(inp, w, tgt, max_chunk_elems=1 << 24):
    """分块 fp32 FLCE 参考 (避免大 T*V logits 物化; 每 chunk ≤ 16M 输出元素)"""
    T = inp.shape[0]
    V = w.shape[0]
    chunk = max(1, max_chunk_elems // V)
    wf = w.float()  # 提出循环, 避免重复物化
    outs = []
    for i in range(0, T, chunk):
        logits = inp[i:i + chunk].float() @ wf.t()
        outs.append(F.cross_entropy(logits, tgt[i:i + chunk], reduction="none"))
    return torch.cat(outs)


def bench_fused_linear_cross_entropy():
    op = "fused_linear_cross_entropy"
    for idx, (T, H, V, tag) in enumerate(S.FLCE_SHAPES):
        inp = torch.randn(T, H, device=DEV, dtype=DT)
        w = torch.randn(V, H, device=DEV, dtype=DT)
        tgt = torch.randint(0, V, (T,), device=DEV)
        ref = flce_ref_chunked(inp, w, tgt)
        out = LigerFusedLinearCrossEntropyFunction.apply(
            inp.clone(), w, tgt, None, None, -100, 0.0, 0.0, "none")[0]
        err, rel = rel_err(out, ref)
        flops = 2 * T * H * V + 5 * T * V          # GEMM + CE
        nbytes = 2 * T * H + 2 * V * H + 4 * T     # 输入+权重, logits 不物化
        m = bench_one(lambda: LigerFusedLinearCrossEntropyFunction.apply(
            inp, w, tgt, None, None, -100, 0.0, 0.0, "none"), flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"input=({T},{H}) x weight=({V},{H})",
               "dims": f"T={T}, H={H}, V={V}",
               "elements": T * V, "flops": flops, "bytes": nbytes,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 5e-2 else f"CHECK(rel={rel:.2e})",
               "source": LG}
        row.update(m)
        save_result(op + DTYPE_SUFFIX, row)



def bench_mm_fp8():
    """mm 的 MXFP8 版本: 与 mm 相同的 MM_SHAPES, 仅保留 K%128==0 且 N%128==0 的 shape"""
    op = "mm"
    n_run = 0
    for idx, (M, K, N, tag) in enumerate(S.MM_SHAPES):
        if K % 128 or N % 128:
            continue
        n_run += 1
        x = torch.randn(M, K, device=DEV)
        y = torch.randn(K, N, device=DEV)
        aq, asc = _quant_tokens(x)
        bq, bsc = _quant_blocks(y)
        xd = (aq.view(M, K // 128, 128).float() * asc.unsqueeze(-1)).view(M, K)
        yd = (bq.view(N // 128, 128, K // 128, 128).float()
              * bsc[:, None, :, None]).view(N, K).t()
        ref = xd @ yd
        out = flag_gems.fp8_matmul(aq, asc, bq, bsc)
        err, rel = rel_err(out, ref)
        flops = 2 * M * K * N
        nbytes = (M * K + N * K + 4 * (M * (K // 128) + (N // 128) * (K // 128))
                  + 2 * M * N)
        m = bench_one(lambda: flag_gems.fp8_matmul(aq, asc, bq, bsc), flops, nbytes)
        row = {"no": n_run, "tag": tag,
               "shape": f"({M},{K}) x ({K},{N}) fp8-blk128",
               "dims": f"M={M}, K={K}, N={N}, quant=MXFP8-128",
               "elements": M * N, "flops": flops, "bytes": nbytes,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 8e-2 else f"CHECK(rel={rel:.2e})",
               "source": FG}
        row.update(m)
        row["dtype"] = "fp8(e4m3)+scale"
        save_result(op + DTYPE_SUFFIX, row)


ALL = {
    "bmm": bench_bmm,
    "addmm": bench_addmm,
    "mv": bench_mv,
    "fp8_matmul": bench_fp8_matmul,
    "mm_fp8": bench_mm_fp8,
    "fused_linear_cross_entropy": bench_fused_linear_cross_entropy,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n100", action="store_true", help="使用 shapes_100 (每算子100 shape)")
    ap.add_argument("--dtype", default="", choices=["", "fp16", "fp32", "fp8"],
                    help="按 dtype 跑, 结果写入 results_dtypes/<op>_<dtype>.json")
    args = ap.parse_args()
    if args.dtype:
        import common, os as _os
        common.DEFAULT_DTYPE = args.dtype
        common.RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results_dtypes")
        globals()["DT"] = {"fp16": torch.float16, "fp32": torch.float32, "fp8": DT}[args.dtype]
        globals()["DTYPE_SUFFIX"] = "_" + args.dtype
    EXPECTED = 20
    if args.n100:
        import shapes_100
        S = shapes_100  # 重绑定: 各bench函数改用100-shape配置
        EXPECTED = 100
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    skip = set()
    if args.dtype in ("fp16", "fp32"):
        skip.add("fp8_matmul")     # fp8_matmul 本身即 fp8 算子
        skip.add("mm_fp8")
    if args.dtype != "fp8":
        skip.add("mm_fp8")
    if args.dtype == "fp8":
        only = ["mm_fp8"]          # fp8 模式只跑 mm 的 MXFP8 版
    for name, fn in ALL.items():
        if only and name not in only:
            continue
        if name in skip:
            print(f"[SKIP] {name} 不适用于 dtype={args.dtype or 'bf16'}", flush=True)
            continue
        if args.force:
            clear_result(name + DTYPE_SUFFIX)
        done = len(load_results(name + DTYPE_SUFFIX))
        if done >= EXPECTED:
            print(f"[SKIP] {name} 已有 {done}/{EXPECTED} 条结果", flush=True)
            continue
        if done > 0:
            clear_result(name + DTYPE_SUFFIX)
        print(f"\n===== BENCH {name} =====", flush=True)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"[ERROR] {name} 失败, 继续下一个", flush=True)
    print("\nALL DONE", flush=True)
