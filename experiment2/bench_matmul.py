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
        save_result(op, row)


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
        save_result(op, row)


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
        save_result(op, row)


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
        save_result(op, row)


def bench_fused_linear_cross_entropy():
    op = "fused_linear_cross_entropy"
    for idx, (T, H, V, tag) in enumerate(S.FLCE_SHAPES):
        inp = torch.randn(T, H, device=DEV, dtype=DT)
        w = torch.randn(V, H, device=DEV, dtype=DT)
        tgt = torch.randint(0, V, (T,), device=DEV)
        ref = F.cross_entropy(inp.float() @ w.float().t(), tgt, reduction="none")
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
        save_result(op, row)


ALL = {
    "bmm": bench_bmm,
    "addmm": bench_addmm,
    "mv": bench_mv,
    "fp8_matmul": bench_fp8_matmul,
    "fused_linear_cross_entropy": bench_fused_linear_cross_entropy,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    for name, fn in ALL.items():
        if only and name not in only:
            continue
        if args.force:
            clear_result(name)
        done = len(load_results(name))
        if done >= 20:
            print(f"[SKIP] {name} 已有 {done} 条结果", flush=True)
            continue
        if done > 0:
            clear_result(name)
        print(f"\n===== BENCH {name} =====", flush=True)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"[ERROR] {name} 失败, 继续下一个", flush=True)
    print("\nALL DONE", flush=True)
