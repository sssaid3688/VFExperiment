#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlagGems (FlagOpen/FlagGems v5.0.2) 12 个算子 B200 基准测试
运行: /root/miniconda3/envs/kaixin/bin/python bench_flaggems.py [--only op1,op2]

算子: softmax, silu, gelu, add, mul, clamp, sum, argmax, mm, cumsum, topk, embedding
"""

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

import flag_gems
from common import bench_one, save_result, clear_result, load_results, torch_version_str
import shapes as S

DEV = "cuda"
DT = torch.bfloat16
OP_SOURCE = "FlagOpen/FlagGems v5.0.2 (Triton)"

print(f"[ENV] {torch_version_str()}, GPU={torch.cuda.get_device_name(0)}, "
      f"cap={torch.cuda.get_device_capability(0)}", flush=True)


def run_2d_op(op_name, fn_make, ref_fn, flops_fn, bytes_fn, shape_list, dims_desc,
              op_module_fn, status_rtol=2e-2):
    """通用 2D (M,N) 算子 runner"""
    for idx, cfg in enumerate(shape_list):
        M, N, tag = cfg
        x = torch.randn(M, N, device=DEV, dtype=DT)
        fn = fn_make(x)
        flops, nbytes = flops_fn(M, N), bytes_fn(M, N)
        # correctness: 相对误差 (bf16 输出舍入 ~0.4%)
        try:
            with torch.no_grad():
                out = op_module_fn(x)
                ref = ref_fn(x)
                err = (out.float() - ref.float()).abs().max().item()
                rel = err / (ref.float().abs().max().item() + 1e-6)
            status = "OK" if rel <= status_rtol else f"CHECK(rel={rel:.2e})"
        except Exception:
            err, rel, status = float("nan"), float("nan"), "N/A"
        m = bench_one(fn, flops, nbytes)
        row = {
            "no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
            dims_desc: f"M={M}, N={N}",
            "elements": M * N, "flops": flops, "bytes": nbytes,
            "err_vs_ref": f"{err:.3e}" if err == err else "",
            "err_rel": f"{rel:.3e}" if rel == rel else "",
            "status": status,
            "source": OP_SOURCE,
        }
        row.update(m)
        save_result(op_name, row)


def bench_softmax():
    def ref(x):
        return torch.softmax(x.float(), dim=-1).to(DT)
    run_2d_op(
        "softmax",
        lambda x: (lambda: flag_gems.softmax(x, dim=-1)),
        ref, lambda M, N: 4 * M * N,          # sub+exp+add+div 每元素 4 flops(名义)
        lambda M, N: 2 * 2 * M * N,            # 读入+写出
        S.SOFTMAX_SHAPES, "dims",
        lambda x: flag_gems.softmax(x, dim=-1),
    )


def bench_silu():
    run_2d_op(
        "silu",
        lambda x: (lambda: flag_gems.silu(x)),
        lambda x: F.silu(x.float()).to(DT),
        lambda M, N: 4 * M * N,                # sigmoid(≈3)+mul 名义 4 flops
        lambda M, N: 2 * 2 * M * N,
        S.ELEM_SHAPES, "dims",
        lambda x: flag_gems.silu(x),
    )


def bench_gelu():
    run_2d_op(
        "gelu",
        lambda x: (lambda: flag_gems.gelu(x)),
        lambda x: F.gelu(x.float()).to(DT),
        lambda M, N: 10 * M * N,               # erf 版 gelu 名义 ~10 flops
        lambda M, N: 2 * 2 * M * N,
        S.ELEM_SHAPES, "dims",
        lambda x: flag_gems.gelu(x),
    )


def bench_clamp():
    run_2d_op(
        "clamp",
        lambda x: (lambda: flag_gems.clamp(x, -1.0, 1.0)),
        lambda x: torch.clamp(x, -1.0, 1.0),
        lambda M, N: 2 * M * N,                # 两次比较
        lambda M, N: 2 * 2 * M * N,
        S.ELEM_SHAPES, "dims",
        lambda x: flag_gems.clamp(x, -1.0, 1.0),
        status_rtol=0.0,
    )


def _rel_status(err, ref):
    rel = err / (ref.abs().max().item() + 1e-6)
    return rel, ("OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})")


def bench_add():
    for idx, (M, N, tag) in enumerate(S.ELEM_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        y = torch.randn(M, N, device=DEV, dtype=DT)
        ref = x.float() + y.float()
        err = (flag_gems.add(x, y).float() - ref).abs().max().item()
        rel, status = _rel_status(err, ref)
        m = bench_one(lambda: flag_gems.add(x, y), M * N, 3 * 2 * M * N)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N}", "elements": M * N, "flops": M * N,
               "bytes": 3 * 2 * M * N,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}", "status": status,
               "source": OP_SOURCE}
        row.update(m)
        save_result("add", row)


def bench_mul():
    for idx, (M, N, tag) in enumerate(S.ELEM_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        y = torch.randn(M, N, device=DEV, dtype=DT)
        ref = x.float() * y.float()
        err = (flag_gems.mul(x, y).float() - ref).abs().max().item()
        rel, status = _rel_status(err, ref)
        m = bench_one(lambda: flag_gems.mul(x, y), M * N, 3 * 2 * M * N)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N}", "elements": M * N, "flops": M * N,
               "bytes": 3 * 2 * M * N,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}", "status": status,
               "source": OP_SOURCE}
        row.update(m)
        save_result("mul", row)


def bench_sum():
    for idx, (M, N, tag) in enumerate(S.SOFTMAX_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=torch.float32)
        # 正确性: fp32 (bf16 求和存在累加顺序差异)
        err = (flag_gems.sum_dim(x, [-1], True).float() - x.sum(-1, keepdim=True)).abs().max().item()
        xb = torch.randn(M, N, device=DEV, dtype=DT)
        m = bench_one(lambda: flag_gems.sum_dim(xb, [-1], True), M * N, 2 * M * N + M * 2)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N} (dim=-1)", "elements": M * N, "flops": M * N,
               "bytes": 2 * M * N + 2 * M,
               "err_vs_ref_fp32": f"{err:.3e}", "status": "OK" if err <= 1e-3 else "CHECK",
               "source": OP_SOURCE}
        row.update(m)
        save_result("sum", row)


def bench_argmax():
    for idx, (M, N, tag) in enumerate(S.SOFTMAX_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        match = (flag_gems.argmax(x, dim=-1) == x.argmax(dim=-1)).float().mean().item()
        m = bench_one(lambda: flag_gems.argmax(x, dim=-1), M * N, 2 * M * N + 8 * M)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N} (dim=-1)", "elements": M * N, "flops": M * N,
               "bytes": 2 * M * N + 8 * M,
               "err_vs_ref": "", "status": "OK" if match == 1.0 else f"match={match:.4f}",
               "source": OP_SOURCE}
        row.update(m)
        save_result("argmax", row)


def bench_mm():
    for idx, (M, K, N, tag) in enumerate(S.MM_SHAPES):
        a = torch.randn(M, K, device=DEV, dtype=DT)
        b = torch.randn(K, N, device=DEV, dtype=DT)
        out = flag_gems.mm(a, b)
        ref = a.float() @ b.float()
        scale = ref.abs().max().item() + 1e-6
        err = (out.float() - ref).abs().max().item() / scale
        m = bench_one(lambda: flag_gems.mm(a, b), 2 * M * K * N,
                      2 * (M * K + K * N + M * N))
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {K}) x ({K}, {N})",
               "dims": f"M={M}, K={K}, N={N}", "elements": M * N,
               "flops": 2 * M * K * N, "bytes": 2 * (M * K + K * N + M * N),
               "err_vs_ref": f"{err:.3e}", "status": "OK" if err <= 5e-2 else "CHECK",
               "source": OP_SOURCE}
        row.update(m)
        save_result("mm", row)


def bench_cumsum():
    for idx, (M, N, tag) in enumerate(S.SOFTMAX_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=torch.float32)
        ref = x.cumsum(-1)
        err = (flag_gems.cumsum(x, -1) - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        xb = torch.randn(M, N, device=DEV, dtype=DT)
        m = bench_one(lambda: flag_gems.cumsum(xb, -1), M * N, 2 * 2 * M * N)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N} (dim=-1)", "elements": M * N, "flops": M * N,
               "bytes": 2 * 2 * M * N,
               "err_vs_ref_fp32": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 5e-4 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result("cumsum", row)


def bench_topk():
    for idx, (M, N, k, tag) in enumerate(S.TOPK_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        v1, _ = flag_gems.topk(x, k, dim=-1)
        v2, _ = torch.topk(x.float(), k, dim=-1)
        err = (v1.float() - v2).abs().max().item()
        m = bench_one(lambda: flag_gems.topk(x, k, dim=-1),
                      2 * k * M * N,                       # 名义: 每元素 k 次比较+交换
                      2 * M * N + (2 + 4) * k * M)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N}), k={k}",
               "dims": f"M={M}, N={N}, k={k}", "elements": M * N,
               "flops": 2 * k * M * N, "bytes": 2 * M * N + 6 * k * M,
               "err_vs_ref": f"{err:.3e}", "status": "OK" if err <= 2e-2 else "CHECK",
               "source": OP_SOURCE}
        row.update(m)
        save_result("topk", row)


def bench_embedding():
    for idx, (T, V, H, tag) in enumerate(S.EMB_SHAPES):
        table = torch.randn(V, H, device=DEV, dtype=DT)
        idx_t = torch.randint(0, V, (T,), device=DEV)
        out = flag_gems.embedding(table, idx_t)
        err = (out.float() - table[idx_t].float()).abs().max().item()
        m = bench_one(lambda: flag_gems.embedding(table, idx_t),
                      0, 8 * T + 2 * T * H + 2 * T * H)
        row = {"no": idx + 1, "tag": tag, "shape": f"idx=({T},) table=({V}, {H})",
               "dims": f"T={T}, V={V}, H={H}", "elements": T * H,
               "flops": 0, "bytes": 8 * T + 2 * T * H + 2 * T * H,
               "err_vs_ref": f"{err:.3e}", "status": "OK" if err == 0.0 else "CHECK",
               "source": OP_SOURCE}
        row.update(m)
        save_result("embedding", row)


ALL = {
    "softmax": bench_softmax, "silu": bench_silu, "gelu": bench_gelu,
    "add": bench_add, "mul": bench_mul, "clamp": bench_clamp,
    "sum": bench_sum, "argmax": bench_argmax, "mm": bench_mm,
    "cumsum": bench_cumsum, "topk": bench_topk, "embedding": bench_embedding,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的算子名")
    ap.add_argument("--force", action="store_true", help="清空已有结果重跑")
    ap.add_argument("--n100", action="store_true", help="使用 shapes_100 (每算子100 shape)")
    args = ap.parse_args()
    EXPECTED = 20
    if args.n100:
        import shapes_100
        S = shapes_100  # 重绑定: 各bench函数改用100-shape配置
        EXPECTED = 100
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    for name, fn in ALL.items():
        if only and name not in only:
            continue
        if args.force:
            clear_result(name)
        done = len(load_results(name))
        if done >= EXPECTED:
            print(f"[SKIP] {name} 已有 {done}/{EXPECTED} 条结果", flush=True)
            continue
        if done > 0:
            clear_result(name)  # 部分结果 → 重跑保证完整
        print(f"\n===== BENCH {name} =====", flush=True)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"[ERROR] {name} 失败, 继续下一个", flush=True)
    print("\nALL DONE", flush=True)
