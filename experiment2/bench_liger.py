#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Liger-Kernel (linkedin/Liger-Kernel v0.8.4) 6 个算子 B200 基准测试
运行: /root/miniconda3/envs/kaixin/bin/python bench_liger.py [--only op1,op2]

算子: rms_norm, layer_norm, fused_add_rms_norm, swiglu, cross_entropy, rope(apply_rotary_pos_emb)
"""

import argparse
import traceback

import torch

from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.rope import LigerRopeFunction

from common import bench_one, save_result, clear_result, load_results, torch_version_str
import shapes as S

DEV = "cuda"
DT = torch.bfloat16
EPS = 1e-6
OP_SOURCE = "linkedin/Liger-Kernel v0.8.4 (Triton)"

print(f"[ENV] {torch_version_str()}, GPU={torch.cuda.get_device_name(0)}, "
      f"cap={torch.cuda.get_device_capability(0)}", flush=True)


def bench_rms_norm():
    op = "rms_norm"
    for idx, (M, N, tag) in enumerate(S.NORM_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        w = torch.randn(N, device=DEV, dtype=DT)
        # correctness (fp32)
        xf, wf = x.float(), w.float()
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * wf
        out = LigerRMSNormFunction.apply(x.clone(), w, EPS)
        err = (out.float() - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        m = bench_one(lambda: LigerRMSNormFunction.apply(x, w, EPS),
                      5 * M * N,                      # 平方+均值+rsqrt+2次乘 名义5 flops
                      2 * 2 * M * N + 2 * N)          # 读X+写O+读weight
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N}", "elements": M * N,
               "flops": 5 * M * N, "bytes": 2 * 2 * M * N + 2 * N,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result(op, row)


def bench_layer_norm():
    op = "layer_norm"
    for idx, (M, N, tag) in enumerate(S.NORM_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        w = torch.randn(N, device=DEV, dtype=DT)
        b = torch.randn(N, device=DEV, dtype=DT)
        xf = x.float()
        ref = torch.nn.functional.layer_norm(xf, (N,), w.float(), b.float(), EPS)
        out = LigerLayerNormFunction.apply(x.clone(), w, b, EPS)
        err = (out.float() - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        m = bench_one(lambda: LigerLayerNormFunction.apply(x, w, b, EPS),
                      8 * M * N,                      # 均值+方差+规范化+仿射 名义8 flops
                      2 * 2 * M * N + 2 * 2 * N)
        row = {"no": idx + 1, "tag": tag, "shape": f"({M}, {N})",
               "dims": f"M={M}, N={N}", "elements": M * N,
               "flops": 8 * M * N, "bytes": 2 * 2 * M * N + 4 * N,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result(op, row)


def bench_fused_add_rms_norm():
    op = "fused_add_rms_norm"
    for idx, (M, N, tag) in enumerate(S.NORM_SHAPES):
        x = torch.randn(M, N, device=DEV, dtype=DT)
        r = torch.randn(M, N, device=DEV, dtype=DT)
        w = torch.randn(N, device=DEV, dtype=DT)
        # correctness: 参考实现 (residual + rmsnorm)
        xf = (x.float() + r.float())
        ref_o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
        o1, o2 = LigerFusedAddRMSNormFunction.apply(x.clone(), r.clone(), w, EPS)
        err = max((o1.float() - ref_o).abs().max().item(),
                  (o2.float() - xf).abs().max().item())
        rel = err / (max(ref_o.abs().max().item(), xf.abs().max().item()) + 1e-6)
        m = bench_one(lambda: LigerFusedAddRMSNormFunction.apply(x, r, w, EPS),
                      6 * M * N,                      # add + rms_norm 名义6 flops
                      4 * 2 * M * N + 2 * N)          # 读X,R + 写O,R' + weight
        row = {"no": idx + 1, "tag": tag, "shape": f"X=({M}, {N}), R=({M}, {N})",
               "dims": f"M={M}, N={N}", "elements": M * N,
               "flops": 6 * M * N, "bytes": 4 * 2 * M * N + 2 * N,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result(op, row)


def bench_swiglu():
    op = "swiglu"
    for idx, (T, H, tag) in enumerate(S.SWIGLU_SHAPES):
        a = torch.randn(T, H, device=DEV, dtype=DT)   # gate
        b = torch.randn(T, H, device=DEV, dtype=DT)   # up
        ref = torch.nn.functional.silu(a.float()) * b.float()
        out = LigerSiLUMulFunction.apply(a.clone(), b.clone())
        err = (out.float() - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        m = bench_one(lambda: LigerSiLUMulFunction.apply(a, b),
                      5 * T * H,                      # silu(4) + mul(1)
                      3 * 2 * T * H)                  # 读a,b + 写out
        row = {"no": idx + 1, "tag": tag, "shape": f"a=({T}, {H}), b=({T}, {H})",
               "dims": f"T={T}, H={H}", "elements": T * H,
               "flops": 5 * T * H, "bytes": 3 * 2 * T * H,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result(op, row)


def ce_ref_chunked(logits, target, max_chunk_elems=1 << 26):
    """分块 fp32 CE 参考 (避免大 T*V 整体物化 fp32; 每 chunk ≤ 64M 元素 = 256MB)"""
    T, V = logits.shape
    chunk = max(1, max_chunk_elems // V)
    outs = []
    for i in range(0, T, chunk):
        lc = logits[i:i + chunk].float()
        t_c = target[i:i + chunk]
        outs.append(-lc.gather(1, t_c.view(-1, 1)).squeeze(1) + torch.logsumexp(lc, -1))
    return torch.cat(outs)


def bench_cross_entropy():
    op = "cross_entropy"
    for idx, (T, V, tag) in enumerate(S.CE_SHAPES):
        logits = torch.randn(T, V, device=DEV, dtype=DT)
        target = torch.randint(0, V, (T,), device=DEV)
        # correctness (fp32 分块参考)
        ref = ce_ref_chunked(logits, target)
        out = LigerCrossEntropyFunction.apply(logits.clone(), target, None, -100,
                                                0.0, 0.0, "none")[0]
        err = (out.float() - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        m = bench_one(
            lambda: LigerCrossEntropyFunction.apply(logits, target, None, -100,
                                                    0.0, 0.0, "none"),
            5 * T * V,                                # max+exp+sum+div+gather 名义5 flops
            2 * T * V + 4 * T)                        # 读logits + fp32 loss 输出
        row = {"no": idx + 1, "tag": tag, "shape": f"({T}, {V})",
               "dims": f"T={T}, V={V}", "elements": T * V,
               "flops": 5 * T * V, "bytes": 2 * T * V + 4 * T,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result(op, row)


def bench_rope():
    op = "rope"
    for idx, (B, Hq, Hkv, Sq, D, tag) in enumerate(S.ROPE_SHAPES):
        q = torch.randn(B, Hq, Sq, D, device=DEV, dtype=DT)
        k = torch.randn(B, Hkv, Sq, D, device=DEV, dtype=DT)
        cos = torch.randn(1, Sq, D // 2, device=DEV, dtype=DT)   # [1,S,D/2] (llama格式)
        sin = torch.randn(1, Sq, D // 2, device=DEV, dtype=DT)
        # correctness: 手写旋转 (b, h, s, d) 行内 d 前/后半旋转
        qf = q.float(); kf = k.float()
        cf = cos.float().view(1, 1, Sq, D // 2); sf = sin.float().view(1, 1, Sq, D // 2)
        def rot(t):
            t1, t2 = t[..., : D // 2], t[..., D // 2:]
            return torch.cat([t1 * cf - t2 * sf, t2 * cf + t1 * sf], dim=-1)
        refq, refk = rot(qf), rot(kf)
        oq, ok = LigerRopeFunction.apply(q.clone(), k.clone(), cos, sin)
        err = max((oq.float() - refq).abs().max().item(),
                  (ok.float() - refk).abs().max().item())
        rel = err / (max(refq.abs().max().item(), refk.abs().max().item()) + 1e-6)
        m = bench_one(lambda: LigerRopeFunction.apply(q, k, cos, sin),
                      5 * B * Sq * D * (Hq + Hkv),     # 每2元素6次乘加≈每元素3 flops, 名义5
                      2 * 2 * (2 * B * Sq * Hq * D + 2 * B * Sq * Hkv * D))
        row = {"no": idx + 1, "tag": tag,
               "shape": f"q=({B},{Hq},{Sq},{D}), k=({B},{Hkv},{Sq},{D})",
               "dims": f"B={B}, Hq={Hq}, Hkv={Hkv}, S={Sq}, D={D}",
               "elements": B * Sq * D * (Hq + Hkv),
               "flops": 5 * B * Sq * D * (Hq + Hkv),
               "bytes": 2 * 2 * (2 * B * Sq * Hq * D + 2 * B * Sq * Hkv * D),
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": OP_SOURCE}
        row.update(m)
        save_result(op, row)


ALL = {
    "rms_norm": bench_rms_norm,
    "layer_norm": bench_layer_norm,
    "fused_add_rms_norm": bench_fused_add_rms_norm,
    "swiglu": bench_swiglu,
    "cross_entropy": bench_cross_entropy,
    "rope": bench_rope,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
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
            clear_result(name)
        print(f"\n===== BENCH {name} =====", flush=True)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"[ERROR] {name} 失败, 继续下一个", flush=True)
    print("\nALL DONE", flush=True)
