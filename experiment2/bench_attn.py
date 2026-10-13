#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B200 Attention 算子基准测试 (python3.10 环境: vllm 0.24.0 + flash-attn 2.8.3)

1) unified_attention  - vLLM (vllm-project/vllm v0.24.0)
       vllm.v1.attention.ops.triton_unified_attention.unified_attention
       paged KV cache + varlen query, causal
2) flash_attention    - flash-attention (Dao-AILab/flash-attention v2.8.3)
       flash_attn.flash_attn_func, causal

运行: python3.10 bench_attn.py [--only unified_attention,flash_attention]
"""

import argparse
import traceback

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as vllm_unified_attention

from common import bench_one, save_result, clear_result, load_results, torch_version_str
import shapes as S

DEV = "cuda"
DT = torch.bfloat16
BLOCK = 128  # vLLM paged KV block size

SRC_VLLM = "vllm-project/vllm v0.24.0 (Triton, vllm.v1.attention.ops.triton_unified_attention)"
SRC_FA = "Dao-AILab/flash-attention v2.8.3 (CUDA, flash_attn_func)"

print(f"[ENV] {torch_version_str()}, GPU={torch.cuda.get_device_name(0)}, "
      f"cap={torch.cuda.get_device_capability(0)}", flush=True)


def sdpa_ref(q4, k4, v4, causal, Hq, Hkv):
    """q4: [B,Sq,Hq,D]; k4/v4: [B,Skv,Hkv,D] -> fp32 参考"""
    q = q4.transpose(1, 2).float()
    k = k4.transpose(1, 2).float()
    v = v4.transpose(1, 2).float()
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = q4.shape[-1] ** -0.5
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return o.transpose(1, 2)  # [B,Sq,Hq,D] fp32


def make_dense(B, Sq, Skv, Hq, Hkv, D):
    q4 = torch.randn(B, Sq, Hq, D, device=DEV, dtype=DT)
    k4 = torch.randn(B, Skv, Hkv, D, device=DEV, dtype=DT)
    v4 = torch.randn(B, Skv, Hkv, D, device=DEV, dtype=DT)
    return q4, k4, v4


def bench_flash_attention():
    op = "flash_attention"
    for idx, (B, Sq, Skv, Hq, Hkv, D, tag) in enumerate(S.ATTN_SHAPES):
        q4, k4, v4 = make_dense(B, Sq, Skv, Hq, Hkv, D)
        causal = True  # FA2 causal = bottom-right 对齐; Sq=1 时等价于 attend 全部 kv
        out = flash_attn_func(q4, k4, v4, causal=causal)
        # decode(Sq=1) 时 causal 语义 = 最后一个 token 看全部 kv → 参考用非 causal
        ref = sdpa_ref(q4, k4, v4, causal=(Sq > 1), Hq=Hq, Hkv=Hkv)
        err = (out.float() - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-6)
        flops, nbytes = S.attn_flops_bytes(B, Sq, Skv, Hq, Hkv, D, causal=True)
        m = bench_one(lambda: flash_attn_func(q4, k4, v4, causal=causal), flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"B={B}, Sq={Sq}, Skv={Skv}, Hq={Hq}, Hkv={Hkv}, D={D}",
               "dims": f"B={B}, Sq={Sq}, Skv={Skv}, Hq={Hq}, Hkv={Hkv}, D={D}",
               "elements": B * Sq * Hq * D, "flops": flops, "bytes": nbytes,
               "causal": "True",
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": SRC_FA}
        row.update(m)
        save_result(op, row)


def bench_unified_attention():
    op = "unified_attention"
    for idx, (B, Sq, Skv, Hq, Hkv, D, tag) in enumerate(S.ATTN_SHAPES):
        nb_per = (Skv + BLOCK - 1) // BLOCK
        total_blocks = B * nb_per
        # paged KV cache
        kc = torch.randn(total_blocks, BLOCK, Hkv, D, device=DEV, dtype=DT)
        vc = torch.randn(total_blocks, BLOCK, Hkv, D, device=DEV, dtype=DT)
        # packed varlen query: B 段, 每段 Sq 个 token
        q = torch.randn(B * Sq, Hq, D, device=DEV, dtype=DT)
        block_table = (torch.arange(B, device=DEV).view(B, 1) * nb_per
                       + torch.arange(nb_per, device=DEV).view(1, nb_per)).int()
        cu_seqlens_q = torch.zeros(B + 1, device=DEV, dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(torch.full((B,), Sq, device=DEV), 0)
        seqused_k = torch.full((B,), Skv, device=DEV, dtype=torch.int32)
        out = torch.empty(B * Sq, Hq, D, device=DEV, dtype=DT)
        sm_scale = D ** -0.5

        def fn():
            vllm_unified_attention(q, kc, vc, out, cu_seqlens_q, Sq, seqused_k, Skv,
                                   sm_scale, True, (-1, -1), block_table, 0.0,
                                   None, None, None)
        fn()
        torch.cuda.synchronize()
        # correctness: 逐 batch 取 cache 重建 dense k/v 与 sdpa 对比
        err = 0.0
        ref_scale = 1e-6
        for b in range(min(B, 2)):
            kb = kc[b * nb_per:(b + 1) * nb_per].reshape(-1, Hkv, D).unsqueeze(0)
            vb = vc[b * nb_per:(b + 1) * nb_per].reshape(-1, Hkv, D).unsqueeze(0)
            qb = q[b * Sq:(b + 1) * Sq].unsqueeze(0)  # [1,Sq,Hq,D]
            ref = sdpa_ref(qb, kb, vb, causal=(Sq > 1), Hq=Hq, Hkv=Hkv).reshape(Sq, Hq, D)
            got = out[b * Sq:(b + 1) * Sq]
            err = max(err, (got.float() - ref).abs().max().item())
            ref_scale = max(ref_scale, ref.abs().max().item())
        rel = err / ref_scale
        flops, nbytes = S.attn_flops_bytes(B, Sq, Skv, Hq, Hkv, D, causal=True)
        m = bench_one(fn, flops, nbytes)
        row = {"no": idx + 1, "tag": tag,
               "shape": f"B={B}, Sq={Sq}, Skv={Skv}, Hq={Hq}, Hkv={Hkv}, D={D}",
               "dims": f"B={B}, Sq={Sq}, Skv={Skv}, Hq={Hq}, Hkv={Hkv}, D={D}",
               "elements": B * Sq * Hq * D, "flops": flops, "bytes": nbytes,
               "causal": "True", "kv_block": BLOCK,
               "err_vs_ref": f"{err:.3e}", "err_rel": f"{rel:.3e}",
               "status": "OK" if rel <= 2e-2 else f"CHECK(rel={rel:.2e})",
               "source": SRC_VLLM}
        row.update(m)
        save_result(op, row)


ALL = {
    "unified_attention": bench_unified_attention,
    "flash_attention": bench_flash_attention,
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
