#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rope 失败路径定位: 扫 (S, H, q/k) 组合, 对比 NeoX 参考."""
import torch

import smoke_test as st

DEV = "npu:0"
torch.npu.set_device(0)
m = st.get_liger_op("rope")


def run_case(B, Hq, Hkv, S, D, with_k):
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    freqs = torch.outer(torch.arange(S).float(), inv)
    cos = freqs.cos()[None]
    sin = freqs.sin()[None]
    q = torch.randn(B, Hq, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16) if with_k else None
    res = m.rope_forward(st.npu_dev(q.clone()), st.npu_dev(k.clone()) if with_k else None,
                         st.npu_dev(cos).to(torch.bfloat16), st.npu_dev(sin).to(torch.bfloat16))
    torch.npu.synchronize()
    cb = cos.to(torch.bfloat16).float()
    sb = sin.to(torch.bfloat16).float()
    maxd = 0.0
    for x, xo in ((q, res[0]), (k, res[1])) if with_k else ((q, res[0]),):
        xf, of = x.float(), xo.float().cpu()
        l, r = xf[..., : D // 2], xf[..., D // 2:]
        ref = torch.cat([l * cb - r * sb, r * cb + l * sb], -1)
        maxd = max(maxd, (of - ref).abs().max().item())
    return maxd


print(f"{'B':>2} {'Hq':>3} {'Hkv':>3} {'S':>4} {'D':>4} {'k?':>3}  max_abs_diff")
for (B, Hq, Hkv, S, D, wk) in [
    (1, 1, 1, 8, 8, True),
    (1, 2, 2, 8, 8, True),
    (1, 1, 1, 256, 128, True),
    (1, 2, 1, 256, 128, True),   # GQA
    (1, 8, 2, 256, 128, True),   # smoke 配置
    (1, 8, 2, 256, 128, False),  # 只 q
    (1, 1, 1, 64, 128, True),
    (1, 1, 1, 128, 128, True),
    (1, 4, 4, 256, 64, True),
]:
    d = run_case(B, Hq, Hkv, S, D, wk)
    flag = "  <-- FAIL" if d > 0.05 else ""
    print(f"{B:>2} {Hq:>3} {Hkv:>3} {S:>4} {D:>4} {str(wk):>3}  {d:.4f}{flag}")
