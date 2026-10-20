#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rope 输出模式探针: 判断 liger _ascend rope 输出符合哪种候选语义."""
import torch

import smoke_test as st

DEV = "npu:0"
torch.npu.set_device(0)
m = st.get_liger_op("rope")

S, D = 8, 8
inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
freqs = torch.outer(torch.arange(S).float(), inv)
cos, sin = freqs.cos()[None], freqs.sin()[None]
q = torch.randn(1, 1, S, D, dtype=torch.bfloat16)

qo = m.rope_forward(st.npu_dev(q.clone()), None,
                    st.npu_dev(cos).to(torch.bfloat16), st.npu_dev(sin).to(torch.bfloat16))[0]
torch.npu.synchronize()
out = qo.float().cpu()[0, 0]          # (S, D)
qf = q.float()[0, 0]
cb = cos.to(torch.bfloat16).float()   # (1,S,D/2) bf16 舍入
sb = sin.to(torch.bfloat16).float()


def neox(q_row, c, s):  # half-split, 半宽表
    l, r = q_row[: D // 2], q_row[D // 2:]
    return torch.cat([l * c - r * s, r * c + l * s])


cands = {
    "refA(neox,pos=s)": lambda s_: neox(qf[s_], cb[0, s_], sb[0, s_]),
    "shift+1": lambda s_: neox(qf[s_], cb[0, min(s_ + 1, S - 1)], sb[0, min(s_ + 1, S - 1)]),
    "pos=0 const": lambda s_: neox(qf[s_], cb[0, 0], sb[0, 0]),
    "no-rotate": lambda s_: qf[s_],
    "freqs^T (s<->d)": lambda s_: neox(qf[s_], cb[0, : D // 2].flip(0), sb[0, : D // 2].flip(0)),
}
print("row | out-vs-cand max abs diff")
for name, f in cands.items():
    diffs = [(out[s_] - f(s_)).abs().max().item() for s_ in range(S)]
    print(f"{name:>18}: " + " ".join(f"{d:7.4f}" for d in diffs))

print("\nout row1 :", out[1].tolist())
print("refA row1:", neox(qf[1], cb[0, 1], sb[0, 1]).tolist())
