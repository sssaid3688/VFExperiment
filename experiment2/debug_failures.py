#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""针对冒烟失败项的定向诊断 (只读, 不改任何环境/上游代码)."""

import smoke_test as st  # 复用 sys.path / 桩 / 惰性导入
import torch
import torch.nn.functional as F

DEV = "npu:0"
torch.npu.set_device(0)


def dbg_sum():
    print("\n========== #11 sum ==========")
    g = st.get_flag_gems()
    for shape, dt in [((4, 128), torch.float32), ((512, 4096), torch.float32),
                      ((512, 4096), torch.bfloat16)]:
        x = torch.randn(*shape, dtype=dt)
        try:
            out = g.sum_dim(st.npu_dev(x), dim=[-1])
            torch.npu.synchronize()
            ref = x.float().sum(-1)
            d = (out.float().cpu() - ref).abs()
            print(f"  {shape} {dt}: out[:4]={out.float().flatten()[:4].tolist()} "
                  f"ref[:4]={ref.flatten()[:4].tolist()} maxabs={d.max():.3e}")
        except Exception as e:
            print(f"  {shape} {dt}: EXC {type(e).__name__}: {e}")


def dbg_rope():
    print("\n========== #16 rope ==========")
    m = st.get_liger_op("rope")
    S, D = 8, 8
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    freqs = torch.outer(torch.arange(S).float(), inv)
    cos, sin = freqs.cos()[None], freqs.sin()[None]  # (1,S,D/2)
    q = torch.randn(1, 1, S, D, dtype=torch.bfloat16)

    qo = m.rope_forward(st.npu_dev(q.clone()), None, st.npu_dev(cos).to(torch.bfloat16),
                        st.npu_dev(sin).to(torch.bfloat16))[0]
    torch.npu.synchronize()
    qo = qo.float().cpu()

    # ref A: half-split (NeoX) 全宽等价
    cf = cos.to(torch.bfloat16).float().repeat_interleave(2, -1)
    sf = sin.to(torch.bfloat16).float().repeat_interleave(2, -1)
    refA = (q.float() * cf + st.rotate_half(q.float()) * sf)[0, 0]
    # ref B: interleaved (GPT-J): 成对 (2i, 2i+1) 旋转
    qf = q.float()[0, 0]
    cos_i = cos[0].repeat_interleave(2, -1)
    sin_i = sin[0].repeat_interleave(2, -1)
    refB = qf * cos_i + st.rotate_half(
        torch.cat((-qf.chunk(2, -1)[1], qf.chunk(2, -1)[0]), -1) * -1
    ) * sin_i  # placeholder, 重新精确构造见下
    # 精确 GPT-J: out[2i]=q[2i]*c[i]-q[2i+1]*s[i]; out[2i+1]=q[2i+1]*c[i]+q[2i]*s[i]
    even, odd = qf[..., 0::2], qf[..., 1::2]
    refB = torch.stack((even * cos[0] - odd * sin[0], odd * cos[0] + even * sin[0]), -1).flatten(-2)

    print("  q[0,:8] :", qf[0, :8].tolist())
    print("  out[:8] :", qo[0, 0, :8].tolist())
    print("  refA[:8]:", refA[0, :8].tolist())
    print("  refB[:8]:", refB[0, :8].tolist())
    print("  |out-refA|max =", (qo[0, 0] - refA).abs().max().item())
    print("  |out-refB|max =", (qo[0, 0] - refB).abs().max().item())


def dbg_bmm():
    print("\n========== #21 bmm ==========")
    import flag_gems
    import sys as _sys

    g = st.get_flag_gems()
    # flag_gems.ops 命名空间里 bmm 属性被 from-import 的函数遮蔽, 从 sys.modules 取真实模块
    bmm_mod = _sys.modules["flag_gems.ops.bmm"]
    print("  flag_gems.bmm is overridden by _ascend:",
          flag_gems.bmm is not bmm_mod.__dict__["bmm"])
    for B, M, K, N in [(2, 64, 64, 64), (16, 256, 512, 128), (4, 128, 512, 64)]:
        a = torch.randn(B, M, K, dtype=torch.bfloat16)
        b = torch.randn(B, K, N, dtype=torch.bfloat16)
        ref = torch.bmm(a.float(), b.float())
        for name, fn in [("global(_ascend)", flag_gems.bmm), ("orig module", bmm_mod.__dict__["bmm"])]:
            try:
                out = fn(st.npu_dev(a), st.npu_dev(b))
                torch.npu.synchronize()
                d = (out.float().cpu() - ref).abs()
                print(f"  ({B},{M},{K})x({B},{K},{N}) {name}: maxabs={d.max():.3e} "
                      f"nan={torch.isnan(out.float().cpu()).any().item()}")
            except Exception as e:
                print(f"  ({B},{M},{K}) {name}: EXC {type(e).__name__}: {str(e)[:150]}")


def dbg_fa2():
    print("\n========== #02 tutorial FA2 ==========")
    try:
        mod = st.get_tutorial_fa2()
        Z, H, N, D = 1, 2, 64, 64
        q = torch.randn(Z, H, N, D, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        o = mod._attention.apply(st.npu_dev(q), st.npu_dev(k), st.npu_dev(v), True, D**-0.5)
        torch.npu.synchronize()
        print("  import+run OK, out shape:", tuple(o.shape),
              "nan:", torch.isnan(o.float().cpu()).any().item())
    except Exception:
        import traceback

        print("  EXC:", traceback.format_exc()[-800:])


if __name__ == "__main__":
    dbg_sum()
    dbg_rope()
    dbg_bmm()
    dbg_fa2()
