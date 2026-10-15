# -*- coding: utf-8 -*-
"""
每算子 100 个 shape 的生成器 (--n100 模式使用)

设计目标:
- 覆盖 XS/S/M/L/XL 全尺寸阶梯 + 狭长形态(单行超宽/超高窄列/GEMV/超长序列/decode)
- 包含真实模型尺寸 (Llama/GPT/Qwen hidden/vocab/FFN, attention头配置)
- 尊重各算子约束 (fp8: K,N %128; rope: 规避 B>1&S=1; attention: Sq<=Skv, D in {64,128,256})
- 固定随机种子, 可复现; 每算子恰好 100 个唯一 shape
"""

import random

from shapes import attn_flops_bytes  # noqa: F401  (re-export)

RNG = random.Random(20261014)

# ------------------------- 公共阶梯 -------------------------
HIDDEN = [64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 11008, 12288, 16384]
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
          16384, 32768, 65536, 131072, 262144, 524288, 1048576]
VOCABS = [1000, 4096, 8192, 16000, 32000, 50257, 65000, 100000, 128256, 152064, 250000]
DIMS = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 8192, 11008, 16384]


def _tag_e(e):
    if e <= 1 << 16:
        return "XS"
    if e <= 1 << 20:
        return "S"
    if e <= 1 << 24:
        return "M"
    if e <= 1 << 28:
        return "L"
    return "XL"


def _tag_f(f):
    if f <= 1 << 28:
        return "XS"
    if f <= 1 << 32:
        return "S"
    if f <= 1 << 36:
        return "M"
    if f <= 1 << 40:
        return "L"
    return "XL"


def tag2d(M, N):
    e = M * N
    if M == 1:
        return "narrow-1row"
    if N == 1:
        return "narrow-1col"
    if N <= 128 and M >= 65536:
        return "narrow-tall"
    if M / max(N, 1) >= 32:
        return "tall-thin"
    if N / max(M, 1) >= 32:
        return "wide"
    return _tag_e(e)


def _dedupe(pool):
    seen, out = set(), []
    for item in pool:
        key = item[:-1] if isinstance(item[-1], str) else item
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _fill_to(pool, n, gen, tries=20000):
    """用 gen() 随机补齐到 n 个 (gen 返回含tag的tuple, 不合法返回None)"""
    pool = _dedupe(pool)
    t = 0
    while len(pool) < n and t < tries:
        t += 1
        item = gen()
        if item is None:
            continue
        key = item[:-1]
        if key not in {p[:-1] for p in pool}:
            pool.append(item)
    assert len(pool) >= n, f"仅生成 {len(pool)}/{n}"
    return pool[:n]


# ------------------------- 通用 2D (elementwise / softmax / norm / reduce) -------------------------
def build_2d(n, n_ladder, max_elems, row_max, tall_ns=(64, 128, 256)):
    """构造 (M, N, tag) 列表"""
    pool = []
    # 1) 固定 N=4096 扫 M (真实hidden)
    for M in [1, 16, 128, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]:
        if M * 4096 <= max_elems:
            pool.append((M, 4096, tag2d(M, 4096)))
    # 2) 固定 M=4096 扫 N
    for N in n_ladder:
        if 4096 * N <= max_elems:
            pool.append((4096, N, tag2d(4096, N)))
    # 3) 方阵
    for t in [128, 1024, 4096, 8192, 16384]:
        if t * t <= max_elems:
            pool.append((t, t, tag2d(t, t)))
    # 4) 单行超宽 (狭长)
    for N in [1024, 4096, 16384, 65536, 262144, row_max]:
        if N <= row_max:
            pool.append((1, N, "narrow-1row"))
    # 5) 超高窄列 (狭长)
    for Nn in tall_ns:
        for M in [65536, 262144, 1048576]:
            if M * Nn <= max_elems:
                pool.append((M, Nn, tag2d(M, Nn)))
    # 6) 真实模型尺寸交叉
    for (M, N) in [(2048, 11008), (4096, 11008), (2048, 12288), (4096, 5120),
                   (8192, 4096), (16384, 4096), (32768, 4096), (65536, 8192),
                   (1024, 128256 if 128256 <= row_max else 32768),
                   (131072, 2048), (8, 4096), (2, 8192), (32, 16384)]:
        if M * N <= max_elems and N <= row_max:
            pool.append((M, N, tag2d(M, N)))
    # 7) 随机补齐
    Ms = [m for m in TOKENS if m <= 1048576]
    def gen():
        M = RNG.choice(Ms)
        N = RNG.choice(n_ladder + [row_max])
        if N > row_max or M * N > max_elems or M * N < 16:
            return None
        return (M, N, tag2d(M, N))
    return _fill_to(pool, n, gen)


ELEM_SHAPES = build_2d(100, HIDDEN, max_elems=1 << 29, row_max=1 << 21,
                       tall_ns=(64, 128, 256))          # silu/gelu/clamp
SOFTMAX_SHAPES = build_2d(100, [n for n in HIDDEN if n <= 32768] + [32768],
                          max_elems=1 << 28, row_max=1 << 18,
                          tall_ns=(64, 128, 256))       # softmax/sum/argmax/cumsum
NORM_SHAPES = build_2d(100, [n for n in HIDDEN if n <= 16384],
                       max_elems=1 << 28, row_max=16384,
                       tall_ns=(64, 128, 256))          # rms/layer/fused_add_rms

# ------------------------- mm / addmm -------------------------
def tagmm(M, K, N):
    f = 2 * M * K * N
    if M == 1:
        return "narrow-GEMV"
    if M <= 8:
        return "narrow-Msmall"
    if K >= 32768:
        return "thin-K"
    if M >= 65536 and N <= 256:
        return "tall-thin"
    if N >= 32768:
        return "wide-N"
    return _tag_f(f)


def build_mm(n, fp8=False):
    pool = []
    if fp8:
        base = [128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 11008, 12288, 16384]
        Ms = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
              65536, 131072]
    else:
        base = DIMS
        Ms = DIMS + [32768, 65536, 131072]
    # 方阵
    for d in [b for b in base if b >= 128]:
        pool.append((d, d, d, tagmm(d, d, d)))
    # 固定 4096 扫 N / M / K
    for Nn in base:
        if Nn >= 128:
            pool.append((4096, 4096, Nn, tagmm(4096, 4096, Nn)))
    for M in Ms:
        pool.append((M, 4096, 4096, tagmm(M, 4096, 4096)))
    for K in base:
        pool.append((4096, K, 4096, tagmm(4096, K, 4096)))
    # GEMV 与狭长
    for (M, K, N) in [(1, 4096, 4096), (1, 128, 16384), (1, 16384, 128),
                      (1, 131072, 128 if fp8 else 64), (1, 64 if not fp8 else 128, 131072),
                      (131072, 128, 128), (65536, 128, 4096), (128, 131072, 128),
                      (32768, 128, 128), (128, 32768, 128), (128, 4096, 65536),
                      (64, 4096, 65536), (8, 16384, 16384), (16, 8192, 8192),
                      (2, 65536, 128 if fp8 else 64)]:
        pool.append((M, K, N, tagmm(M, K, N)))
    # 真实模型
    for (M, K, N) in [(4096, 4096, 11008), (8192, 4096, 4096), (4096, 11008, 4096),
                      (4096, 5120, 5120), (2048, 2048, 12288), (8192, 8192, 8192),
                      (16384, 16384, 16384), (512, 512, 11008), (1024, 1024, 4096)]:
        pool.append((M, K, N, tagmm(M, K, N)))

    def cap(M, K, N):
        return (2 * M * K * N <= 35e12 and M * K + K * N + M * N <= 2.5e9
                and M <= 131072 and K <= 131072 and N <= 65536)

    def gen():
        M = RNG.choice(Ms)
        K = RNG.choice(base)
        N = RNG.choice(base + [32768, 65536])
        if not cap(M, K, N) or (fp8 and (K % 128 or N % 128)):
            return None
        return (M, K, N, tagmm(M, K, N))
    return _fill_to(pool, n, gen)


MM_SHAPES = build_mm(100)
FP8_MM_SHAPES = build_mm(100, fp8=True)

# ------------------------- bmm -------------------------
def tagbmm(B, M, K, N):
    f = 2 * B * M * K * N
    if B >= 1024:
        return "narrow-batch"
    if B == 1 and M == 1:
        return "narrow-dot"
    if M <= 128 and N <= 128 and K <= 128:
        return "tiny-mat"
    return _tag_f(f)


def build_bmm(n):
    pool = []
    # 方阵批量
    for B in [1, 2, 4, 8, 16, 32]:
        for m in [512, 1024, 2048]:
            pool.append((B, m, m, m, tagbmm(B, m, m, m)))
    # 注意力 QK^T / PV 式
    for S in [256, 1024, 4096, 8192]:
        pool.append((32, S, 128, S, "attnQK"))
        pool.append((32, S, S, 128, "attnPV"))
    for S in [2048, 8192]:
        pool.append((16, S, 64, S, "attnQK-D64"))
    # 海量微矩阵 (狭长)
    for B in [256, 1024, 4096, 16384]:
        pool.append((B, 8, 8, 8, "narrow-batch"))
    for B in [128, 512, 2048]:
        pool.append((B, 32, 32, 32, "narrow-batch"))
    # 退化/超高窄
    pool += [(1, 1024, 1024, 1024, tagbmm(1, 1024, 1024, 1024)),
             (1, 1, 4096, 4096, "narrow-dot"),
             (1, 131072, 64, 64, "narrow-tall"),
             (2, 65536, 128, 128, "L-tall"),
             (1, 8192, 8192, 8192, tagbmm(1, 8192, 8192, 8192)),
             (16, 8192, 8192, 8192, "XL"),
             (64, 1024, 64, 1024, "b64"), (128, 64, 64, 64, "b128"),
             (256, 32, 32, 32, "b256"),
             (8, 1024, 1024, 1024, tagbmm(8, 1024, 1024, 1024))]

    def cap(B, M, K, N):
        return (2 * B * M * K * N <= 20e12
                and B * (M * K + K * N + M * N) <= 2e9
                and B <= 16384 and M <= 131072 and K <= 65536 and N <= 65536)

    def gen():
        B = RNG.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384])
        M = RNG.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536])
        K = RNG.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
        N = RNG.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
        if not cap(B, M, K, N):
            return None
        return (B, M, K, N, tagbmm(B, M, K, N))
    return _fill_to(pool, n, gen)


BMM_SHAPES = build_bmm(100)

# ------------------------- mv -------------------------
def build_mv(n):
    pool = []
    for K in [128, 1024, 4096, 11008, 16384]:
        pool.append((4096, K, tag2d(4096, K)))
    for M in [1024, 4096, 11008, 8192, 16384, 5120]:
        pool.append((M, 4096, tag2d(M, 4096)))
    pool += [(1, 4096, "narrow-dot"), (1, 1048576, "narrow-longK"),
             (4096, 1, "narrow-K1"), (1048576, 1, "narrow-M1M"),
             (131072, 64, "narrow-tallM"), (64, 131072, "narrow-tallK"),
             (65536, 64, "tallM"), (32768, 128, "tallM2"),
             (1024, 128, "XS-thinK"), (128, 1024, "XS"),
             (16384, 16384, "XL"), (12288, 12288, "L"), (2048, 2048, "M"),
             (11008, 4096, "M-llamaFFN"), (4096, 11008, "M-llamaDown")]

    def cap(M, K):
        return M * K <= 1 << 28 and M <= 1048576 and K <= 1048576

    def gen():
        M = RNG.choice([1, 64, 128, 1024, 2048, 4096, 5120, 8192, 11008, 12288,
                        16384, 32768, 65536, 131072])
        K = RNG.choice([1, 64, 128, 1024, 2048, 4096, 5120, 8192, 11008, 12288,
                        16384, 32768, 65536, 131072, 1048576])
        if not cap(M, K):
            return None
        return (M, K, tag2d(M, K))
    return _fill_to(pool, n, gen)


MV_SHAPES = build_mv(100)

# ------------------------- attention -------------------------
def tagattn(B, Sq, Skv, Hq, Hkv, D):
    f, _ = attn_flops_bytes(B, Sq, Skv, Hq, Hkv, D, causal=True)
    if Sq == 1:
        return "decode" + ("-longctx" if Skv >= 32768 else "")
    if Skv >= 32768:
        return "longctx"
    if Sq < Skv:
        return "spec-chunk"
    if Hq == Hkv:
        return "MHA"
    if Hq // Hkv >= 8:
        return "GQA8"
    return _tag_f(f)


def build_attn(n):
    pool = []
    # 单序列 prefill 阶梯
    for S in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        pool.append((1, S, S, 32, 8, 128, tagattn(1, S, S, 32, 8, 128)))
    # batch prefill
    for B in [2, 4, 8, 16, 32, 64]:
        pool.append((B, 1024, 1024, 32, 8, 128, tagattn(B, 1024, 1024, 32, 8, 128)))
    for B in [2, 4, 8]:
        pool.append((B, 4096, 4096, 32, 8, 128, tagattn(B, 4096, 4096, 32, 8, 128)))
        pool.append((B, 2048, 2048, 16, 16, 64, tagattn(B, 2048, 2048, 16, 16, 64)))
    # 头维/头数变化
    for D in [64, 128, 256]:
        pool.append((2, 4096, 4096, 32, 8, D, tagattn(2, 4096, 4096, 32, 8, D)))
    for (Hq, Hkv) in [(16, 16), (32, 32), (64, 8), (128, 16), (8, 8), (4, 4), (32, 4)]:
        pool.append((1, 4096, 4096, Hq, Hkv, 128, tagattn(1, 4096, 4096, Hq, Hkv, 128)))
    pool += [(1, 32768, 32768, 8, 8, 64, tagattn(1, 32768, 32768, 8, 8, 64)),
             (1, 131072, 131072, 16, 4, 64, "XL-128K"),
             (1, 1024, 1024, 64, 8, 64, tagattn(1, 1024, 1024, 64, 8, 64)),
             (1, 16384, 16384, 32, 8, 128, tagattn(1, 16384, 16384, 32, 8, 128))]
    # decode 阶梯 (B 大则 ctx 小, 控KV内存)
    for (B, C) in [(1, 512), (1, 32768), (1, 131072), (2, 1024), (8, 1024),
                   (16, 2048), (32, 1024), (64, 4096), (128, 1024), (256, 512),
                   (512, 512), (1024, 128)]:
        pool.append((B, 1, C, 32, 8, 128, tagattn(B, 1, C, 32, 8, 128)))
    # chunked/spec 式 (Sq < Skv)
    for (B, Sq, Skv) in [(1, 1024, 4096), (1, 2048, 16384), (4, 512, 2048),
                         (16, 256, 1024), (8, 1024, 8192), (1, 4096, 32768)]:
        pool.append((B, Sq, Skv, 32, 8, 128, tagattn(B, Sq, Skv, 32, 8, 128)))

    def cap(B, Sq, Skv, Hq, Hkv, D):
        f, _ = attn_flops_bytes(B, Sq, Skv, Hq, Hkv, D, causal=True)
        kv = B * Skv * Hkv * D
        return (f <= 40e12 and kv <= 256e6 and Sq <= Skv
                and B * Sq * Hq * D <= 64e6)

    Ds = [64, 128, 256]
    HQ = [(8, 8), (16, 16), (32, 32), (32, 8), (64, 8), (16, 4), (32, 4), (64, 16), (4, 4), (8, 4)]
    Ss = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    def gen():
        B = RNG.choice([1, 1, 1, 2, 4, 8, 16, 32, 64, 128, 256])
        Hq, Hkv = RNG.choice(HQ)
        D = RNG.choice(Ds)
        Skv = RNG.choice(Ss)
        if RNG.random() < 0.25:
            Sq = 1
        else:
            Sq = RNG.choice([s for s in Ss if s <= Skv])
        if not cap(B, Sq, Skv, Hq, Hkv, D):
            return None
        return (B, Sq, Skv, Hq, Hkv, D, tagattn(B, Sq, Skv, Hq, Hkv, D))
    return _fill_to(pool, n, gen)


ATTN_SHAPES = build_attn(100)

# ------------------------- swiglu -------------------------
def build_swiglu(n):
    pool = []
    for H in [1024, 2048, 4096, 5504, 8192, 11008]:
        pool.append((1024, H, tag2d(1024, H)))
    for T in [1, 128, 1024, 4096, 8192, 16384, 32768, 65536]:
        pool.append((T, 4096, tag2d(T, 4096)))
    pool += [(1, 8192, "narrow-1row"), (1, 65536, "narrow-1row"),
             (1048576, 64, "narrow-tall"), (524288, 128, "narrow-tall"),
             (262144, 256, "tall-thin"), (65536, 128, "tall"),
             (131072, 2048, "L"), (16384, 11008, "XL")]

    def gen():
        T = RNG.choice([t for t in TOKENS if t <= 1048576])
        H = RNG.choice([h for h in HIDDEN if h <= 16384])
        if T * H > 1 << 28:
            return None
        return (T, H, tag2d(T, H))
    return _fill_to(pool, n, gen)


SWIGLU_SHAPES = build_swiglu(100)

# ------------------------- cross_entropy -------------------------
def build_ce(n):
    pool = []
    for V in VOCABS:
        pool.append((1024, V, tag2d(1024, V)))
    for T in [1, 128, 1024, 4096, 8192, 16384, 32768, 65536, 131072]:
        pool.append((T, 32000, tag2d(T, 32000)))
    pool += [(1, 152064, "narrow-1row"), (1, 250000, "narrow-1row"),
             (262144, 4096, "narrow-tall"), (131072, 8192, "tall-thin"),
             (65536, 1024, "tall"), (8192, 152064, "L"),
             (16384, 128256, "XL"), (4096, 128256, "L")]

    def cap(T, V):
        return T * V <= 2e9 and T <= 131072 and V <= 250000

    def gen():
        T = RNG.choice([t for t in TOKENS if t <= 131072])
        V = RNG.choice(VOCABS)
        if not cap(T, V):
            return None
        return (T, V, tag2d(T, V))
    return _fill_to(pool, n, gen)


CE_SHAPES = build_ce(100)

# ------------------------- fused_linear_cross_entropy -------------------------
def build_flce(n):
    pool = []
    for V in [32000, 50257, 128256, 152064]:
        pool.append((1024, 4096, V, _tag_f(2 * 1024 * 4096 * V)))
    for T in [1, 128, 1024, 4096, 8192, 16384, 32768, 65536, 131072]:
        pool.append((T, 4096, 32000, _tag_f(2 * T * 4096 * 32000)))
    for H in [1024, 2048, 5120, 8192, 12288]:
        pool.append((2048, H, 32000, _tag_f(2 * 2048 * H * 32000)))
    pool += [(1, 8192, 152064, "narrow-1tok-wide"), (1, 4096, 128256, "XS-1tok"),
             (262144, 256, 32000, "narrow-H"), (65536, 512, 32000, "narrow-H"),
             (4096, 4096, 128256, "XL"), (8192, 4096, 128256, "XL")]

    def cap(T, H, V):
        return (V * H <= 2e9 and 2 * T * H * V <= 32e12 and T <= 131072)

    def gen():
        T = RNG.choice([t for t in TOKENS if t <= 131072])
        H = RNG.choice(HIDDEN)
        V = RNG.choice(VOCABS)
        if not cap(T, H, V):
            return None
        return (T, H, V, _tag_f(2 * T * H * V))
    return _fill_to(pool, n, gen)


FLCE_SHAPES = build_flce(100)

# ------------------------- rope -------------------------
def tagrope(B, Hq, Hkv, S, D):
    e = B * S * D * (Hq + Hkv)
    if S == 1:
        return "decode-b1"
    if S == 2 and B > 1:
        return "decode-spec2"
    if S >= 65536:
        return "longctx"
    if B >= 64:
        return "batch"
    return _tag_e(e)


def build_rope(n):
    pool = []
    for S in [128, 1024, 4096, 8192, 16384, 32768, 65536, 131072]:
        pool.append((1, 32, 8, S, 128, tagrope(1, 32, 8, S, 128)))
    for B in [2, 4, 8, 16, 32, 64, 128, 256]:
        pool.append((B, 32, 8, 1024, 128, tagrope(B, 32, 8, 1024, 128)))
    for D in [64, 128, 256]:
        pool.append((2, 32, 8, 4096, D, tagrope(2, 32, 8, 4096, D)))
    for (Hq, Hkv) in [(8, 8), (16, 16), (64, 8), (16, 4), (32, 32), (4, 4)]:
        pool.append((1, Hq, Hkv, 4096, 128, tagrope(1, Hq, Hkv, 4096, 128)))
    # decode: 规避 liger B>1&S=1 bug → 用 S=2 (spec) 与 B=1&S=1
    for B in [8, 32, 128, 256]:
        pool.append((B, 32, 8, 2, 128, "decode-spec2"))
    pool += [(1, 32, 8, 1, 128, "decode-b1"), (1, 8, 8, 131072, 128, "longctx"),
             (1, 2, 2, 524288, 64, "narrow-long"), (1, 1, 1, 262144, 64, "narrow-1head"),
             (1, 16, 4, 65536, 64, "longctx"), (256, 16, 4, 64, 64, "batch")]

    def cap(B, Hq, Hkv, S, D):
        # 规避 B>1 且 S=1 (liger v0.8.4 stride bug)
        if B > 1 and S == 1:
            return None
        return B * S * D * (Hq + Hkv) <= 268e6 and B <= 256 and S <= 524288

    def gen():
        B = RNG.choice([1, 1, 1, 2, 4, 8, 16, 32, 64, 128, 256])
        Hq, Hkv = RNG.choice([(8, 8), (16, 16), (32, 32), (32, 8), (64, 8),
                              (16, 4), (32, 4), (4, 4), (8, 4)])
        D = RNG.choice([64, 128, 256])
        S = RNG.choice([1, 2, 64, 128, 512, 1024, 2048, 4096, 8192, 16384,
                        32768, 65536, 131072, 262144])
        if not cap(B, Hq, Hkv, S, D):
            return None
        return (B, Hq, Hkv, S, D, tagrope(B, Hq, Hkv, S, D))
    return _fill_to(pool, n, gen)


ROPE_SHAPES = build_rope(100)

# ------------------------- topk -------------------------
def build_topk(n):
    pool = []
    for N in [4096, 32000, 128256, 250000]:
        pool.append((1024, N, 5, tag2d(1024, N)))
    for M in [1, 128, 1024, 8192, 65536, 131072]:
        pool.append((M, 32000, 5, tag2d(M, 32000)))
    for k in [1, 5, 10, 20, 50, 100]:
        pool.append((4096, 32000, k, tag2d(4096, 32000)))
    pool += [(1, 128256, 5, "narrow-1row"), (1, 250000, 10, "narrow-1row"),
             (1048576, 64, 5, "narrow-tall"), (262144, 256, 5, "tall-thin"),
             (65536, 1024, 50, "tall"), (4096, 8192, 100, "L")]

    def cap(M, N, k):
        return M * N <= 512e6 and M <= 1048576 and N <= 250000

    def gen():
        M = RNG.choice([m for m in TOKENS if m <= 1048576])
        N = RNG.choice([256, 1024, 4096, 8192, 16000, 32000, 50257, 100000,
                        128256, 152064, 250000])
        # 大 N 时限制 k: flag_gems bucket-sort topk 在 大N x 大k 时 ptxas 编译爆炸
        if N > 50000:
            k = RNG.choice([1, 5, 10])
        else:
            k = RNG.choice([1, 5, 10, 20, 50, 100])
        if not cap(M, N, k):
            return None
        return (M, N, k, tag2d(M, N))
    return _fill_to(pool, n, gen)


TOPK_SHAPES = build_topk(100)

# ------------------------- embedding -------------------------
def build_emb(n):
    pool = []
    for V in [32000, 50257, 128256, 152064, 250000]:
        pool.append((1024, V, 4096, _tag_e(1024 * 4096)))
    for T in [1, 128, 1024, 4096, 16384, 65536, 262144, 1048576]:
        pool.append((T, 32000, 1024, _tag_e(T * 1024)))
    for H in [256, 512, 1024, 2048, 4096, 8192]:
        pool.append((2048, 32000, H, _tag_e(2048 * H)))
    pool += [(1, 152064, 8192, "narrow-1row"), (1, 250000, 1024, "narrow-1row"),
             (1048576, 512, 64, "narrow-tall"), (262144, 1024, 256, "tall-thin"),
             (65536, 1000, 128, "tall"), (8192, 250000, 512, "XL-wideV")]

    def cap(T, V, H):
        return V * H <= 1e9 and T * H <= 128e6 and T <= 1048576 and V <= 250000

    def gen():
        T = RNG.choice([t for t in TOKENS if t <= 1048576])
        V = RNG.choice(VOCABS)
        H = RNG.choice(HIDDEN)
        if not cap(T, V, H):
            return None
        return (T, V, H, _tag_e(T * H))
    return _fill_to(pool, n, gen)


EMB_SHAPES = build_emb(100)

if __name__ == "__main__":
    for name in ["ELEM_SHAPES", "SOFTMAX_SHAPES", "NORM_SHAPES", "MM_SHAPES",
                 "FP8_MM_SHAPES", "BMM_SHAPES", "MV_SHAPES", "ATTN_SHAPES",
                 "SWIGLU_SHAPES", "CE_SHAPES", "FLCE_SHAPES", "ROPE_SHAPES",
                 "TOPK_SHAPES", "EMB_SHAPES"]:
        lst = globals()[name]
        keys = [x[:-1] for x in lst]
        assert len(lst) == 100, f"{name}: {len(lst)}"
        assert len(set(map(str, keys))) == 100, f"{name}: 存在重复"
        print(f"{name}: 100 OK")
    print("ALL 100-SHAPE SETS OK")
