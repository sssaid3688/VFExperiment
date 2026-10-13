# -*- coding: utf-8 -*-
"""
B200 算子性能基准测试 - 公共工具

计时方法(与 triton.testing.do_bench 同风格):
  - 每次 rep 前清空 L2 cache (写零 256MB buffer), 用 CUDA event 计时单次调用
  - 自适应迭代次数: 按预估耗时把总计时时间控制在 ~0.4s
指标:
  - latency_us (mean / min / p50)
  - TFLOPS     = flops_model / latency
  - GBps       = bytes_model / latency   (有效带宽, 内存受限算子的核心指标)
  - 相对 B200 峰值 (BF16 dense ~2250 TFLOPS, HBM3e ~8000 GB/s) 的百分比
结果立即以 JSON 落盘 (results/*.json), 每 case 一个条目, 断电/异常不丢数据。
"""

import json
import math
import os
import time

import torch

# ---------------- B200 (SM100) 理论峰值 ----------------
PEAK_BF16_TFLOPS = 2250.0   # dense BF16/FP16 tensor core
PEAK_BW_GBPS = 8000.0       # HBM3e

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

_FLUSH_BUF = None


def _flush_buf():
    global _FLUSH_BUF
    if _FLUSH_BUF is None:
        # 256MB, 用于清空 L2 (B200 L2 = 126MB)
        _FLUSH_BUF = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    return _FLUSH_BUF


def bench_one(fn, flops, nbytes, dtype_str="bf16", time_budget_s=0.4):
    """对一个 callable 做 GPU 基准, 返回指标 dict. fn 必须已捕获所有输入."""
    # 预估单次耗时(保守), 决定迭代次数
    est_s = max(nbytes / 6.5e12, flops / 1.5e12, 3e-6)
    iters = int(min(300, max(25, round(time_budget_s / est_s))))
    warmup = max(5, min(25, iters // 5 + 3))
    flush = _flush_buf()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()  # 清 L2
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()

    us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, stops))
    mean_us = sum(us) / len(us)
    p50_us = us[len(us) // 2]
    min_us = us[0]

    tflops = flops / (mean_us * 1e-6) / 1e12 if flops > 0 else float("nan")
    gbps = nbytes / (mean_us * 1e-6) / 1e9 if nbytes > 0 else float("nan")
    return {
        "dtype": dtype_str,
        "iters": iters,
        "latency_us_mean": round(mean_us, 3),
        "latency_us_min": round(min_us, 3),
        "latency_us_p50": round(p50_us, 3),
        "TFLOPS": round(tflops, 3) if flops > 0 else "",
        "GBps": round(gbps, 2) if nbytes > 0 else "",
        "TFLOPS_peak_pct": round(tflops / PEAK_BF16_TFLOPS * 100, 2) if flops > 0 else "",
        "BW_peak_pct": round(gbps / PEAK_BW_GBPS * 100, 2) if nbytes > 0 else "",
    }


def save_result(op_name, row):
    """追加一行结果到 results/<op_name>.json (JSONL)"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    row = dict(row)
    row["op"] = op_name
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(RESULTS_DIR, op_name + ".json")
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[SAVED] {op_name}: {row.get('shape','')} "
          f"mean={row.get('latency_us_mean')}us TFLOPS={row.get('TFLOPS')} GBps={row.get('GBps')}",
          flush=True)


def load_results(op_name):
    path = os.path.join(RESULTS_DIR, op_name + ".json")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def clear_result(op_name):
    path = os.path.join(RESULTS_DIR, op_name + ".json")
    if os.path.exists(path):
        os.remove(path)


def err_str(err):
    if err is None or (isinstance(err, float) and math.isnan(err)):
        return ""
    return f"{err:.3e}" if isinstance(err, float) else str(err)


def torch_version_str():
    return f"torch {torch.__version__}"
