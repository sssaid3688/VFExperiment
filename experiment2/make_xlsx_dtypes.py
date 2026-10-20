#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 results_dtypes/*.json (+ 复用 fp8_matmul 主结果) -> B200_ops_benchmark_dtypes.xlsx

dtype 对比版工作簿:
- 与 bf16 主工作簿 (B200_ops_benchmark.xlsx) 的 25 算子 x 100 shape 完全同 shape
- 每算子一个 sheet, 行内 dtype 列标注 fp16 / fp32 / fp8(e4m3)
- 总览含 支持/跳过矩阵 与 各 dtype 最佳性能

运行: /root/miniconda3/envs/kaixin/bin/python make_xlsx_dtypes.py
"""

import os
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import RESULTS_DIR, load_results, PEAK_BF16_TFLOPS, PEAK_BW_GBPS
from make_xlsx import OPS, PERF_COLS, HDR_FILL, HDR_FONT, BORDER, TITLE_FONT, SUB_FONT, OK_FONT, CHECK_FONT, fmt_num

DTYPES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_dtypes")

# ---------------- 每算子的 dtype 计划: (标签, 文件名, 不支持原因None=可跑) ----------------
def dtype_plan(res_name):
    plan = {
        "fp16": (f"{res_name}_fp16", None),
        "fp32": (f"{res_name}_fp32", None),
    }
    if res_name == "flash_attention":
        plan["fp32"] = (None, "FA2 仅支持 fp16/bf16")
    if res_name == "unified_attention":
        plan["fp8"] = (None, "该 Triton 路径无 Q/KV fp8 (B200 fp8 注意力需 FA4/cuDNN)")
    if res_name == "mm":
        plan["fp8"] = ("mm_fp8", None)  # MXFP8, K/N 需 %128
    elif res_name == "fp8_matmul":
        # 原生 fp8 算子: fp16/fp32 不适用, fp8 直接复用主结果
        plan = {
            "fp16": (None, "算子本身为 fp8(e4m3) 量化 GEMM"),
            "fp32": (None, "算子本身为 fp8(e4m3) 量化 GEMM"),
            "fp8": ("__main__fp8_matmul", None),  # 特殊标记: 读主 results/
        }
    else:
        plan["fp8"] = (None, "无 fp8 实现 (逐元素/归一化/损失类 fp8 无意义或无开源kernel)")
    return plan


def load_dtype_rows(res_name):
    """返回 [(dtype_label, rows), ...] 与 跳过信息 [(dtype_label, reason)]"""
    blocks, skips = [], []
    for d in ("fp16", "fp32", "fp8"):
        fname, reason = dtype_plan(res_name)[d]
        if reason:
            skips.append((d, reason))
            continue
        if fname == "__main__fp8_matmul":
            rows = [dict(r, dtype="fp8(e4m3)+scale") for r in load_results(res_name)]
        else:
            path = os.path.join(DTYPES_DIR, fname + ".json")
            rows = []
            if os.path.exists(path):
                with open(path) as f:
                    rows = [__import__("json").loads(l) for l in f if l.strip()]
        if not rows:
            skips.append((d, "未运行"))
        else:
            if d == "fp8" and res_name == "mm":
                for r in rows:
                    r["dtype"] = "fp8(e4m3)+scale"
            blocks.append((d, rows))
    return blocks, skips


def write_op_sheet(wb, res_name, sheet_name, cn_name, source):
    blocks, skips = load_dtype_rows(res_name)
    if not blocks:
        return
    ws = wb.create_sheet(sheet_name)
    ws.cell(1, 1, f"{cn_name} — dtype 对比 (fp16 / fp32 / fp8) — NVIDIA B200 (SM100), 同 shapes_100 的 100 shape").font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(PERF_COLS))
    note = "计时: CUDA Event, 每次rep前清空L2; TFLOPS=FLOPs模型/时延, GB/s=Bytes模型/时延; 峰值参考: BF16 dense 2250 / FP16 2250 / TF32~1125(实际按2250口径) / HBM3e 8000 GB/s"
    if skips:
        note += " | 未覆盖: " + "; ".join(f"{d}({r})" for d, r in skips)
    ws.cell(2, 1, note).font = SUB_FONT
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(PERF_COLS))
    hdr = 4
    for ci, (title, key, width) in enumerate(PERF_COLS, 1):
        c = ws.cell(hdr, ci, title)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, BORDER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width = width
    r = hdr + 1
    for d, rows in blocks:
        for row in rows:
            for ci, (title, key, width) in enumerate(PERF_COLS, 1):
                v = row.get(key, "")
                if key in ("elements", "flops", "bytes"):
                    v = fmt_num(v)
                c = ws.cell(r, ci, v)
                c.border = BORDER
                c.font = Font(size=10)
                if key in ("TFLOPS", "GBps", "latency_us_mean", "latency_us_min",
                           "latency_us_p50", "TFLOPS_peak_pct", "BW_peak_pct"):
                    if isinstance(v, (int, float)):
                        c.number_format = "0.00"
                    c.alignment = Alignment(horizontal="right")
                elif key == "status":
                    c.font = OK_FONT if v == "OK" else CHECK_FONT
                    c.alignment = Alignment(horizontal="center")
                elif key in ("no.", "iters", "dtype"):
                    c.alignment = Alignment(horizontal="center")
            r += 1
    ws.freeze_panes = "C5"
    ws.auto_filter.ref = f"A{hdr}:{get_column_letter(len(PERF_COLS))}{r - 1}"
    return blocks, skips


def main():
    wb = Workbook()
    wb.remove(wb.active)
    stats = {}
    for res_name, sheet_name, cn_name, cat, repo, fw, usage, env in OPS:
        out = write_op_sheet(wb, res_name, sheet_name, cn_name, repo)
        if out:
            stats[res_name] = out
    # 总览
    ws = wb.create_sheet("总览", 0)
    ws.sheet_view.showGridLines = False
    ws.cell(1, 1, "NVIDIA B200 算子 dtype 对比基准 (fp16 / fp32 / fp8)").font = Font(bold=True, size=16, color="2F5496")
    info = [
        ("说明", "与 bf16 主工作簿 (B200_ops_benchmark.xlsx) 完全相同的 25 算子 x 100 shape (shapes_100.py); 本簿按 dtype 分块, 行内 dtype 列标注"),
        ("fp16", "全部 25 算子中 24 个可跑 (fp8_matmul 本身为 fp8 算子除外)"),
        ("fp32", "FlagGems/Liger/vLLM-Triton 均支持; flash_attention 不支持 (FA2 仅 fp16/bf16)"),
        ("fp8", "仅 GEMM 族有真实 fp8 路径: mm 以 MXFP8(128分块scale) 跑同 shape 集 (K/N 非 128 倍数的 shape 跳过); fp8_matmul 为原生 fp8 算子(复用主结果); 注意力/逐元素/归一化/损失无 fp8 实现或不适用"),
        ("指标口径", "TFLOPS/GB/s 与 bf16 簿一致; 峰值占比统一按 BF16 dense 2250 TFLOPS 与 HBM3e 8000 GB/s 口径 (fp16 峰值同 2250; fp8 dense 峰值为 4500, 读 fp8 行的占比时注意)"),
        ("正确性", "fp32 参考实现比对 (fp8 行为与反量化 fp32 GEMM 比对, 容差 8e-2)"),
        ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    r = 3
    for k, v in info:
        ws.cell(r, 1, k).font = Font(bold=True, size=10)
        ws.cell(r, 2, v).font = Font(size=10)
        r += 1
    r += 1
    headers = ["序号", "Sheet", "算子", "fp16(best TFLOPS/GBps, shapes)", "fp32(best TFLOPS/GBps, shapes)", "fp8(best TFLOPS/GBps, shapes)", "未覆盖 dtype 及原因"]
    widths = [6, 26, 24, 34, 34, 34, 52]
    for ci, (h, w) in enumerate(zip(headers, widths), 1):
        c = ws.cell(r, ci, h)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, BORDER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width = w
    r += 1
    for i, (res_name, sheet_name, cn_name, cat, repo, fw, usage, env) in enumerate(OPS, 1):
        cells = [i, sheet_name, cn_name]
        blocks, skips = stats.get(res_name, ([], []))
        for d in ("fp16", "fp32", "fp8"):
            blk = [b for b in blocks if b[0] == d]
            if blk:
                rows = blk[0][1]
                tps = [x["TFLOPS"] for x in rows if isinstance(x.get("TFLOPS"), (int, float))]
                bws = [x["GBps"] for x in rows if isinstance(x.get("GBps"), (int, float))]
                bt = max(tps) if tps else float("nan")
                bg = max(bws) if bws else float("nan")
                cells.append(f"{bt:.1f} TF / {bg:.0f} GB/s ({len(rows)} shapes)")
            else:
                cells.append("—")
        cells.append("; ".join(f"{d}: {rsn}" for d, rsn in skips) if skips else "")
        for ci, v in enumerate(cells, 1):
            c = ws.cell(r, ci, v)
            c.border = BORDER
            c.font = Font(size=10)
        r += 1
    ws.freeze_panes = "A12"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "B200_ops_benchmark_dtypes.xlsx")
    wb.save(out)
    print(f"[XLSX] 已生成: {out}")
    print(f"[XLSX] sheets: {wb.sheetnames}")


if __name__ == "__main__":
    main()
