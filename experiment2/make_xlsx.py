#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 results/*.json -> B200_ops_benchmark.xlsx
- Sheet 1  : 总览 (实验信息 + 20算子清单)
- Sheet 2+ : 每算子一个 sheet (20 shape × 性能指标)

运行: /root/miniconda3/envs/kaixin/bin/python make_xlsx.py
"""

import os
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import load_results, PEAK_BF16_TFLOPS, PEAK_BW_GBPS

# ---------------- 算子元信息 (顺序: 必需5个在前) ----------------
OPS = [
    # (结果名, sheet名, 中文名, 类别, 来源仓库, 框架, 实际应用场景, 运行环境)
    ("unified_attention", "01_unified_attention", "unified_attention", "注意力",
     "vllm-project/vllm v0.24.0", "Triton",
     "vLLM v1统一注意力算子, LLM推理服务(Prefill/Decode)", "python3.10 + torch2.11.0+cu130"),
    ("flash_attention", "02_flash_attention", "flash_attention", "注意力",
     "Dao-AILab/flash-attention v2.8.3", "CUDA",
     "FlashAttention-2, LLM训练/推理注意力标准实现", "python3.10 + torch2.11.0+cu130"),
    ("softmax", "03_softmax", "softmax", "归约/归一化",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "注意力权重/LDAM损失/分类输出层", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("silu", "04_silu", "silu", "激活函数",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "SwiGLU激活(Llama/Gemma/Qwen MLP)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("rms_norm", "05_rms_norm", "RMSNorm", "归一化",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "Llama/Qwen/DeepSeek层归一化", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("mm", "06_mm", "matmul (mm)", "GEMM",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "Linear层/投影/FFN 矩阵乘", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("layer_norm", "07_layer_norm", "layer_norm", "归一化",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "GPT/BERT/ViT层归一化", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("gelu", "08_gelu", "gelu", "激活函数",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "GPT/BERT FFN激活", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("add", "09_add", "add", "逐元素",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "残差连接/偏置加法", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("mul", "10_mul", "mul", "逐元素",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "门控乘法/缩放/掩码", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("sum", "11_sum", "sum (dim=-1)", "归约",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "均值池化/损失聚合/softmax分母", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("argmax", "12_argmax", "argmax", "归约",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "贪心解码/评估指标", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("cross_entropy", "13_cross_entropy", "cross_entropy", "损失函数",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "LLM训练损失(分块融合, 免logits物化)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("fused_add_rms_norm", "14_fused_add_rms_norm", "fused_add_rms_norm", "融合算子",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "Llama残差加+RMSNorm融合(post-attn/FFN)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("swiglu", "15_swiglu", "swiglu (SiLU·mul)", "融合算子",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "Llama/DeepSeek FFN gate/up 融合", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("rope", "16_rope", "RoPE (apply_rotary_pos_emb)", "位置编码",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "旋转位置编码(Llama/Qwen/Mistral)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("clamp", "17_clamp", "clamp", "逐元素",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "梯度裁剪/数值截断/logit softcap", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("cumsum", "18_cumsum", "cumsum", "前缀扫描",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "累积分布/PrefixLM mask/流量统计", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("topk", "19_topk", "topk", "检索",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "LLM采样(top-k/top-p)/MoE路由", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("embedding", "20_embedding", "embedding", "嵌入查表",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "词嵌入/ID特征查表", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("bmm", "21_bmm", "bmm (批量矩阵乘)", "矩阵乘",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "注意力QK^T/PV逐头bmm、批量Linear", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("addmm", "22_addmm", "addmm (偏置融合GEMM)", "矩阵乘",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "Linear层 y=x@W+b (投影/FFN)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("mv", "23_mv", "mv (矩阵-向量乘)", "矩阵乘",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "decode阶段Linear (GEMV)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("fp8_matmul", "24_fp8_matmul", "fp8_matmul (MXFP8量化GEMM)", "矩阵乘",
     "FlagOpen/FlagGems v5.0.2", "Triton",
     "FP8低精度GEMM (128分块scale, 量化LLM推理)", "py3.11(kaixin) + torch2.11.0+cu128"),
    ("fused_linear_cross_entropy", "25_fused_linear_cross_entropy", "fused_linear_cross_entropy", "矩阵乘融合",
     "linkedin/Liger-Kernel v0.8.4", "Triton",
     "LLM训练输出层: lm_head GEMM+CE融合, 免logits物化", "py3.11(kaixin) + torch2.11.0+cu128"),
]

# 每个算子 sheet 的列定义
PERF_COLS = [
    ("no.", "no", 6),
    ("规模档位", "tag", 12),
    ("shape", "shape", 34),
    ("维度明细", "dims", 30),
    ("dtype", "dtype", 7),
    ("元素量", "elements", 12),
    ("FLOPs(模型)", "flops", 14),
    ("Bytes(模型)", "bytes", 14),
    ("时延 mean (us)", "latency_us_mean", 14),
    ("时延 min (us)", "latency_us_min", 13),
    ("时延 p50 (us)", "latency_us_p50", 13),
    ("迭代次数", "iters", 9),
    ("TFLOPS", "TFLOPS", 11),
    ("带宽 (GB/s)", "GBps", 12),
    ("TFLOPS峰值占比%", "TFLOPS_peak_pct", 15),
    ("带宽峰值占比%", "BW_peak_pct", 14),
    ("绝对误差", "err_vs_ref", 12),
    ("相对误差", "err_rel", 12),
    ("正确性", "status", 10),
    ("来源仓库", "source", 38),
]

THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HDR_FILL = PatternFill("solid", fgColor="2F5496")
HDR_FONT = Font(bold=True, color="FFFFFF", size=10)
TITLE_FONT = Font(bold=True, size=14, color="2F5496")
SUB_FONT = Font(size=10, color="595959")
OK_FONT = Font(color="006100", size=10)
CHECK_FONT = Font(color="9C0006", size=10)


def fmt_num(v):
    """大数字加千分位"""
    if isinstance(v, (int, float)) and abs(v) >= 1000:
        return f"{v:,.0f}"
    return v


def write_op_sheet(wb, res_name, sheet_name, cn_name, source):
    rows = load_results(res_name)
    if not rows:
        print(f"[WARN] {res_name} 无结果, 跳过")
        return
    ws = wb.create_sheet(sheet_name)
    # 标题行
    ws.cell(1, 1, f"{cn_name}  —  {source}  —  NVIDIA B200 (SM100), bf16").font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(PERF_COLS))
    ws.cell(2, 1, "计时: CUDA Event, 每次rep前清空L2(256MB), 自适应迭代; "
                  "TFLOPS=FLOPs模型/时延, GB/s=Bytes模型/时延; "
                  "峰值参考: BF16 dense 2250 TFLOPS / HBM3e 8000 GB/s").font = SUB_FONT
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(PERF_COLS))
    hdr_row = 4
    for ci, (title, key, width) in enumerate(PERF_COLS, 1):
        c = ws.cell(hdr_row, ci, title)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, BORDER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width = width
    r = hdr_row + 1
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
    ws.auto_filter.ref = f"A{hdr_row}:{get_column_letter(len(PERF_COLS))}{r - 1}"
    return rows


def write_summary(wb, per_op_stats):
    ws = wb.create_sheet("总览", 0)
    ws.sheet_view.showGridLines = False
    ws.cell(1, 1, "NVIDIA B200 常用算子性能基准 (20 算子 × 20 shape)").font = Font(bold=True, size=16, color="2F5496")
    info = [
        ("GPU", "NVIDIA B200 (SM100.0), 183GB HBM3e"),
        ("理论峰值", f"BF16/FP16 dense {PEAK_BF16_TFLOPS:.0f} TFLOPS;  HBM3e {PEAK_BW_GBPS:.0f} GB/s"),
        ("测试 dtype", "bfloat16 (sum/cumsum 正确性校验用 fp32)"),
        ("计时方法", "CUDA Event 单次计时; 每次 rep 前写零 256MB buffer 清空 L2; 预热≥5次; 自适应迭代25~300次 (总时长~0.4s/shape)"),
        ("性能指标", "TFLOPS = FLOPs模型 / mean时延;  GB/s = Bytes模型 / mean时延;  峰值占比% = 相对上述理论峰值"),
        ("正确性", "与 PyTorch eager/fp32 参考实现对比, 相对误差 ≤ 2e-2 判 OK (bf16 输出舍入 ~0.4%)"),
        ("shape 设计", "每算子 20 个: XS/S/M/L/XL + 狭长形态(单行超宽/超高窄列/GEMV/超长序列/decode/投机解码)"),
        ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    r = 3
    for k, v in info:
        ws.cell(r, 1, k).font = Font(bold=True, size=10)
        c = ws.cell(r, 2, v)
        c.font = Font(size=10)
        r += 1
    r += 1
    headers = ["序号", "Sheet", "算子", "类别", "来源仓库(权威开源)", "框架",
               "实际应用场景", "运行环境", "shape数",
               "最佳TFLOPS", "最佳GB/s", "时延范围(us)", "正确性"]
    widths = [6, 22, 26, 10, 34, 8, 44, 30, 8, 12, 11, 20, 8]
    for ci, (h, w) in enumerate(zip(headers, widths), 1):
        c = ws.cell(r, ci, h)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, BORDER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width = w
    r += 1
    for i, (res_name, sheet_name, cn_name, cat, repo, fw, usage, env) in enumerate(OPS, 1):
        st = per_op_stats.get(res_name, {})
        vals = [i, sheet_name, cn_name, cat, repo, fw, usage, env,
                st.get("n", ""), st.get("best_tflops", ""), st.get("best_gbps", ""),
                st.get("lat_range", ""), st.get("ok", "")]
        for ci, v in enumerate(vals, 1):
            c = ws.cell(r, ci, v)
            c.border = BORDER
            c.font = Font(size=10)
            if ci in (10, 11) and isinstance(v, (int, float)):
                c.number_format = "0.00"
            if ci == 13 and v == "400 OK" or (ci == 13 and v == "20/20 OK"):
                c.font = OK_FONT
        r += 1
    ws.freeze_panes = "A14"
    note_r = r + 1
    ws.cell(note_r, 1, "注1").font = Font(bold=True, size=10)
    ws.cell(note_r, 2, "unified_attention 为 vLLM v1 统一注意力接口的 Triton 实现 "
                       "(vllm.v1.attention.ops.triton_unified_attention), paged KV cache + varlen query, causal。").font = Font(size=10)
    ws.cell(note_r + 1, 1, "注2").font = Font(bold=True, size=10)
    ws.cell(note_r + 1, 2, "rope 的 decode 档使用 S=2 (投机解码 verify 场景): Liger-Kernel v0.8.4 的 rope 在 B>1 且 S=1 时 "
                           "transpose 后 stride 语义不符 (q.stride(1)=D 而非 H*D), 上游限制, 已规避。").font = Font(size=10)
    ws.cell(note_r + 2, 1, "注3").font = Font(bold=True, size=10)
    ws.cell(note_r + 2, 2, "flash_attention 为 flash_attn 2.8.3 CUDA kernel (FA2); 统一 causal=True (bottom-right 对齐, "
                           "Sq=1 时等价于 attend 全部上下文)。").font = Font(size=10)


def main():
    wb = Workbook()
    wb.remove(wb.active)
    stats = {}
    for res_name, sheet_name, cn_name, cat, repo, fw, usage, env in OPS:
        rows = load_results(res_name)
        if rows:
            tps = [r["TFLOPS"] for r in rows if isinstance(r.get("TFLOPS"), (int, float))]
            bws = [r["GBps"] for r in rows if isinstance(r.get("GBps"), (int, float))]
            lats = [r["latency_us_mean"] for r in rows]
            ok = sum(1 for r in rows if r.get("status") == "OK")
            stats[res_name] = {
                "n": len(rows),
                "best_tflops": max(tps) if tps else "",
                "best_gbps": max(bws) if bws else "",
                "lat_range": f"{min(lats):.1f} ~ {max(lats):,.1f}",
                "ok": f"{ok}/{len(rows)} OK",
            }
    # 先写算子 sheet, 再插总览到最前
    for res_name, sheet_name, cn_name, cat, repo, fw, usage, env in OPS:
        write_op_sheet(wb, res_name, sheet_name, cn_name, repo)
    write_summary(wb, stats)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "B200_ops_benchmark.xlsx")
    wb.save(out)
    print(f"[XLSX] 已生成: {out}")
    print(f"[XLSX] sheets: {wb.sheetnames}")


if __name__ == "__main__":
    main()
