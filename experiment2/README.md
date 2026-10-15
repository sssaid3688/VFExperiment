# Experiment 2 — NVIDIA B200 常用算子性能基准 (25 算子 × 100 shape)

## 产出

- **`B200_ops_benchmark.xlsx`** — 全部性能数据。Sheet `总览` 为实验信息与算子清单；其余 **25 个 sheet 每算子一个**，每 sheet 100 行 shape，包含：
  - shape / 维度明细 / dtype / 元素量 / FLOPs / Bytes
  - **时延 (mean/min/p50, us)**、**TFLOPS**、**带宽 GB/s**、TFLOPS 峰值占比 %、带宽峰值占比 %
  - 相对 PyTorch 参考实现的误差与正确性状态、来源仓库
- `results/*.json` — 每 shape 一条 JSONL 原始记录（crash-safe）
- `bench_flaggems.py` / `bench_liger.py` / `bench_attn.py` / `bench_matmul.py` — 基准脚本
- `common.py`（计时与指标）/ `shapes_100.py`（25×100 shape 生成器, 固定种子可复现）/ `shapes.py`（旧20-shape配置）/ `make_xlsx.py`（汇总）/ `run_n100.sh`（串行全量入口）
- 归档: `results_20shape/` 与 `B200_ops_benchmark_20shape.xlsx` 为旧 20-shape 版本

## 算子清单（全部来自权威开源仓，均已在 B200 上实跑验证）

| # | 算子 | 来源仓库 | 类别 | 实际应用场景 |
|---|------|----------|------|--------------|
| 1 | unified_attention | vllm-project/vllm v0.24.0 (Triton) | 注意力 | vLLM v1 统一注意力算子（paged KV + varlen, causal），LLM 推理服务 |
| 2 | flash_attention | Dao-AILab/flash-attention v2.8.3 (CUDA) | 注意力 | FlashAttention-2，LLM 训练/推理标准注意力 |
| 3 | softmax | FlagOpen/FlagGems v5.0.2 (Triton) | 归约 | 注意力权重、分类输出 |
| 4 | silu | FlagOpen/FlagGems v5.0.2 (Triton) | 激活 | SwiGLU 激活（Llama/Gemma/Qwen MLP） |
| 5 | rms_norm (RMSNorm) | linkedin/Liger-Kernel v0.8.4 (Triton) | 归一化 | Llama/Qwen/DeepSeek 层归一化 |
| 6 | mm (matmul) | FlagOpen/FlagGems v5.0.2 (Triton) | GEMM | Linear/投影/FFN 矩阵乘 |
| 7 | layer_norm | linkedin/Liger-Kernel v0.8.4 (Triton) | 归一化 | GPT/BERT/ViT 层归一化 |
| 8 | gelu | FlagOpen/FlagGems v5.0.2 (Triton) | 激活 | GPT/BERT FFN 激活 |
| 9 | add | FlagOpen/FlagGems v5.0.2 (Triton) | 逐元素 | 残差连接、偏置加法 |
| 10 | mul | FlagOpen/FlagGems v5.0.2 (Triton) | 逐元素 | 门控乘法、缩放、掩码 |
| 11 | sum (dim=-1) | FlagOpen/FlagGems v5.0.2 (Triton) | 归约 | 均值池化、损失聚合 |
| 12 | argmax | FlagOpen/FlagGems v5.0.2 (Triton) | 归约 | 贪心解码、评估指标 |
| 13 | cross_entropy | linkedin/Liger-Kernel v0.8.4 (Triton) | 损失 | LLM 训练损失（分块融合） |
| 14 | fused_add_rms_norm | linkedin/Liger-Kernel v0.8.4 (Triton) | 融合 | Llama 残差加 + RMSNorm（post-attn/FFN） |
| 15 | swiglu (SiLU·mul) | linkedin/Liger-Kernel v0.8.4 (Triton) | 融合 | Llama/DeepSeek FFN gate/up 融合 |
| 16 | rope (apply_rotary_pos_emb) | linkedin/Liger-Kernel v0.8.4 (Triton) | 位置编码 | 旋转位置编码（Llama/Qwen/Mistral） |
| 17 | clamp | FlagOpen/FlagGems v5.0.2 (Triton) | 逐元素 | 梯度裁剪、logit softcap |
| 18 | cumsum | FlagOpen/FlagGems v5.0.2 (Triton) | 扫描 | 累积分布、PrefixLM mask |
| 19 | topk | FlagOpen/FlagGems v5.0.2 (Triton) | 检索 | LLM top-k 采样、MoE 路由 |
| 20 | embedding | FlagOpen/FlagGems v5.0.2 (Triton) | 查表 | 词嵌入、ID 特征查表 |
| 21 | bmm (批量矩阵乘) | FlagOpen/FlagGems v5.0.2 (Triton) | 矩阵乘 | 注意力 QK^T/PV 逐头 bmm、批量 Linear |
| 22 | addmm (偏置融合 GEMM) | FlagOpen/FlagGems v5.0.2 (Triton) | 矩阵乘 | Linear 层 y=x@W+b（投影/FFN） |
| 23 | mv (矩阵-向量乘) | FlagOpen/FlagGems v5.0.2 (Triton) | 矩阵乘 | decode 阶段 Linear（GEMV） |
| 24 | fp8_matmul (MXFP8 量化 GEMM) | FlagOpen/FlagGems v5.0.2 (Triton) | 矩阵乘 | FP8 低精度 GEMM（128 分块 scale，量化 LLM 推理） |
| 25 | fused_linear_cross_entropy | linkedin/Liger-Kernel v0.8.4 (Triton) | 矩阵乘融合 | LLM 训练输出层：lm_head GEMM+CE 融合，免 logits 物化 |

运行环境：
- **python3.10**（unified_attention + flash_attention）：torch 2.11.0+cu130、vllm 0.24.0、flash-attn 2.8.3
- **py3.11 kaixin conda env**（其余 18 个）：torch 2.11.0+cu128、triton 3.6.0、FlagGems 5.0.2、Liger-Kernel 0.8.4

## 方法

- GPU：NVIDIA B200 (SM100.0, 183GB HBM3e)；dtype：bf16
- 计时：CUDA Event 单次计时，每次 rep 前写零 256MB buffer **清空 L2**；预热 ≥5；自适应迭代 25~300 次（每 shape 约 0.4s）
- TFLOPS = FLOPs 模型 / mean 时延；GB/s = Bytes 模型 / mean 时延；峰值参考 2250 TFLOPS (BF16 dense) / 8000 GB/s (HBM3e)
- 注意力 FLOPs：causal (bottom-right 对齐) 精确 pair 数 = Sq·(Skv−Sq) + Sq(Sq+1)/2，每 pair 4·Hq·D
- fp8_matmul 输入为 MXFP8 量化张量（e4m3 + 128 分块 scale），正确性与"反量化后 fp32 GEMM"比对（fp8 舍入容差 8e-2）；其 TFLOPS 峰值占比列按 BF16 2250 峰值口径（fp8 dense 理论峰值为 4500）
- 正确性：与 PyTorch eager/fp32 参考实现对比, 相对误差 ≤ 2e-2 判 OK（bf16 输出舍入 ~0.4%）；**500/500 全部 OK**
- 全部基准**串行**运行（曾发现并行运行互相干扰 GPU 导致数据失真，已全部串行重跑）

## Shape 设计（每算子 100 个, shapes_100.py 程序化生成: 固定阶梯 + 狭长形态 + 真实模型尺寸 + 固定随机种子补齐）

XS/S/M/L/XL 大中小全覆盖，并包含**狭长**形态：
- 逐元素/归约类：`(1, 1M)` 单行超宽、`(1M, 64)` 超高窄列、`(524288, 128)` 等
- mm/addmm/fp8：`(1, K, N)` GEMV、`(131072, 64, 64)` 超高窄、`(64, 131072, 64)` 超长 K、`(64, 4096, 65536)` 超宽 N；Llama 实际尺寸 (4096,4096,11008) 等
- bmm：`(4096, 8, 8, 8)` 海量微矩阵、注意力 QK^T/PV 式 `(32, 4096, 128, 4096)`、`(1, 131072, 64, 64)` 超高窄
- mv：`(1, 1M)` 长内积、`(1048576, 1)`、`(131072, 64)` 超高窄、Llama FFN (11008, 4096)
- fused_linear_cross_entropy：Llama-2/3/GPT-2/Qwen 词表 × 各 hidden；`(262144, 256, 32000)` 超长 token 窄 hidden
- attention：128K 长序列 (1,131072,131072,16,4,64)、decode (128,1,1024)、单 token 32K 上下文、D=256 大头维、GQA/MHA
- CE：Llama-2/3/GPT-2/Qwen 词表 (32000/128256/50257/152064)；topk：k=1~100；embedding：大词表 (250000, 1024)

## 关键结论（详见 xlsx）

- 计算受限 GEMM 族：**fused_linear_cross_entropy 1369 TFLOPS**、**mm 1074**、**addmm 1027**（≈48% BF16 峰值）、fp8_matmul (MXFP8) 940、**bmm 710**（注意力式逐头 bmm 低于大方阵 GEMM）、flash_attention 434、unified_attention 162 TFLOPS（Triton 注意力与 CUDA FA2 在 Blackwell 上差距明显）
- 带宽受限：融合算子最优——swiglu 7073 GB/s、fused_add_rms_norm 6931 GB/s、cumsum 6701 GB/s、sum 6216 GB/s（≈ 85% HBM3e 峰值）；**mv (GEMV) 5014 GB/s** 为典型带宽受限矩阵乘
- 小 shape（<1MB）全部受 kernel 启动开销支配（~5-15us 地板）

## 已知上游限制

- Liger-Kernel v0.8.4 rope 在 **B>1 且 S=1**（纯 decode batch）时 transpose 后 stride 语义错误（q.stride(1)=D 而非 H·D），正确性校验失败；decode 档改用 **S=2 投机解码 verify** shape 并在总览 sheet 注明
- FlagGems 对 SM100 无专属调优配置（日志提示 "Unsupported GPU arch specialization"），性能为 Triton 默认调度结果

## 复现

```bash
cd /home/ubuntu/workspace/oyhj/papers2/experiment2
# 一键串行全量 (推荐, GPU独占)
./run_n100.sh
# 或分步:
# 1) FlagGems 12 算子 (kaixin env)
/root/miniconda3/envs/kaixin/bin/python bench_flaggems.py --n100 --force
# 2) Liger 6 算子 (kaixin env)
/root/miniconda3/envs/kaixin/bin/python bench_liger.py --n100 --force
# 3) attention 2 算子 (python3.10, 需 vllm + flash-attn)
python3.10 bench_attn.py --n100 --force
# 4) 矩阵乘族 5 算子 (kaixin env)
/root/miniconda3/envs/kaixin/bin/python bench_matmul.py --n100 --force
# 5) 汇总 xlsx
/root/miniconda3/envs/kaixin/bin/python make_xlsx.py
```


## 100-shape 版本补充说明 (2026-10-14)

- 每算子 **100 个 shape** (`shapes_100.py`, 种子 20261014): 尺寸阶梯 + 狭长 + 真实模型尺寸 (Llama/GPT/Qwen hidden/vocab/FFN、attention GQA/MHA/D64-256、decode/chunked-prefill/128K 长序列等)
- 大 shape 的 CE/FLCE 参考使用**分块 fp32** 计算避免大 logits 物化
- 过程中修复的三个问题 (对数据质量至关重要):
  1. **并发污染**: 曾有两个链同时跑在同一 GPU (一个残留自旧会话), 互相干扰使时延膨胀 ~80x — 已全部杀掉后单链串行重跑
  2. **attention 参考实现 bug**: Sq<Skv 时 flash/vLLM 的 causal 是 bottom-right 对齐, 参考误用 top-left `is_causal`; 且 unified 的 paged 参考未截掉块内 padding — 修复后全部 OK (kernel 本身一直正确)
  3. **flag_gems topk 编译爆炸**: 大 N(250000)×大 k(100) 组合使 ptxas 编译挂死 — topk 策略改为 N>50000 时 k≤10
- 正确性: **2500/2500 OK**; cumsum/sum 用 fp32 校验 (cumsum 相对容差 5e-4, 长行累加顺序差异)
