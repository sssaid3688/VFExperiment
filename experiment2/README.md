# Experiment 2 — NVIDIA B200 常用算子性能基准 (20 算子 × 20 shape)

## 产出

- **`B200_ops_benchmark.xlsx`** — 全部性能数据。Sheet `总览` 为实验信息与算子清单；其余 **20 个 sheet 每算子一个**，每 sheet 20 行 shape，包含：
  - shape / 维度明细 / dtype / 元素量 / FLOPs / Bytes
  - **时延 (mean/min/p50, us)**、**TFLOPS**、**带宽 GB/s**、TFLOPS 峰值占比 %、带宽峰值占比 %
  - 相对 PyTorch 参考实现的误差与正确性状态、来源仓库
- `results/*.json` — 每 shape 一条 JSONL 原始记录（crash-safe）
- `bench_flaggems.py` / `bench_liger.py` / `bench_attn.py` — 基准脚本
- `common.py`（计时与指标）/ `shapes.py`（20×20 shape 配置）/ `make_xlsx.py`（汇总）

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

运行环境：
- **python3.10**（unified_attention + flash_attention）：torch 2.11.0+cu130、vllm 0.24.0、flash-attn 2.8.3
- **py3.11 kaixin conda env**（其余 18 个）：torch 2.11.0+cu128、triton 3.6.0、FlagGems 5.0.2、Liger-Kernel 0.8.4

## 方法

- GPU：NVIDIA B200 (SM100.0, 183GB HBM3e)；dtype：bf16
- 计时：CUDA Event 单次计时，每次 rep 前写零 256MB buffer **清空 L2**；预热 ≥5；自适应迭代 25~300 次（每 shape 约 0.4s）
- TFLOPS = FLOPs 模型 / mean 时延；GB/s = Bytes 模型 / mean 时延；峰值参考 2250 TFLOPS (BF16 dense) / 8000 GB/s (HBM3e)
- 注意力 FLOPs：causal (bottom-right 对齐) 精确 pair 数 = Sq·(Skv−Sq) + Sq(Sq+1)/2，每 pair 4·Hq·D
- 正确性：与 PyTorch eager/fp32 参考比对，相对误差 ≤ 2e-2 判 OK（bf16 输出舍入约 0.4%）；**400/400 全部 OK**
- 全部基准**串行**运行（曾发现并行运行互相干扰 GPU 导致数据失真，已全部串行重跑）

## Shape 设计（每算子 20 个）

XS/S/M/L/XL 大中小全覆盖，并包含**狭长**形态：
- 逐元素/归约类：`(1, 1M)` 单行超宽、`(1M, 64)` 超高窄列、`(524288, 128)` 等
- mm：`(1, K, N)` GEMV、`(131072, 64, 64)` 超高窄、`(64, 131072, 64)` 超长 K、`(64, 4096, 65536)` 超宽 N；Llama 实际尺寸 (4096,4096,11008) 等
- attention：128K 长序列 (1,131072,131072,16,4,64)、decode (128,1,1024)、单 token 32K 上下文、D=256 大头维、GQA/MHA
- CE：Llama-2/3/GPT-2/Qwen 词表 (32000/128256/50257/152064)；topk：k=1~100；embedding：大词表 (250000, 1024)

## 关键结论（详见 xlsx）

- 计算受限：**mm 峰值 1074 TFLOPS**（48% 峰值）、flash_attention 434 TFLOPS、unified_attention 162 TFLOPS（Triton 注意力与 CUDA FA2 在 Blackwell 上差距明显）
- 带宽受限：融合算子最优——swiglu 7073 GB/s、fused_add_rms_norm 6931 GB/s、cumsum 6701 GB/s、sum 6216 GB/s（≈ 85% HBM3e 峰值）
- 小 shape（<1MB）全部受 kernel 启动开销支配（~5-15us 地板）

## 已知上游限制

- Liger-Kernel v0.8.4 rope 在 **B>1 且 S=1**（纯 decode batch）时 transpose 后 stride 语义错误（q.stride(1)=D 而非 H·D），正确性校验失败；decode 档改用 **S=2 投机解码 verify** shape 并在总览 sheet 注明
- FlagGems 对 SM100 无专属调优配置（日志提示 "Unsupported GPU arch specialization"），性能为 Triton 默认调度结果

## 复现

```bash
cd /home/ubuntu/workspace/oyhj/papers2/experiment2
# 1) FlagGems 12 算子 (kaixin env)
/root/miniconda3/envs/kaixin/bin/python bench_flaggems.py --force
# 2) Liger 6 算子 (kaixin env)
/root/miniconda3/envs/kaixin/bin/python bench_liger.py --force
# 3) attention 2 算子 (python3.10, 需 vllm + flash-attn)
python3.10 bench_attn.py --force
# 4) 汇总 xlsx
/root/miniconda3/envs/kaixin/bin/python make_xlsx.py
```
