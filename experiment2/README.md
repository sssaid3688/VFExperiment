# experiment2 — B200 基准 25 算子的昇腾(A5/950PR)开源适配代码冒烟测试

> 日期: 2026-10-15 | 设备: **NPU:0 = Ascend950PR_9579 (A5)**（全程锁定卡 0）
> 环境: `/home/oyhj/venv_cann` (torch 2.7.1 + torch_npu 2.7.1.post10 + **triton-ascend 3.2.2**) + CANN 9.1.0-beta.3 (`~/Ascend/ascend-toolkit/set_env.sh`)
> 结论: **21 / 25 跑通（含正确性校验），4 个跑不通**（均为上游代码/编译器问题，未改任何环境）

## 一、目录结构

```
experiment2/
├── run_smoke.sh          # 一键运行: venv_cann + set_env.sh + 卡0 + GEMS_VENDOR=ascend
├── smoke_test.py         # 25 算子冒烟: 小 shape + fp32 参考对照(阈值 2e-2, 同 B200 口径)
├── debug_failures.py     # 失败项定向诊断脚本 (sum/rope/bmm/FA2)
├── debug_rope_probe.py   # rope 输出语义探针
├── debug_rope_matrix.py  # rope 多配置矩阵验证
├── glue/vllm/            # vllm v0.24.0 triton_unified_attention 的最小 shim
│   └── v1/attention/ops/triton_unified_attention.py   (上游原文件拷贝)
│   └── v1/attention/ops/triton_attention_helpers.py   (上游原文件拷贝)
│   └── envs.py / logger.py / platforms.py / triton_utils.py / v1/kv_cache_interface.py (桩)
├── third_party/
│   ├── FlagGems-5.0.2/            # tar.gz 解压, 含 runtime/backend/_ascend 昇腾后端
│   ├── Liger-Kernel-0.8.4/        # tar.gz 解压, 含 ops/backends/_ascend 昇腾后端
│   ├── triton-ascend-v3.2.2/      # sparse clone, 只取 python/tutorials (与本机版本匹配)
│   └── triton-ascend-tutorials/   # main 分支 06-fused-attention.py (参考, 3.2.2 上编译不过)
└── results/               # smoke_<时间戳>.csv / .log
```

运行: `cd experiment2 && ./run_smoke.sh`

## 二、冒烟结果（21 PASS / 4 FAIL）

| # | 算子 | 来源 | 结果 | 备注 |
|---|---|---|---|---|
| 2 | flash_attention | triton-ascend v3.2.2 tutorial 06 (FA2) | **PASS** | fp16 通过；bf16 编译失败(见注①) |
| 3 | softmax | FlagGems `_ascend` | **PASS** | max_rel 3.9e-3 |
| 4 | silu | FlagGems 通用 | **PASS** | |
| 5 | rms_norm | Liger `_ascend`(910B 调优) | **PASS** | UB 检测经 tbe get_soc_spec 正常 |
| 6 | mm | FlagGems `_ascend` | **PASS** | autotune 经内存桩正常 |
| 7 | layer_norm | Liger `_ascend` | **PASS** | |
| 8 | gelu | FlagGems 通用 | **PASS** | |
| 9 | add | FlagGems 通用 | **PASS** | |
| 10 | mul | FlagGems 通用 | **PASS** | |
| 12 | argmax | FlagGems `_ascend` | **PASS** | 精确匹配 |
| 13 | cross_entropy | Liger `_ascend` | **PASS** | |
| 14 | fused_add_rms_norm | Liger `_ascend` | **PASS** | Y 与 residual 均正确 |
| 15 | swiglu | Liger `_ascend` | **PASS** | |
| 16 | rope | Liger `_ascend` | **PASS** | cos/sin 需**半宽表 (1,S,D/2)**(NeoX 语义, 见注②) |
| 17 | clamp | FlagGems 通用 | **PASS** | 精确匹配 |
| 18 | cumsum | FlagGems `_ascend` | **PASS** | |
| 20 | embedding | FlagGems `_ascend` | **PASS** | 精确匹配 |
| 22 | addmm | FlagGems `_ascend` | **PASS** | |
| 23 | mv | FlagGems 通用 | **PASS** | |
| 24 | fp8_matmul | FlagGems 通用 | **PASS** | **950PR 的 fp8 tl.dot 可用**(A2/A3 无此硬件) |
| 25 | fused_linear_cross_entropy | Liger `_ascend` | **PASS** | |
| 1 | unified_attention | vllm v0.24.0 | **FAIL** | 见下方根因 1 |
| 11 | sum | FlagGems 通用 | **FAIL** | 见下方根因 2 |
| 19 | topk | FlagGems 通用 | **FAIL** | 见下方根因 3 |
| 21 | bmm | FlagGems `_ascend` 覆盖版 | **FAIL** | 见下方根因 4（有绕过方案） |

## 三、4 个跑不通的根因（均未改环境、未改上游代码）

1. **#01 unified_attention（vllm v0.24.0 Triton 内核）**
   `triton_attention_helpers.py` 的 `compute_tile_loop_bounds` 使用链式布尔
   `A or B or C`，**triton-ascend 3.2.2 的 AST 前端不支持**（要求加括号拆链）。
   解决途径（二选一，均属后续工作）：改上游内核把 `or` 链加括号；或升级
   triton-ascend ≥3.3（需动环境，本轮未做）。A2/A3 用 3.2.2 同样受限。

2. **#11 sum（FlagGems v5.0.2 通用 `ops/sum.py`，无 `_ascend` 覆盖）**
   fp32、归约维较大时**部分行结果错误**：(4,128) 正确；(512,4096) 前几行正确、
   整体 max_abs≈5e+5（两段归约 mid 缓存相关，属内核在 950PR/triton-ascend
   3.2.2 上的正确性 bug）。bf16 输入时误差可过阈值，但 B200 基准的 sum 正确性
   口径是 fp32，故记 FAIL。

3. **#19 topk（FlagGems v5.0.2 通用 `ops/topk.py`）**
   位排(bitonic sort)内核 BiShengIR 编译失败：
   `'builtin.unrealized_conversion_cast' op Unsupported user for root alloc op`
   （`_bitonic_merge`/`argsort` 内的 `tl.sort` 区域）→ 编译器限制。

4. **#21 bmm（FlagGems v5.0.2 `_ascend/ops/bmm.py` 覆盖版）**
   大 shape 如 (16,256,512)×(16,512,128) 输出 NaN/大偏差（非确定，两次运行分别
   NaN 与 max_abs≈103）。**同 shape 走通用(非 `_ascend`)内核正确**（max_abs≈0.25,
   bf16 正常误差）→ 是 vendored `_ascend` bmm 的 bug；绕过方案：使用
   `flag_gems.ops.bmm` 模块内的通用版（小 shape 两版均正确）。

## 四、注记

- **注① FA2 dtype**：v3.2.2 教程内核在非 fp8 路径硬编码 `p.to(tl.float16)`，
  bf16 输入时 `tl.dot(p_fp16, v_bf16)` 报 `Both operands must be same dtype`。
  故 FA2 目前只能 fp16 跑（与 B200 的 bf16 口径有差异，需注意）。
  main 分支教程基于 Triton 3.6 API（`tensor_descriptor`/`set_allocator`/
  `get_active_torch_device`），在 3.2.2 上无法编译。
- **注② Liger `_ascend` rope 约定**：cos/sin 传**半宽表 (1,S,D/2)**（内核只加载
  前 D/2），数学上与 HF 全宽 NeoX 表 `cat([freqs,freqs])` 等价；全配置矩阵
  （含 GQA 8:2、S=256、D=128）实测误差 ≤0.015（bf16 噪声级）。
- **缺失可选依赖的桩（不装包、不改环境）**，都写在 `smoke_test.py::_install_stub_modules`：
  - `transformers.utils.is_torch_npu_available` → `torch.npu.is_available()`
    （Liger 检测 NPU 用，venv 没装 transformers）
  - `flag_gems.utils.models`（sqlalchemy 持久化层）→ 内存 dict 模型
    （flag_gems autotune "先存后读"契约保持，只是不落盘）
  - `pytest`（教程文件 import 用）→ 仅 `mark.parametrize` 恒等桩
- **flag_gems 导入模式**：full-import 成功（GEMS_VENDOR=ascend 直选 `_ascend` 后端）。
- 环境**零改动**：未 pip install 任何包，未动 CANN/venv；所有适配都在
  experiment2 的胶水层（sys.path + sys.modules 桩 + run_smoke.sh 环境变量）。

## 五、结果文件

- 最新一轮: `results/smoke_20261015_163137.csv`（含每个 case 的完整报错栈）
- 历史轮次（迭代过程）: `results/smoke_20261015_16*.csv/.log`
