# B200 原生调度与优化资格审查

## Material Passport

日期：2026-10-22。状态：SOURCE_BYTES_VERIFIED_NOT_GPU_RUN。仅CPU、文件和官方源码核查；无GPU调用、无FlagGems导入、未改环境/旧benchmark/结果。

实机包为FlagGems 5.0.2。已新取并逐字节核对全部1396个.py/yaml/json源文件，1396/1396与固定官方提交`30bfcc0735b8ebd48bd7339ef752ec64df69663f`一致，差异0。不是从版本metadata或旧32e4审计推定。

## 结论

在此固定FlagGems版本及本轮25项选择中，未发现可用的B200/SM100专属实现、调度表或调优参数。`ARCH_MAP`只列8→ampere、9→hopper；major10不启用架构专属目录，不能把Hopper目录里的TMA/warp-specialization算作B200原生分支。NVIDIA的vendor-common函数又被replace_customized_ops明确跳过；仅看到_nvidia/ops/gelu.py文件不能认定它被选择。

确实应补强的是通用原生调度：group_mm的SM90+ TMA特性分支被固定nonTMA leaf绕过；MM/BMM/SDPA及FLA Conv/Group RMS也有未保留的通用搜索或形状派发。它们是公平性/基线覆盖问题，不自动成为B200专属优化证据。

`cutlass_scaled_mm_sm100`是NotImplementedError占位符；`ops/fp8_matmul.py`明文为H20固定部署配置，二者均不能当作B200权威优化路径。本轮相关候选继续要求pure Triton。

## 公共选择链（具体源码）

- [runtime/backend/_nvidia/__init__.py:1–19](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/__init__.py#L1-L19) SHA256 `84868f70b3c3b4f5fa535836c6c6989c82a1e0d2179e86014ddea9921fd27bac`
- [runtime/backend/__init__.py:25–133](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/__init__.py#L25-L133) SHA256 `a1de1ff37b7b19d6f05511eeca1fefc4ae234e1292c20937d27290b8601b5c4b`
- [runtime/__init__.py:28–43](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/__init__.py#L28-L43) SHA256 `a7ec57de11a260b6cdd19830e0f70f7fedb845977e59e78456c8c59afae0371a`
- [runtime/configloader.py:498–545](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/configloader.py#L498-L545) SHA256 `01f6ec52f4979ea70949ceff8201535feff2f2bde1bdde82a21678df7fdb8a46`
- [runtime/configloader.py:345–354](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/configloader.py#L345-L354) SHA256 `01f6ec52f4979ea70949ceff8201535feff2f2bde1bdde82a21678df7fdb8a46`
- [runtime/backend/backend_utils.py:18–39](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/backend_utils.py#L18-L39) SHA256 `0fb368582704d2db1dd3df2ef17cccb6ed94308ed10280d91badb5f7c541b985`
- [__init__.py:1–25](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/__init__.py#L1-L25) SHA256 `7c251abe042cf1a0498ae575023453fe61d6675983256490b2517e2dc39cfa2d`
- [utils/libentry.py:611–813](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/libentry.py#L611-L813) SHA256 `922dec5776e887ffe9205af9c180f530cf679d4bd140b419719b383f59f81ee0`

配置优先级为arch→vendor→default。默认自动检测major10没有arch映射，正常落到_nvidia/tune_configs.yaml及_nvidia/heuristics_config_utils.py通用数据；不是跳过整个NVIDIA配置。人为设置ARCH=8/9属于强制其他架构，不是原生SM100派发；本次不设置该变量。pointwise保留通用CodeGenConfig和形状heuristics；其major>=9 fill_scalar特判不适用于SiLU/GELU/SwiGLU/Sigmoid/ReLU。

## 25项逐项记录

### silu

原pointwise_dynamic生成器保留；使用通用NVIDIA tile/warp规则，非fill_scalar，major>=9的fill特判不适用。

本轮入口：`None`。调度归类：pointwise native retained。本轮已选B200专属路径：否。

- [ops/silu.py:12–17](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/silu.py#L12-L17) SHA `b6e9cab7ba1ff3c4b279f60a37d64695513347ca1ea4058ea42bde1363e97fce`
- [ops/silu.py:30–33](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/silu.py#L30-L33) SHA `b6e9cab7ba1ff3c4b279f60a37d64695513347ca1ea4058ea42bde1363e97fce`
- [utils/codegen_config_utils.py:158–161](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L158-L161) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/codegen_config_utils.py:78–86](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L78-L86) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/pointwise_dynamic.py:814–862](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L814-L862) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`
- [utils/pointwise_dynamic.py:864–910](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L864-L910) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`

### gelu_tanh

保留generic ops.gelu.gelu_tanh的pointwise生成；_nvidia/ops/gelu.py虽存在，但runtime对NVIDIA不替换vendor-common函数，不能说本算子自动选用了它。

本轮入口：`None`。调度归类：pointwise native retained。本轮已选B200专属路径：否。

- [ops/gelu.py:25–30](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/gelu.py#L25-L30) SHA `037d132c838350160fd5e60236d6ffbfc4af91d10782a3babce623a553cf78e3`
- [runtime/backend/_nvidia/ops/gelu.py:85–87](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/ops/gelu.py#L85-L87) SHA `55ff997f61a6aa38a2a0333fe6cb912260150f2eaf0859f489334f6b0aba9966`
- [utils/codegen_config_utils.py:158–161](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L158-L161) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/codegen_config_utils.py:78–86](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L78-L86) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/pointwise_dynamic.py:814–862](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L814-L862) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`
- [utils/pointwise_dynamic.py:864–910](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L864-L910) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`

### swiglu

FlagGems silu_and_mul的pointwise生成器保留，全FP32中间；Liger另有Blackwell宽列分支，但当前N<=8192不触发，低精度中间cast合同亦不同，见单列说明。

本轮入口：`flag_gems.fused.silu_and_mul.silu_and_mul_kernel(a,b,out0=y)`。调度归类：pointwise native retained。本轮已选B200专属路径：否。

- [fused/silu_and_mul.py:12–17](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/silu_and_mul.py#L12-L17) SHA `86d47dffff39ca1ddf9e3a19a6a95a22ce0645f8d421c741d453787e4611cc7d`
- [utils/codegen_config_utils.py:158–161](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L158-L161) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/codegen_config_utils.py:78–86](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L78-L86) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/pointwise_dynamic.py:814–862](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L814-L862) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`
- [utils/pointwise_dynamic.py:864–910](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L864-L910) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`

### rms_norm

本轮用Liger GEMMA=1 row leaf。已选M<=512，Liger原生row/block条件也会选row；FlagGems另按N<=4096选择row，否则rms_norm_loop并用通用配置。没有B200专属门槛。

本轮入口：`Liger _rms_norm_forward_kernel`。调度归类：different provider, not FlagGems dispatch。本轮已选B200专属路径：否。

- [ops/rms_norm.py:236–254](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/rms_norm.py#L236-L254) SHA `2a6c5392ef2b1a3138889ab848658dc58744ab0e6dd12b583aac61c75a4d97ef`
- [ops/rms_norm.py:21–57](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/rms_norm.py#L21-L57) SHA `2a6c5392ef2b1a3138889ab848658dc58744ab0e6dd12b583aac61c75a4d97ef`
- [ops/rms_norm.py:60–130](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/rms_norm.py#L60-L130) SHA `2a6c5392ef2b1a3138889ab848658dc58744ab0e6dd12b583aac61c75a4d97ef`

### layer_norm

本轮用Liger row forward。FlagGems另按N<=128、<=4096、>4096选multiline/persistent/loop并调通用tuner；native统计项存输入dtype，与本轮FP32统计输出不同，替换需显式新合同。

本轮入口：`Liger _layer_norm_forward_kernel`。调度归类：different provider, not FlagGems dispatch。本轮已选B200专属路径：否。

- [ops/layernorm.py:333–395](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/layernorm.py#L333-L395) SHA `00c6190b1074d151a65472486ab79411765eabb3186871e4d71fc1d4a799e4eb`
- [ops/layernorm.py:22–68](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/layernorm.py#L22-L68) SHA `00c6190b1074d151a65472486ab79411765eabb3186871e4d71fc1d4a799e4eb`

### residual_rmsnorm

本轮add→舍入到输入dtype→Liger GEMMA RMS两核。FlagGems native fused_add_rms_norm确有单核，但在FP32 x+r未舍入值上归一化且原地写x/residual，不返回FP32 rstd，不能静默替换当前舍入/副作用/统计合同。_nvidia版本也非B200专属且不被NVIDIA vendor-common自动覆盖。

本轮入口：`Two direct Triton kernels, add then norm; normalized and rounded residual outputs plus stats.`。调度归类：unfused composition, no native fused dispatch。本轮已选B200专属路径：否。

- [fused/fused_add_rms_norm.py:14–54](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/fused_add_rms_norm.py#L14-L54) SHA `2b7a69a53e5fb4ddc862172366bac3221420c8a0f5cbff50c1dae7be4d8d2b59`
- [fused/fused_add_rms_norm.py:57–82](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/fused_add_rms_norm.py#L57-L82) SHA `2b7a69a53e5fb4ddc862172366bac3221420c8a0f5cbff50c1dae7be4d8d2b59`
- [runtime/backend/_nvidia/fused/fused_add_rms_norm.py:14–47](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/fused/fused_add_rms_norm.py#L14-L47) SHA `53e39da81f5636cd733b4a31a29ae1ffc4f0a727d4f405f5fdeed8dc10196456`

本轮固定launch：`{"add_BLOCK_SIZE": 1024, "norm_BLOCK_SIZE": 64, "norm_num_warps": 4}`。

### residual_layernorm

本轮add→输入dtype舍入→Liger LayerNorm。FlagGems归档没有同名fused residual LayerNorm；不能把普通layer_norm称作同合同原生融合/B200优化。

本轮入口：`Two direct Triton kernels, add then norm; normalized and rounded residual outputs plus stats.`。调度归类：unfused composition, no native fused dispatch。本轮已选B200专属路径：否。

- [ops/layernorm.py:333–395](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/layernorm.py#L333-L395) SHA `00c6190b1074d151a65472486ab79411765eabb3186871e4d71fc1d4a799e4eb`

本轮固定launch：`{"add_BLOCK_SIZE": 1024, "norm_BLOCK_SIZE": 64, "norm_num_warps": 4}`。

### rope

本轮Liger原函数加两次Triton copy恢复输出；FlagGems另有普通rotary_embedding，按inplace/layout/头维填充派发，不含B200门槛；公开cos/sin半维布局须适配，不能假定无成本替换。

本轮入口：`Two exact Triton copy kernels + one exact in-place rotation kernel; refresh is inside every timed call.`。调度归类：different provider, not FlagGems dispatch。本轮已选B200专属路径：否。

- [fused/rotary_embedding.py:181–312](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/rotary_embedding.py#L181-L312) SHA `fc2c8cb79177b0cbd0d7cc48d9a13feb02c994763dd7eb591c324211a314b84e`
- [fused/rotary_embedding.py:15–101](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/rotary_embedding.py#L15-L101) SHA `fc2c8cb79177b0cbd0d7cc48d9a13feb02c994763dd7eb591c324211a314b84e`

本轮固定launch：`{"grid": [16], "pad_n_qh": 4, "pad_n_kh": 2, "pad_hd": 16}`。

### group_rmsnorm

本轮FLA row leaf固定warps。所归档FLA原生D<=512通常选BT多行核，其他选row且有通用warps autotune；因此确实绕过形状调度/通用调优，但不是绕过B200专用配置。FlagGems没有相同group-weight合同的独立group RMS入口。

本轮入口：`One direct FLA row Triton RMS kernel; stats included.`。调度归类：different provider; fixed leaf bypasses FLA shape dispatch。本轮已选B200专属路径：否。

- [ops/rms_norm.py:236–254](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/rms_norm.py#L236-L254) SHA `2a6c5392ef2b1a3138889ab848658dc58744ab0e6dd12b583aac61c75a4d97ef`

本轮固定launch：`{"BD": 64, "num_warps": 4}`。

### group_gemm

本轮固定non-TMA grouped_mm_kernel、tile128x64x32、GROUP_M4、warp4/stage4、min(tilecount,148) grid。native group_mm在capability major>=9且host TensorDescriptor可用时选TMA核、grid=实际SM数，原7配置和descriptor pre-hook。是应补测的SM90+硬件特性路径，非SM100专属。

本轮入口：`flag_gems.ops.group_gemm.grouped_mm_kernel non-TMA JIT leaf`。调度归类：BYPASS: native SM90+ TMA and generic 7-config tuner。本轮已选B200专属路径：否。

- [ops/group_gemm.py:12–13](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L12-L13) SHA `863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1`
- [ops/group_gemm.py:56–100](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L56-L100) SHA `863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1`
- [ops/group_gemm.py:47–53](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L47-L53) SHA `863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1`
- [ops/group_gemm.py:563–614](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L563-L614) SHA `863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1`
- [ops/group_gemm.py:311–382](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L311-L382) SHA `863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1`
- [ops/group_gemm.py:385–487](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L385-L487) SHA `863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1`

本轮固定launch：`{"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4, "num_warps": 4, "num_stages": 4, "grid": [2]}`。

### causal_conv1d

本轮FLA原核固定BD/warps，绕过FLA BD16/32/64/128 × NVIDIA warps4/8/16/32搜索及功能heuristics。没有B200特判；FlagGems conv1d是普通卷积经conv2d适配，并非现有causal depthwise [B,T,D]合同入口。

本轮入口：`Exact FLA causal_conv1d_fwd_kernel body, explicit launch metadata`。调度归类：different provider; fixed leaf bypasses generic FLA tuning。本轮已选B200专属路径：否。

- [ops/conv1d.py:9–65](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/conv1d.py#L9-L65) SHA `e1843f33a5b90cf303edcc46a1f8bda71851c1c093c299ce47e59905ebff9423`

本轮固定launch：`{"D": 32, "W": 3, "BT": 64, "BW": 4, "BD": 64, "ACTIVATION": null, "HAS_WEIGHT": true, "HAS_BIAS": true, "HAS_RESIDUAL": false, "USE_INITIAL_STATE": false, "IS_VARLEN": false, "num_warps": 4, "num_stages": 3}`。

### sdpa

本轮直接_attn_fwd JIT leaf固定BM64/BN32/warp4/stage2，绕过通用attention配置和small-head扩展搜索；QK/PV原生allow_tf32=False保留。没有TMA/SM100特判。本轮保持direct pure Triton，不能改调C-extension或其他CUDA fallback。

本轮入口：`flag_gems.ops.attention._attn_fwd JIT leaf with _attn_fwd_inner dependency`。调度归类：BYPASS: generic attention autotuning, not B200 dispatch。本轮已选B200专属路径：否。

- [ops/attention.py:146–329](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/attention.py#L146-L329) SHA `4d1f989252e2b1a4e31d8ad0cab3773c734caa039890bbe39c7631a696cbf8a2`
- [ops/attention.py:760–861](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/attention.py#L760-L861) SHA `4d1f989252e2b1a4e31d8ad0cab3773c734caa039890bbe39c7631a696cbf8a2`
- [ops/attention.py:130–150](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/attention.py#L130-L150) SHA `4d1f989252e2b1a4e31d8ad0cab3773c734caa039890bbe39c7631a696cbf8a2`

本轮固定launch：`{"GROUP_HEAD": 1, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "STAGE": 1, "HAS_ATTN_MASK": false, "PRE_LOAD_V": false, "num_warps": 4, "num_stages": 2}`。

### mrope

本轮Liger Qwen2VL mRoPE原函数+两次copy，所选源无B200分支；与普通FlagGems RoPE不能混称同一算子。FP16缓存合同已在独立修正批次固定。

本轮入口：`Two exact Triton copy kernels + one exact in-place rotation kernel; refresh is inside every timed call.`。调度归类：different provider, no FlagGems mRoPE dispatch。本轮已选B200专属路径：否。

- [fused/rotary_embedding.py:181–312](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/rotary_embedding.py#L181-L312) SHA `fc2c8cb79177b0cbd0d7cc48d9a13feb02c994763dd7eb591c324211a314b84e`

本轮固定launch：`{"grid": [16], "pad_n_qh": 4, "pad_n_kh": 2, "pad_hd": 16}`。

### kv_cache_store

本轮直接reshape_and_cache_flash_kernel固定warp4、每token一个program。原公开入口按use_c_extension选Torch扩展或Triton；这种后端切换不是B200优化。无B200专属配置，维持pure Triton叶核是合理合同。

本轮入口：`flag_gems.fused.reshape_and_cache_flash.reshape_and_cache_flash_kernel direct Triton leaf, avoids optional C extension`。调度归类：pure Triton leaf; bypasses optional C extension。本轮已选B200专属路径：否。

- [fused/reshape_and_cache_flash.py:61–112](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/reshape_and_cache_flash.py#L61-L112) SHA `9855fd04813a3bf0538c8954e54692a618bf5068cb498f1b0fbbdbf74d435d84`
- [fused/reshape_and_cache_flash.py:14–58](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/reshape_and_cache_flash.py#L14-L58) SHA `9855fd04813a3bf0538c8954e54692a618bf5068cb498f1b0fbbdbf74d435d84`

### softmax

本轮直接decorated softmax_kernel_inner，未剥离heuristics；对本grid的K=1与公开入口同分支。TILE_N=nextpow2(N)（<=32768）、warps按tile阈值4/8/16。不是调用yaml中同名backward autotuner，也非B200调优。

本轮入口：`None`。调度归类：native inner heuristics retained。本轮已选B200专属路径：否。

- [ops/softmax.py:284–328](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/softmax.py#L284-L328) SHA `ddcb1a2473a11a68e32665c2c67ac1d7c1c1b3b811be7076b063ebb187f6bfea`
- [ops/softmax.py:88–160](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/softmax.py#L88-L160) SHA `ddcb1a2473a11a68e32665c2c67ac1d7c1c1b3b811be7076b063ebb187f6bfea`
- [runtime/backend/_nvidia/heuristics_config_utils.py:253–257](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py#L253-L257) SHA `3b487b9ccdd612ec18fe98ffb34db954d843c08cbea4e30ee4fe557a42ea8737`
- [runtime/backend/_nvidia/heuristics_config_utils.py:260–267](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py#L260-L267) SHA `3b487b9ccdd612ec18fe98ffb34db954d843c08cbea4e30ee4fe557a42ea8737`

### log_softmax

本轮BM8/BN256/warp8匹配公开forward默认；yaml log_softmax autotune只装饰backward，不应误写成forward绕过已调优搜索。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：same generic forward defaults。本轮已选B200专属路径：否。

- [ops/log_softmax.py:15–55](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/log_softmax.py#L15-L55) SHA `0dfb3b80530050b206e6a21f19142be957727575e1e3d3af13dd1e4c9751cb66`
- [ops/log_softmax.py:98–128](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/log_softmax.py#L98-L128) SHA `0dfb3b80530050b206e6a21f19142be957727575e1e3d3af13dd1e4c9751cb66`
- [ops/log_softmax.py:58–95](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/log_softmax.py#L58-L95) SHA `0dfb3b80530050b206e6a21f19142be957727575e1e3d3af13dd1e4c9751cb66`

### embedding

每token一program、BLOCK_SIZE=nextpow2(H)，与原公开forward同一直接gather核。无B200配置。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：same generic forward launch。本轮已选B200专属路径：否。

- [ops/embedding.py:14–33](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/embedding.py#L14-L33) SHA `58ac369551bf30ba6bdc82145985a10914e5f033d67fa5ede06ac7e23b148e9a`
- [ops/embedding.py:113–129](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/embedding.py#L113-L129) SHA `58ac369551bf30ba6bdc82145985a10914e5f033d67fa5ede06ac7e23b148e9a`

### cumsum

FP16/FP32选N<=16384的native persistent row leaf及原warp规则；large-N才用实际SM数限制多阶段CTA，既有grid不触发。FP8为明确FP32累加的scan_part_sum_abc_kernel、part_num1另行合同，非原生默认row精度冒充。均非B200特判。

本轮入口：`reduce_then_scan_root_scan_kernel_row, original N<=16384 row path`。调度归类：native small-row path retained; alternate FP8 FP32-accumulator leaf。本轮已选B200专属路径：否。

- [ops/cumsum.py:212–239](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/cumsum.py#L212-L239) SHA `84cef7e5c350f9cc2ab73a8ab05cf3419684f3dced0f7c60b307da33f4960300`
- [ops/cumsum.py:242–296](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/cumsum.py#L242-L296) SHA `84cef7e5c350f9cc2ab73a8ab05cf3419684f3dced0f7c60b307da33f4960300`
- [ops/cumsum.py:326–335](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/cumsum.py#L326-L335) SHA `84cef7e5c350f9cc2ab73a8ab05cf3419684f3dced0f7c60b307da33f4960300`
- [ops/cumsum.py:93–136](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/cumsum.py#L93-L136) SHA `84cef7e5c350f9cc2ab73a8ab05cf3419684f3dced0f7c60b307da33f4960300`

### groupnorm

预分配后直接group_norm_kernel，与公开forward的BLOCK_GROUP_SIZE/HW_SIZE=nextpow2一致。无B200搜索或硬件特判；统计输出真实输入dtype仍单独核对。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：same generic forward launch。本轮已选B200专属路径：否。

- [ops/groupnorm.py:184–213](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/groupnorm.py#L184-L213) SHA `de1832c6d1540429d6138bb82e74f7fa69c4bb76a14f9a6da42f7f079f47bc1d`
- [ops/groupnorm.py:15–70](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/groupnorm.py#L15-L70) SHA `de1832c6d1540429d6138bb82e74f7fa69c4bb76a14f9a6da42f7f079f47bc1d`

### sigmoid

原pointwise_dynamic保留；FP32 exp2向量计算可使用真实FP8存储，但不是FP8 TensorCore计算或B200专属实现。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：pointwise native retained。本轮已选B200专属路径：否。

- [ops/sigmoid.py:12–19](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sigmoid.py#L12-L19) SHA `11a5e2e7777a5600fdedc135295171f6d38b9e9fada0fe6a4bca54615f8d864b`
- [ops/sigmoid.py:30–33](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sigmoid.py#L30-L33) SHA `11a5e2e7777a5600fdedc135295171f6d38b9e9fada0fe6a4bca54615f8d864b`
- [utils/codegen_config_utils.py:158–161](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L158-L161) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/codegen_config_utils.py:78–86](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L78-L86) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/pointwise_dynamic.py:814–862](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L814-L862) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`
- [utils/pointwise_dynamic.py:864–910](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L864-L910) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`

### sum_reduce

显式dim=-1、K=1，直接decorated sum_dim_kernel_inner保留softmax_inner heuristics，与公开dim分支一致；没有绕过B200分支。dim=None两阶段全归约是不同合同，未拿来替代。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：native inner heuristics retained。本轮已选B200专属路径：否。

- [ops/sum.py:241–326](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sum.py#L241-L326) SHA `3e4dc0dbf86349d1cc6d069f7a60ec793e7070d3e5107330acf04e9c5084583d`
- [ops/sum.py:154–199](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sum.py#L154-L199) SHA `3e4dc0dbf86349d1cc6d069f7a60ec793e7070d3e5107330acf04e9c5084583d`
- [ops/sum.py:329–364](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sum.py#L329-L364) SHA `3e4dc0dbf86349d1cc6d069f7a60ec793e7070d3e5107330acf04e9c5084583d`

### max_pool2d

原12配置triton.autotune保留；既有性能合同是有限值NCHW2x2/stride2，验证首个并列最大值索引；NaN/全负无穷语义边界不能用硬件优化名义掩盖。无B200配置。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：native generic 12-config autotuner retained。本轮已选B200专属路径：否。

- [ops/max_pool2d_with_indices.py:34–127](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/max_pool2d_with_indices.py#L34-L127) SHA `8d3ebd0e2bdfe3c4868590bc62c000de57162165f3d3b32245869b83d07a0491`
- [ops/max_pool2d_with_indices.py:260–329](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/max_pool2d_with_indices.py#L260-L329) SHA `8d3ebd0e2bdfe3c4868590bc62c000de57162165f3d3b32245869b83d07a0491`

### relu

原pointwise_dynamic保留，通用NVIDIA tile/warp规则；有限值FP8 compare/select/storage不能归为TensorCore FLOPs。无B200专属分支。

本轮入口：`preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn`。调度归类：pointwise native retained。本轮已选B200专属路径：否。

- [ops/relu.py:11–14](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/relu.py#L11-L14) SHA `64a509380f6c7ba83d14eedb45caa1f1797e367c880e488e692d5679ea4aff0b`
- [ops/relu.py:23–26](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/relu.py#L23-L26) SHA `64a509380f6c7ba83d14eedb45caa1f1797e367c880e488e692d5679ea4aff0b`
- [utils/codegen_config_utils.py:158–161](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L158-L161) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/codegen_config_utils.py:78–86](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/codegen_config_utils.py#L78-L86) SHA `28c4017fbdb9a8ed479adf2b43c38fa97d851082849b5c1e42c7329015ad9554`
- [utils/pointwise_dynamic.py:814–862](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L814-L862) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`
- [utils/pointwise_dynamic.py:864–910](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/utils/pointwise_dynamic.py#L864-L910) SHA `517937979ab26bfcb6d8a22ea20eb5707f08c42cd70f1de64e46b3f57f4990b1`

### dense_matmul

固定mm_kernel_general leaf绕过通用mm yaml autotuner；独立连续A/B不是SYRK，StreamK gate明确major==8，在SM100不会选，故不能称漏掉B200 StreamK。Hopper专用mm未由arch10派发。

本轮入口：`flag_gems.ops.mm.mm_kernel_general JIT leaf; bypasses public SYRK/StreamK branches`。调度归类：BYPASS: generic mm tuner; SM80 StreamK inapplicable。本轮已选B200专属路径：否。

- [ops/mm.py:25–107](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/mm.py#L25-L107) SHA `b2a5eb649a6c2ae251f340130249b3b4b5dd862bbcd06df0c3d1349043bfd91c`
- [ops/mm.py:127–156](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/mm.py#L127-L156) SHA `b2a5eb649a6c2ae251f340130249b3b4b5dd862bbcd06df0c3d1349043bfd91c`
- [ops/mm.py:270–283](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/mm.py#L270-L283) SHA `b2a5eb649a6c2ae251f340130249b3b4b5dd862bbcd06df0c3d1349043bfd91c`
- [ops/mm.py:286–309](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/mm.py#L286-L309) SHA `b2a5eb649a6c2ae251f340130249b3b4b5dd862bbcd06df0c3d1349043bfd91c`
- [ops/mm.py:312–330](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/mm.py#L312-L330) SHA `b2a5eb649a6c2ae251f340130249b3b4b5dd862bbcd06df0c3d1349043bfd91c`

本轮固定launch：`{"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 3}`。

### batched_matmul

固定bmm JIT leaf，人工DIVISIBLE_M/N/K等价形状判断，但绕过原生12配置LibTuner和可选USE_FLAGTUNE扩展搜索；不存在B200专属配置。

本轮入口：`flag_gems.ops.bmm.bmm_kernel JIT leaf`。调度归类：BYPASS: generic bmm tuner/heuristic wrappers。本轮已选B200专属路径：否。

- [ops/bmm.py:16–138](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/bmm.py#L16-L138) SHA `7ed3048cc3b4960b5890fd0df98a6ec44d0655719d773243a347acc2ed1c9cb6`
- [ops/bmm.py:141–172](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/bmm.py#L141-L172) SHA `7ed3048cc3b4960b5890fd0df98a6ec44d0655719d773243a347acc2ed1c9cb6`
- [runtime/backend/_nvidia/heuristics_config_utils.py:109–110](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py#L109-L110) SHA `3b487b9ccdd612ec18fe98ffb34db954d843c08cbea4e30ee4fe557a42ea8737`

本轮固定launch：`{"TILE_M": 64, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 1, "DIVISIBLE_M": false, "DIVISIBLE_N": false, "DIVISIBLE_K": true, "num_warps": 4, "num_stages": 3}`。

## 非FlagGems实际来源

下列固定来源是实际baseline选择，不能套用FlagGems5.0.2的包资格。其完整源码SHA和原函数行号逐项保留于audit.json。

- rope: [43293d84d760e402e4fd974b12410a1d47c31761](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/rope.py#L7-L112)，SHA `bfeec91d1aeffb9226a113a4a4a7e5df45b0588e2a4dadad025ef341e0789a5e`。
- mrope: [43293d84d760e402e4fd974b12410a1d47c31761](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/qwen2vl_mrope.py#L7-L107)，SHA `5901fdeca8ad4e5b4836cb4a918226203a9fe7be8c309290c4d02d4be16154d3`。
- rms_norm: [43293d84d760e402e4fd974b12410a1d47c31761](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/rms_norm.py#L68-L150)，SHA `5f725509cafc9273ea1db1655a21ef958ef6b02affb0e8f9d25f4de1efb3dcd4`。
- layer_norm: [43293d84d760e402e4fd974b12410a1d47c31761](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/layer_norm.py#L27-L86)，SHA `b4b8ee6f672344b90f79e0cc462ca141be07fa54604fd8126b60e3a0dbc22bb6`。
- group_rms: [9d478facb282410dc150051d5f57205cd3812f31](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/norm/layernorm.py#L123-L179)，SHA `525ec77fdd5fe108d1980d10e7a14d97be8090e159574ef389e186292abf48f5`。
- copy: [32e4a4cf9b8618185a13eb82d72687fcdbbfb982](https://github.com/flagos-ai/FlagGems/blob/32e4a4cf9b8618185a13eb82d72687fcdbbfb982/src/flag_gems/ops/alias_copy.py#L28-L34)，SHA `6849851d31466d684b3d2d4c72b1309c2e709581d2bf068fceeff0c55ce8c3ad`。
- add: [c817b9b63d40ead1ed023b7663f5ea14f676f4bc](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/python/tutorials/01-vector-add.py#L30-L54)，SHA `842430949e0ccde4fbce07606cce3ac4bac36bf21b2b12619a31b795ca4029b3`。
- causal_conv1d: [9d478facb282410dc150051d5f57205cd3812f31](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/causal_conv1d/ops.py#L34-L147)，SHA `9b59b5004b843bdface81164c0850c0b0e791df688f046fcee1c1810e06f7b16`。

## 真正Blackwell分支的边界

实机Liger swiglu.py、ops/utils.py、utils.py另行新取，3/3与官方43293d84d760e402e4fd974b12410a1d47c31761字节一致。其明确Blackwell二维列分块路径要求nextpow2(N)>=16384，tile1024、warp4；现有N<=8192计划的100个shape全部不触发。低精度SiLU先cast到b.dtype再乘，与当前FP32中间合同不同。可在新增宽形状的FP32独立候选中保留同一数学合同，不能据此回填当前100shape为已做B200优化。

来源：[固定Liger源码](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/swiglu.py#L20-L41)，新字节证据见liger_alternative_byte_verification.json。

## 已准备、尚未运行的候选

`native_dispatch_candidates.py`提供shape_plan/build_case，复用原matrix_attention_ops CPU输入/FP64参考/容差/预分配输出，三个算子均100唯一原计划形状：MM原生LibTuner、BMM原生LibTuner+heuristics、Group GEMM原生SM90+ TMA 7配置及descriptor pre-hook。不修改kernel源码、不剥离.fn、不在fn内分配或执行Torch原生GPU计算。所有缓存环境必须在papers2目录，首次正确性调用时才允许root独占设备编译/调优。

当前仅AST/import/shape-plan检查，未调用build_case，没有编译、数值检查、延迟或优劣结论。root应先串行两shape×dtype gate，记录实际选中配置、编译IR和FP32 allow_tf32=False，再决定是否全量；失败必须保留。新结果应另存批次，不覆盖已导出基线。

`liger_blackwell_swiglu_candidate.py`另提供FP32-only原生dispatcher/settings候选，冻结新计划M=[1,2,4,8,16,32,64,128,256,512] × N=[64,128,256,512,1024,4096,8192,14336,16384,28672]。100唯一，预期Blackwell下30 tiled/70 row；0/99烟测分别覆盖两路径。计划SHA a234f60ef926cdcedd17577e12befe6bb0a0e47fcbf4639c3f6ee0f0fda947d6；每例记录_should_tile、架构及kernel路径。FP16/FP8因中间舍入不同明确排除。该候选没有autotuner，不虚构搜索。没有运行GPU；A5必须用同一新计划/seed/合同另测，不能按case_index将不同旧形状相除。

## 证据归档

- 实机源码压缩包SHA：`9b748c9d9ab1dfd2b18276e8f019956a6a945b1856b9a78b4e58b723adaa9bb3`。
- 官方固定提交压缩包SHA：`2ac3828157b0383220f32953e91c14603726053874c4e89fb68142a1b5ce2fcf`。
- 每个源文件的实机路径、字节、SHA、固定commit URL与一致性：source_byte_verification.json。
- Apache-2.0许可证：official/LICENSE及实机metadata归档。
- audit.json包含25项完整来源和实际baseline入口/固定launch；package_hardware_keyword_hits保留硬件关键字命中，避免把命名/兼容性误写成有效优化。
- 本报告不声称25项均有B200专属最佳实现；也不从SM100 target、TensorCore指令或可成功运行推导专属调优。
