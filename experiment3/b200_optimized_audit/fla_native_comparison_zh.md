# FLA原生Triton候选：与旧固定叶核的对照

## Material Passport

日期：2026-10-22。状态：CPU_SOURCE_ONLY_PREPARED_NOT_GPU_VALIDATED。仅来源、AST、形状及接口核查；未构造GPU case，未编译、验算或计时。

两个候选放在新文件fla_native_candidates.py，旧benchmark、100形状、已有测量结果和软件环境均未修改。实机未安装fla包，因此采用官方固定源码隔离导入，不安装依赖、不伪造fla包、不导入整包或替换全局triton.autotune。

## 保持不变的合同

直接调用旧适配器build_case生成同一个case，再仅替换fn、补充来源与派发说明。原cpu_inputs/device_inputs/outputs/refs、可选math_refs、rtol/atol、逐输出容差、useful/additional字节和旧100形状/CPU seed20261020+index保持原对象。没有为了通过检查修改参考或容差。

Group RMS仍是y[t,g,d]=x*rsqrt(mean_d(x²)+1e-6)*weight[g,d]；原[T,G,D]连续存储按T*G行解释，group-specific权重由行号模G在内核中选择。normalized和FP32 rstd全部保留，rstd沿用旧严格FP32容差。没有新增mean/residual输出，没有中间舍入变化。

Conv仍是零左填充的causal depthwise、bias、无activation/residual/initial/final state/varlen/CP；输入[B,T,D]、权重[D,W]沿用旧case，不重复转置或在计时内准备布局。FP32累加及最终请求dtype rtne存储来自原核。FP16与真实E4M3FN/E5M2存储均不转换为另一个dtype；能否编译并满足旧数值门槛留待root逐精度实测。

## 旧与新选择规则

| 算子 | 旧固定叶核 | 新的原生Triton选择 |
|---|---|---|
| group_rmsnorm | 全100例都用row核、num_warps=4 | CUDA下D≤512选原多行核，BT32/64/128×warp2/4/8共9配置；D>512选原row核，warp2/4/8/16共4配置。旧计划70例多行、30例row；原NB=ceil(T*G/2048)、BD规则保留。 |
| causal_conv1d | 固定BD64、warp4、stage3、BT64 | BT64、BW=nextpow2(W)仍同公开默认；原BD16/32/64/128×NVIDIA warp4/8/16/32共16配置，原功能heuristics与FLA CachedAutotuner完整保留，grid随原生BD变化。 |

所有调优在首次正确性/预热调用中发生，正式计时调用保留原生缓存后的wrapper。fn仅直接启动原始decorated前向Triton核，不含Torch原生GPU计算、输出分配、额外copy、backward或状态helper。另提供native_tuning_metadata()供root在每例首次调用后立即记录wrapper/best_config；实际编译IR及cache metadata仍是最终执行配置证据。

## 派发范围与明确不支持项

这是原生CUDA pure-Triton默认分支的形状选择及autotune适配器，使用预分配输出，**不是**完整FLA多后端注册表的重新实现。普通Group Norm的NVIDIA默认没有需要选入的Ascend实现；Conv可选Gluon/CUDA/mix不会被偷偷替换进来。FLA_GLUON或FLA_CONV_GLUON显式启用、FLA_CONV_BACKEND非triton时明确NotImplementedError。

FLA_CACHE_MODE=disabled是官方默认，含义是跳过外部预调配置查找、仍执行原生autotune；不是禁用调优。非默认cache mode明确不支持，不暗改环境。原FLA_CACHE_RESULTS和版本支持分支由未改的官方env.py计算，原cache.py独立加载。所有现有Triton/FlagGems/XDG/CUDA/TMP缓存目录必须位于papers2，PYTHONDONTWRITEBYTECODE=1。

未知dtype、非CUDA、布局或输出统计合同不同、原FLA维度限制不满足时明确不支持。GPU编译失败/数值不通过应保留失败，不改数学核或替换低精度。两项均不声称B200专属优化或优于旧配置。

## 固定来源与核验

固定提交：9d478facb282410dc150051d5f57205cd3812f31。新取9个官方文件与该git提交blob逐字节一致；三个前向函数的完整装饰器、参数、函数体AST均匹配官方原函数。两个对应旧叶核剥离装饰器后，函数体/签名AST与新候选一致。

- [layer_norm_fwd_kernel L38–L111](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/norm/layernorm.py#L38-L111)，完整decorated AST SHA `cd0d04df0fc40f5b96e174e4a179432b7b2e910f5d816fa259cb46d2cbcd5d12`。
- [layer_norm_fwd_kernel_row L114–L179](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/norm/layernorm.py#L114-L179)，完整decorated AST SHA `74e767b996834559fc8dd06a29011f745021b57e8b73d07cc1354e96c7c0880c`。
- [causal_conv1d_fwd_kernel L34–L147](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/causal_conv1d/ops.py#L34-L147)，完整decorated AST SHA `b05817f18ce205f66e06f67d1ef45c319029a2ccfddcce94202ebfb4b456a0c3`。
- [fla/utils/env.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/9d478facb282410dc150051d5f57205cd3812f31/fla/utils/env.py)，文件SHA `2c965b41300b94aa5c4b6770b7b241bff01d959f1887b2c2eed0830c9b340f63`。
- [fla/ops/utils/cache.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/9d478facb282410dc150051d5f57205cd3812f31/fla/ops/utils/cache.py)，文件SHA `a8ffda1ca609efaf366207a87ad5002253d39fd4b5e1fd9446345245d110e6f9`。
- [fla/utils/hardware.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/9d478facb282410dc150051d5f57205cd3812f31/fla/utils/hardware.py)，文件SHA `52335d5cf1d6b044e448e10ff376ff2d13fe8656da78a2c38aec195ab8420d11`。
- [fla/backends.py](https://raw.githubusercontent.com/fla-org/flash-linear-attention/9d478facb282410dc150051d5f57205cd3812f31/fla/backends.py)，文件SHA `05b20ee02d6b242f98db4fdd281f8861fc8f973a45d05ed07911a48774a84971`。

完整文件/函数/接口证据见fla_native_sources/manifest.json及fla_native_comparison.json；MIT LICENSE随bundle保存。

root可用同接口shape_plan/build_case加入独占串行烟测，indices0/99后再决定100形状全量。当前source-only准备不构成通过声明。
