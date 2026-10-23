# B200前向基线：来源、原生调度与优化资格

核查日期：2026年10月22日。仅核查源码、文件和已存测量合同，没有启动GPU，没有修改软件环境或已有数据。

现有结果可作为披露配置与计算范围的权威Triton前向基线。25项“B200专属最佳实现”的资格尚未建立；固定叶核的部分算子还应补测原生调度。

实机FlagGems为5.0.2。新取的1396个Python/配置源文件全部与官方固定提交 `30bfcc0735b8ebd48bd7339ef752ec64df69663f` 逐字节一致，差异0。以下判断基于这些实文件，而非仅凭包版本或先前另一提交的审计。

## 哪些确有硬件优化路径

- FlagGems默认架构表只映射SM80和SM90；没有SM100/Blackwell目录派发。major10仍使用NVIDIA通用配置。手工强制Hopper架构不构成B200原生支持证据。
- Group GEMM在计算能力major≥9且TensorDescriptor可用时选TMA。这是SM90及更新架构的硬件特性路径，当前固定non-TMA叶核绕过了它，值得补测；不能称B200专属。
- Liger另有明确Blackwell SwiGLU列分块路径，但要求nextpow2(N)≥16384。旧计划N最大8192，100例均不触发。低精度实现还在SiLU后先舍入再乘，不与当前FP32中间合同混用。
- FlagGems的`cutlass_scaled_mm_sm100`实际为未实现占位；`fp8_matmul.py`的固定配置明文针对H20。它们均不能作为可用B200优化基线。

公共调度证据：[generic GELU L25–30](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/gelu.py#L25-L30)；[NVIDIA架构表](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/backend/_nvidia/__init__.py#L9-L12)；[NVIDIA vendor-common替换条件](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/runtime/__init__.py#L28-L44)；[Group MM原生选择 L563–614](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L563-L614)。

## 25项的实际测量范围

表中所有固定源码链接都包含不可变提交及行号。实际来自Liger/FLA/其他提交的核单独列出，不套用FlagGems5.0.2资格。当前已选的25项均未执行明确B200专属分支。

| 算子 | 当前测量口径与局限 | 逐项源码证据 |
|---|---|---|
| silu | 保留原生pointwise生成和形状规则，预分配输出；通用NVIDIA向量核。 | [原生/对应入口 L12–17](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/silu.py#L12-L17) |
| gelu_tanh | 保留generic pointwise；_nvidia/ops/gelu存在但未被NVIDIA vendor-common自动替换。 | [原生/对应入口 L25–30](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/gelu.py#L25-L30) |
| swiglu | FlagGems SiLU与乘法保持FP32中间；当前N≤8192，未使用Liger的Blackwell宽列分支。 | [原生/对应入口 L12–17](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/silu_and_mul.py#L12-L17) |
| rms_norm | Liger GEMMA模式、offset0、FP32中间与rstd；当前形状原生也会走row核，无B200特判。 | [实际Liger核](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/rms_norm.py#L68-L150)；[原生/对应入口 L236–254](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/rms_norm.py#L236-L254) |
| layer_norm | Liger单row前向，FP32 mean/rstd；没有采用FlagGems的三段形状派发/搜索，统计输出口径不同。 | [实际Liger核](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/layer_norm.py#L27-L86)；[原生/对应入口 L333–395](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/layernorm.py#L333-L395) |
| residual_rmsnorm | 两核：add后舍入到输入dtype，再做RMSNorm；原生fused RMS在未舍入和上归一化并原地写输入，不能直接换。 | [实际add](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/python/tutorials/01-vector-add.py#L30-L54)、[实际Norm](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/rms_norm.py#L68-L150)；[原生/对应入口 L14–54](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/fused_add_rms_norm.py#L14-L54) |
| residual_layernorm | 两核：add后舍入，再做LayerNorm；应标明未融合组合，不能当作原生融合最佳性能。 | [实际add](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/python/tutorials/01-vector-add.py#L30-L54)、[实际Norm](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/layer_norm.py#L27-L86)；[FG相近入口（非同合同） L333–395](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/layernorm.py#L333-L395) |
| rope | 三核：两次Triton copy恢复Q/K输出，再做Liger旋转；copy耗时与流量已计入。 | [实际Liger核](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/rope.py#L7-L112)、[copy](https://github.com/flagos-ai/FlagGems/blob/32e4a4cf9b8618185a13eb82d72687fcdbbfb982/src/flag_gems/ops/alias_copy.py#L28-L34)；[原生/对应入口 L181–312](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/rotary_embedding.py#L181-L312) |
| group_rmsnorm | FLA row核固定warps；绕过原生小D多行BT路径和warps搜索。FlagGems无同group-weight合同入口。 | [实际FLA核](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/norm/layernorm.py#L123-L179)；[FG相近入口（非同合同） L236–254](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/rms_norm.py#L236-L254) |
| group_gemm | 固定non-TMA叶核与tile/grid；绕过原生SM90+ TMA选择和7配置搜索，是优先补强项。 | [原生/对应入口 L12–13](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/group_gemm.py#L12-L13) |
| causal_conv1d | FLA因果depthwise核固定BD/warps；绕过原生BD×warps搜索。FlagGems普通conv1d不是同一入口。 | [实际FLA核](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/modules/causal_conv1d/ops.py#L34-L147)；[FG相近入口（非同合同） L9–65](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/conv1d.py#L9-L65) |
| sdpa | 固定BM64/BN32/warp4/stage2的Triton叶核；绕过原生attention搜索，小头配置亦未参与选择。 | [原生/对应入口 L146–329](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/attention.py#L146-L329) |
| mrope | 三核：两次copy加Liger Qwen2VL旋转；低精度cache合同需按修正批次解释。 | [实际Liger核](https://github.com/linkedin/Liger-Kernel/blob/43293d84d760e402e4fd974b12410a1d47c31761/src/liger_kernel/ops/qwen2vl_mrope.py#L7-L107)、[copy](https://github.com/flagos-ai/FlagGems/blob/32e4a4cf9b8618185a13eb82d72687fcdbbfb982/src/flag_gems/ops/alias_copy.py#L28-L34) |
| kv_cache_store | 直接Triton叶核，固定每token一program；绕过可选C扩展以维持纯Triton，非B200调优。 | [原生/对应入口 L61–112](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/fused/reshape_and_cache_flash.py#L61-L112) |
| softmax | 保留inner前向heuristics；K=1与公开入口同分支。yaml同名调优用于backward，不能写成前向搜索。 | [原生/对应入口 L284–328](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/softmax.py#L284-L328) |
| log_softmax | 前向BM8/BN256/warp8与公开默认一致；yaml搜索装饰的是backward。 | [原生/对应入口 L15–55](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/log_softmax.py#L15-L55) |
| embedding | 直接gather叶核，每token一program、列维nextpow2，与公开前向规则一致。 | [原生/对应入口 L14–33](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/embedding.py#L14-L33) |
| cumsum | FP16/32保留N≤16384原生row路径；FP8另用FP32累加原核并单独验证partial_sum。 | [原生/对应入口 L212–239](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/cumsum.py#L212-L239) |
| groupnorm | 预分配后原group_norm核，组维与空间维nextpow2；统计输出dtype按原合同。 | [原生/对应入口 L184–213](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/groupnorm.py#L184-L213) |
| sigmoid | 保留pointwise生成；真实FP8存储下仍是FP32向量计算，不计FP8 TensorCore吞吐。 | [原生/对应入口 L12–19](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sigmoid.py#L12-L19) |
| sum_reduce | 显式dim=-1，保留inner heuristics；不是dim=None的另一种全张量归约。 | [原生/对应入口 L241–326](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/sum.py#L241-L326) |
| max_pool2d | 保留原12配置autotuner；有限值2×2/stride2合同验证首个并列最大值索引，NaN/全−∞不在此性能合同。 | [原生/对应入口 L34–127](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/max_pool2d_with_indices.py#L34-L127) |
| relu | 保留pointwise生成；FP8 compare/select/copy不能归为TensorCore计算。 | [原生/对应入口 L11–14](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/relu.py#L11-L14) |
| dense_matmul | 固定通用叶核，绕过mm搜索；独立A/B不是SYRK，StreamK只给SM80条件，不会在SM100自动选择。 | [原生/对应入口 L25–107](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/mm.py#L25-L107) |
| batched_matmul | 固定通用叶核，绕过12配置LibTuner及可选FlagTune；人工DIVISIBLE元数据等价，但未作原生配置选择。 | [原生/对应入口 L16–138](https://github.com/flagos-ai/FlagGems/blob/30bfcc0735b8ebd48bd7339ef752ec64df69663f/src/flag_gems/ops/bmm.py#L16-L138) |

## 已准备的补强候选

第一组保持旧形状、种子、实际输入、独立CPU FP64参考、容差及预分配输出：Dense MM/BMM恢复原生LibTuner，Group GEMM恢复原生SM90+ TMA、7配置和descriptor预钩子。它们仍是NVIDIA通用或硬件特性候选，没有预先赋予B200专属或性能胜出结论。

第二组仅FP32，使用Liger未修改的原生Blackwell/row dispatcher：新计划为M=[1,2,4,8,16,32,64,128,256,512] × N=[64,128,256,512,1024,4096,8192,14336,16384,28672]。计划在计时前冻结，共100唯一形状；Blackwell上预期30例tiled、70例row，实机每例仍需记录_should_tile和所选核。烟测0/99覆盖两路径。

新SwiGLU计划SHA256：`a234f60ef926cdcedd17577e12befe6bb0a0e47fcbf4639c3f6ee0f0fda947d6`。FP16/FP8明确因中间舍入合同不同而排除，不修改数学核追求通过。A5后续必须采用相同新形状/种子/FP32合同，不能按index把新旧不同形状相除。

候选仅完成CPU静态与100唯一形状检查，本审计没有编译、正确性验证或计时。串行烟测和全量是否通过，须以另外保存的真实运行证据为准。

## 论文中应明确的边界

固定tile/warps与禁用原生autotune的结果，应写为披露配置的基线，并说明部分算子尚未覆盖原生形状派发和硬件特性。点算子的原生代码生成、MaxPool的autotuner及Softmax等的heuristics分别保留；应逐项区分，不能笼统写成全部禁用或全部优化。

ResidualNorm的两核组合、RoPE/mRoPE的两次copy与旋转、统计输出及临时项，均属于被计时范围，应在对比中保留对应成本。B200上的实际编译目标或TensorCore指令证明执行能力，不能单独证明B200专属调优或最优性能。FP8存储合同、FP32向量算术和原生FP8矩阵乘应分别解释。

完整逐项源码SHA、实机路径、固定commit、行号、当前入口/launch及字节核对见随附审计明细；原始包文件和许可证已归档。审计不从源码名称推导加速结论。
