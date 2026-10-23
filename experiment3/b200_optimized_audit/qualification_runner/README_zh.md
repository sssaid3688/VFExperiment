# FLA源码烟测与正式计时入口（尚未启动）

稳定交付补强：FULL资格文件必须把`approved_queue_sha256`设为root实际批准的队列文件字节SHA256；supervisor启动前与子runner再次核验，不接受仅有APPROVED标签。每pair同时绑定`dependency_sha256`：复用的旧matrix/norm适配器及其provenance/manifest、FLA源码manifest、SM100 extraction manifest、native FlagGems调度配置/运行辅助文件。运行前后校验依赖，全部依赖字节另存`dependency_snapshots`。候选自身源码、计划、新runner、capture与contract辅助模块SHA也必须匹配。

`sourcegate_ir_evidence`保留实际0/99编译清单的远程路径与SHA；root须另填实际IR审核产物`reviewed_ir_evidence=[{"path":"绝对路径","sha256":"实际SHA256"}]`。两类文件都必须真实存在且字节匹配；空列表或占位值不能进入FULL。现有22组仅是两shape通过，不表示已获source/IR资格。

物理输入hash原样保留。新增`logical_input_hashes`由实际CPU舍入输入恢复：SM100 B逆转置为逻辑[...,K,N]；仅真实E5M2 SDPA的V逆转置为[B,H,S,D]，FP16 V保持原样；Conv权重由[D,W]恢复为[W,D]。设备指针/尺寸元数据另列，不作数学输入配对。逆转置只在CPU执行且位于计时外；A5配对需形状、dtype、seed和所有数学输入hash一致，不能仅按index配对不同cohort。

此目录只准备新文件。旧sourcegate29项结果、旧基线与运行中脚本均未修改。候选/源码bundle完整性由上级目录fla_native_comparison.json、fla_native_sources/manifest.json记录。

## FLA source smoke8组合

provider=`fla`；算子group_rmsnorm、causal_conv1d；各FP32、FP16、E4M3FN、E5M2；每组合只跑原100形状计划indices0/99。队列独立命名fla_sourcegate_20261023_v1，不覆盖原批次。Group RMS预期70多行/30行，完整原生9/4配置；Conv完整原生16配置。实际是否编译/数值通过必须由启动后记录判定。

所有缓存与输出位于B200 papers2/experiment3/b200_optimized_audit/qualification_runner；继续用既有b200_remaining/device0.lock。supervisor先非阻塞取得同一锁，若占用则不启动任务；锁描述符传给子进程，runner核实该原锁。独占串行，每pair硬超时900秒；每5秒记录GPU进程，每组前亦检查。发现外部GPU进程或监控错误时标记STOP_REQUEST/污染，子进程在下个安全检查处退出，不把污染计时判有效；只有明确硬超时才终止子进程组。

root决定启动时在B200执行：

```bash
bash /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/launch_fla_source_smoke.sh
```

后台运行的可选命令（由root执行，本次未启动）：

```bash
nohup bash /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/launch_fla_source_smoke.sh > /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/fla_sourcegate_20261023_v1.log 2>&1 < /dev/null &
```

首次运行写新batch；再次执行同batch会拒绝覆盖。重新安排必须显式选择新batch，不自动重试失败或超时。

## 正式FULL runner

runner_qualified.py复制当前runner_stage2.py的guard/tensorhash/check等验证函数，新增provider=fla，采用单独捕获模块。每个case流程：构建原数据与预分配输出、核验/归档来源；只在首次正确性调用期间挂JITFunction.run及可用的LibEntry.run，捕获调优trial及缓存返回的实际CompiledKernel；finally恢复原始方法，再归档IR、验证CPU FP64参考；正式计时前与每组后断言原始run仍在。下一case重新安装一轮。捕获对象在计时前清空，不能把活动wrapper或append成本留在计时循环。

LibEntry观察用于弥补其hot cache直接返回CompiledKernel而不再进入JITFunction.run的情况。hook观察数可能重叠，明确不是GPU launch次数。没有观察到CompiledKernel时不假称捕获IR并禁止该case进入计时。失败时仅归档实际返回过的partial IR，编译在返回前失败则只有真实异常证据。后端metadata的ieee或SM100标签不自动授予IEEE或B200优化资格，必须看实际TTIR/PTX/数学合同。

正式计时保留5次do_bench(warmup25,rep100,return_mode=all)；它们是同输入同进程重复测量。保存每个raw event、group median、q25/q75/IQR、sample-CV、输入hash、source SHA、实际dtype、初末与期间GPU进程、正确性和post-check。失败/污染记录保留，performance_eligible=False。完整队列一pair硬超时3600秒。

queue_full_pending_review.json只根据已保存的真实v2 smoke summary列出22个两端PASS组合；full_source_review_template.json仍是PENDING。6个unsupported及SM100 Dense FP32失败排除，FLA8组还未运行，尚未加入正式列表。SM100 Dense FP32（含stage2变体）在runner和supervisor均明确禁止；native已通过IEEE来源需root审核后使用。

root审核实际源码、IR/PTX、FP32精度等资格，填入逐pair source_review_passed / ir_review_passed / precision_contract_review_passed / reviewed_ir_evidence，确认当前adapter、plan、新runner及capture helper SHA匹配；仅保留确认的pair，再将资格文件状态改APPROVED。本次没有自动批准，也不要求用户重复授权已安排的研究工作。

完成root审核后，显式执行（占位参数必须替换，当前不能运行）：

```bash
bash /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/launch_full_reviewed.sh REVIEWED_QUEUE_JSON NEW_FULL_BATCH REVIEWED_QUALIFICATION_JSON
```

未提供APPROVED资格文件时，FULL在Torch/GPU构建之前拒绝启动。正式新批次只追加新证据，不改旧基线、旧XLSX或其他在跑文件。FLA后续通过source smoke并审核后，应冻结另一个明确FULL队列/资格文件，不能预填通过结果。

## 已执行的本地CPU检查

候选及bundle的官方9源文件字节/3个完整decorated函数AST/2个旧叶核数学AST/2×100唯一原计划核对；所有新Python文件内存语法检查；capture_scope用明确CPU mock验证恢复、下一case重启、LibEntry tuple观察和异常finally恢复。mock仅测试捕获机制，不写入任何实验结果。未在远端导入Torch/Triton/FLA、未构造GPU case、未运行烟测或FULL。
