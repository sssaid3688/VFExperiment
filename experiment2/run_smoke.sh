#!/usr/bin/env bash
# 25 算子昇腾冒烟测试 —— 只用卡 0
# 用法: cd experiment2 && ./run_smoke.sh
set -o pipefail

EXP2="$(cd "$(dirname "$0")" && pwd)"

# python 环境: venv_cann (torch 2.7.1 + torch_npu + triton-ascend 3.2.2)
source /home/oyhj/venv_cann/bin/activate
# CANN 9.1.0-beta.3
source /home/oyhj/Ascend/ascend-toolkit/set_env.sh

export TORCH_DEVICE_BACKEND_AUTOLOAD=0   # 规避 torch_npu autoload 与 triton 注册冲突
export ASCEND_RT_VISIBLE_DEVICES=0       # 锁定卡 0
export GEMS_VENDOR=ascend                # flag_gems 直接选 _ascend 后端

mkdir -p "$EXP2/results"
TS=$(date +%Y%m%d_%H%M%S)
python -u "$EXP2/smoke_test.py" 2>&1 | tee "$EXP2/results/smoke_${TS}.log"
exit "${PIPESTATUS[0]}"
