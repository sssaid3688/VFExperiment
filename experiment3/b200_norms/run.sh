#!/bin/bash
set -euo pipefail
PILOT_ROOT=/home/ubuntu/workspace/oyhj/papers2/experiment3
cd "$PILOT_ROOT"
mkdir -p "$PILOT_ROOT/b200_norms/cache"/{triton,flaggems,xdg,cuda,tmp,torch,inductor,extensions}
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$PILOT_ROOT/b200_norms/cache/triton" FLAGGEMS_CACHE_DIR="$PILOT_ROOT/b200_norms/cache/flaggems"
export XDG_CACHE_HOME="$PILOT_ROOT/b200_norms/cache/xdg" CUDA_CACHE_PATH="$PILOT_ROOT/b200_norms/cache/cuda"
export TMPDIR="$PILOT_ROOT/b200_norms/cache/tmp" TMP="$PILOT_ROOT/b200_norms/cache/tmp" TEMP="$PILOT_ROOT/b200_norms/cache/tmp"
export TORCH_HOME="$PILOT_ROOT/b200_norms/cache/torch" TORCHINDUCTOR_CACHE_DIR="$PILOT_ROOT/b200_norms/cache/inductor" TORCH_EXTENSIONS_DIR="$PILOT_ROOT/b200_norms/cache/extensions"
set +e
timeout --signal=TERM --kill-after=15s 2400s /root/miniconda3/envs/kaixin/bin/python -u "$PILOT_ROOT/b200_norms/benchmark.py" --mode "$1"
PILOT_EXIT=$?
set -e
printf '{"exit_code":%s,"hard_timeout_s":2400,"kill_after_s":15}\n' "$PILOT_EXIT" > "$PILOT_ROOT/b200_norms/$1/process_exit.json"
exit "$PILOT_EXIT"
