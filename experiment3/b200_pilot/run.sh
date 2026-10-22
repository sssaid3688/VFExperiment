#!/bin/bash
set -euo pipefail
PILOT_ROOT=/home/ubuntu/workspace/oyhj/papers2/experiment3
cd "$PILOT_ROOT"
mkdir -p "$PILOT_ROOT/cache/triton" "$PILOT_ROOT/cache/flaggems" "$PILOT_ROOT/cache/xdg" "$PILOT_ROOT/cache/cuda" "$PILOT_ROOT/cache/tmp" "$PILOT_ROOT/cache/torch" "$PILOT_ROOT/cache/inductor" "$PILOT_ROOT/cache/extensions" "$PILOT_ROOT/b200_pilot"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$PILOT_ROOT/cache/triton"
export FLAGGEMS_CACHE_DIR="$PILOT_ROOT/cache/flaggems"
export XDG_CACHE_HOME="$PILOT_ROOT/cache/xdg"
export CUDA_CACHE_PATH="$PILOT_ROOT/cache/cuda"
export TMPDIR="$PILOT_ROOT/cache/tmp"
export TMP="$PILOT_ROOT/cache/tmp"
export TEMP="$PILOT_ROOT/cache/tmp"
export TORCH_HOME="$PILOT_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PILOT_ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$PILOT_ROOT/cache/extensions"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
/root/miniconda3/envs/kaixin/bin/python -u "$PILOT_ROOT/b200_pilot/benchmark.py" --mode "$1"
