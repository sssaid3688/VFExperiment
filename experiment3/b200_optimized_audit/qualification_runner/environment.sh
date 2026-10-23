#!/usr/bin/env bash
set -euo pipefail
QUAL_ROOT=/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner
cd "$QUAL_ROOT"
mkdir -p cache/{triton,flaggems,xdg,cuda,tmp,torch,inductor,extensions,dump}
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$QUAL_ROOT/cache/triton" FLAGGEMS_CACHE_DIR="$QUAL_ROOT/cache/flaggems" XDG_CACHE_HOME="$QUAL_ROOT/cache/xdg" CUDA_CACHE_PATH="$QUAL_ROOT/cache/cuda"
export TMPDIR="$QUAL_ROOT/cache/tmp" TMP="$QUAL_ROOT/cache/tmp" TEMP="$QUAL_ROOT/cache/tmp" TORCH_HOME="$QUAL_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$QUAL_ROOT/cache/inductor" TORCH_EXTENSIONS_DIR="$QUAL_ROOT/cache/extensions" TRITON_DUMP_DIR="$QUAL_ROOT/cache/dump"
export PYTHONPATH="$QUAL_ROOT/..:$QUAL_ROOT/../../b200_remaining:$QUAL_ROOT/../../b200_sm100_sources${PYTHONPATH:+:$PYTHONPATH}"
