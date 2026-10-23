set -eu
ROOT=/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_run
cd "$ROOT"
mkdir -p cache/triton cache/flaggems cache/xdg cache/cuda tmp
export TRITON_CACHE_DIR="$ROOT/cache/triton" FLAGGEMS_CACHE_DIR="$ROOT/cache/flaggems" XDG_CACHE_HOME="$ROOT/cache/xdg" CUDA_CACHE_PATH="$ROOT/cache/cuda" TMPDIR="$ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/../b200_sm100_sources:$ROOT/../b200_optimized_audit:$ROOT/../b200_remaining"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
nohup /root/miniconda3/envs/kaixin/bin/python -B -u supervisor.py > sourcegate_20261022_v2.log 2>&1 < /dev/null &
echo "launched_pid=$!"
