set -euo pipefail
BASE=/home/ubuntu/workspace/oyhj/papers2/experiment3
cd "$BASE"
mkdir -p softmax_b200/cache/{triton,flaggems,xdg,cuda,tmp,torch,inductor,extensions}
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$BASE/softmax_b200/cache/triton" FLAGGEMS_CACHE_DIR="$BASE/softmax_b200/cache/flaggems" XDG_CACHE_HOME="$BASE/softmax_b200/cache/xdg" CUDA_CACHE_PATH="$BASE/softmax_b200/cache/cuda" TMPDIR="$BASE/softmax_b200/cache/tmp" TMP="$BASE/softmax_b200/cache/tmp" TEMP="$BASE/softmax_b200/cache/tmp" TORCH_HOME="$BASE/softmax_b200/cache/torch" TORCHINDUCTOR_CACHE_DIR="$BASE/softmax_b200/cache/inductor" TORCH_EXTENSIONS_DIR="$BASE/softmax_b200/cache/extensions"
timeout --kill-after=15s 400s /root/miniconda3/envs/kaixin/bin/python -B -u softmax_forward.py --machine b200 --mode smoke > softmax_b200/smoke.log 2>&1
# Full run is gated on successful smoke process exit and summary.
set +e
timeout --kill-after=15s 1200s /root/miniconda3/envs/kaixin/bin/python -B -u softmax_forward.py --machine b200 --mode full > softmax_b200/full.log 2>&1
result=$?
set -e
printf '{"exit_code":%s,"hard_timeout_s":1200}\n' "$result" > softmax_b200/process_exit.json
exit "$result"
