set -euo pipefail
BASE=/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_remaining
cd "$BASE"
mkdir -p cache/{triton,flaggems,xdg,cuda,tmp,torch,inductor,extensions}
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TRITON_CACHE_DIR="$BASE/cache/triton" FLAGGEMS_CACHE_DIR="$BASE/cache/flaggems" XDG_CACHE_HOME="$BASE/cache/xdg" CUDA_CACHE_PATH="$BASE/cache/cuda" TMPDIR="$BASE/cache/tmp" TMP="$BASE/cache/tmp" TEMP="$BASE/cache/tmp" TORCH_HOME="$BASE/cache/torch" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" TORCH_EXTENSIONS_DIR="$BASE/cache/extensions"
exec /root/miniconda3/envs/kaixin/bin/python -B -u supervisor.py --queue "$1" --batch "$2"
