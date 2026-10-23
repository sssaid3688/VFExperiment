set -eu
R=/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_run
tail -n 12 "$R/sourcegate_20261022_v2.log"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
df -h "$R"
