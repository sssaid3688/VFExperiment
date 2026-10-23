#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'Usage: bash launch_full_reviewed.sh QUEUE_JSON NEW_BATCH ROOT_REVIEWED_QUALIFICATION_JSON' >&2
  exit 2
fi
source /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/environment.sh
exec /root/miniconda3/envs/kaixin/bin/python -B -u supervisor_qualified.py --queue "$1" --batch "$2" --mode full --qualification "$3" --pair-timeout 3600
