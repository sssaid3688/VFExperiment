#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/environment.sh
exec /root/miniconda3/envs/kaixin/bin/python -B -u supervisor_qualified.py --queue queue_fla_source_smoke.json --batch fla_sourcegate_20261023_v1 --mode smoke --pair-timeout 900
