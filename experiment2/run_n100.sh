#!/bin/bash
# 25算子 × 100 shape 全量串行基准 (GPU独占)
cd /home/ubuntu/workspace/oyhj/papers2/experiment2
KX=/root/miniconda3/envs/kaixin/bin/python

echo "[chain] start $(date)" >> logs/r3_chain.log
$KX bench_flaggems.py --n100 --force > logs/r3_flaggems.log 2>&1
echo "[chain] flaggems done $(date)" >> logs/r3_chain.log
$KX bench_liger.py --n100 --force > logs/r3_liger.log 2>&1
echo "[chain] liger done $(date)" >> logs/r3_chain.log
python3.10 bench_attn.py --n100 --force > logs/r3_attn.log 2>&1
echo "[chain] attn done $(date)" >> logs/r3_chain.log
$KX bench_matmul.py --n100 --force > logs/r3_matmul.log 2>&1
echo "[chain] matmul done $(date)" >> logs/r3_chain.log
$KX make_xlsx.py > logs/r3_xlsx.log 2>&1
echo "[chain] xlsx done $(date)" >> logs/r3_chain.log
echo N100_ALL_DONE > logs/r3_done.flag
