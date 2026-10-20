#!/bin/bash
# 25算子 x 100 shape 的 fp16/fp32/fp8 dtype 对比基准 (GPU独占串行)
cd /home/ubuntu/workspace/oyhj/papers2/experiment2
KX=/root/miniconda3/envs/kaixin/bin/python

echo "[dt] start $(date)" >> logs/dt_chain.log
$KX bench_flaggems.py --n100 --dtype fp16 > logs/dt_fg_fp16.log 2>&1
echo "[dt] fg fp16 done $(date)" >> logs/dt_chain.log
$KX bench_flaggems.py --n100 --dtype fp32 > logs/dt_fg_fp32.log 2>&1
echo "[dt] fg fp32 done $(date)" >> logs/dt_chain.log
$KX bench_liger.py --n100 --dtype fp16 > logs/dt_liger_fp16.log 2>&1
echo "[dt] liger fp16 done $(date)" >> logs/dt_chain.log
$KX bench_liger.py --n100 --dtype fp32 > logs/dt_liger_fp32.log 2>&1
echo "[dt] liger fp32 done $(date)" >> logs/dt_chain.log
$KX bench_matmul.py --n100 --dtype fp16 > logs/dt_mm_fp16.log 2>&1
echo "[dt] mm fp16 done $(date)" >> logs/dt_chain.log
$KX bench_matmul.py --n100 --dtype fp32 > logs/dt_mm_fp32.log 2>&1
echo "[dt] mm fp32 done $(date)" >> logs/dt_chain.log
$KX bench_matmul.py --n100 --dtype fp8 > logs/dt_mm_fp8.log 2>&1
echo "[dt] mm fp8 done $(date)" >> logs/dt_chain.log
python3.10 bench_attn.py --n100 --dtype fp16 > logs/dt_attn_fp16.log 2>&1
echo "[dt] attn fp16 done $(date)" >> logs/dt_chain.log
python3.10 bench_attn.py --n100 --dtype fp32 > logs/dt_attn_fp32.log 2>&1
echo "[dt] attn fp32 done $(date)" >> logs/dt_chain.log
$KX make_xlsx_dtypes.py > logs/dt_xlsx.log 2>&1
echo "[dt] xlsx done $(date)" >> logs/dt_chain.log
echo DTYPES_ALL_DONE > logs/dt_done.flag
