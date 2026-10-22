# B200 pilot stability audit

400/400 cases passed numerical validation and completed five groups. This is a B200-only portable Triton pilot; A5 comparative timing and VF attribution remain outstanding.

No foreign compute PID was observed in the initial/final snapshots or 100 shape-level guards. This sampling cannot exclude brief activity between guards. GPU SM clock snapshots remained 1965 MHz; power and utilization varied with workload.

33/400 cases have coefficient of variation of the five group medians greater than 10%; maximum is 21.78%. These are post-run descriptive diagnostics, not pre-registered exclusions. All raw results remain intact.

The frequent group medians near 4.096/6.144 microseconds indicate substantial quantization/launch-floor structure at these small workloads. CV=0 of five rounded group medians does not prove zero measurement uncertainty. Inspect within-group IQR as well. Do not infer architectural efficiency from a single tiny-kernel median.

| Operator | Dtype | Cases | CV>10% | Median within-group IQR/median | Max within-group IQR/median |
|---|---|---:|---:|---:|---:|
| silu | fp16 | 100 | 7 | 0.010 | 0.492 |
| silu | fp32 | 100 | 5 | 0.010 | 0.496 |
| gelu_tanh | fp16 | 100 | 12 | 0.010 | 0.496 |
| gelu_tanh | fp32 | 100 | 9 | 0.006 | 0.496 |

## Cases needing timing review

| Case | Shape | Five group medians, us | CV |
|---|---|---|---:|
| gelu_tanh_fp16_002 | [1, 256] | 4.160, 6.144, 4.224, 6.048, 6.016 | 19.359% |
| gelu_tanh_fp32_002 | [1, 256] | 6.048, 6.048, 4.160, 6.144, 4.160 | 19.811% |
| gelu_tanh_fp16_003 | [1, 512] | 6.112, 4.192, 4.192, 6.112, 4.128 | 21.500% |
| gelu_tanh_fp32_003 | [1, 512] | 4.224, 6.144, 6.016, 6.112, 6.144 | 14.707% |
| gelu_tanh_fp16_004 | [1, 768] | 4.128, 4.160, 6.144, 4.256, 4.224 | 19.082% |
| silu_fp32_004 | [1, 768] | 4.160, 6.112, 6.112, 4.192, 6.144 | 19.955% |
| gelu_tanh_fp32_004 | [1, 768] | 6.112, 6.144, 4.256, 6.144, 6.144 | 14.599% |
| gelu_tanh_fp16_005 | [1, 1024] | 4.288, 6.016, 6.112, 5.984, 4.160 | 18.738% |
| silu_fp16_006 | [1, 1536] | 4.192, 6.112, 4.224, 4.192, 6.048 | 20.764% |
| silu_fp16_007 | [1, 2048] | 6.144, 4.160, 6.048, 6.048, 6.112 | 15.138% |
| gelu_tanh_fp16_011 | [2, 128] | 4.160, 6.048, 4.224, 6.080, 6.112 | 19.430% |
| gelu_tanh_fp32_011 | [2, 128] | 6.080, 6.080, 4.128, 4.160, 6.144 | 20.165% |
| gelu_tanh_fp16_012 | [2, 256] | 4.160, 4.192, 4.208, 6.112, 4.192 | 18.820% |
| gelu_tanh_fp32_012 | [2, 256] | 4.192, 6.144, 6.112, 6.112, 6.080 | 14.996% |
| gelu_tanh_fp16_013 | [2, 512] | 6.144, 6.144, 6.112, 4.224, 4.160 | 19.856% |
| silu_fp16_014 | [2, 768] | 6.048, 6.112, 4.192, 6.112, 6.112 | 14.907% |
| silu_fp16_015 | [2, 1024] | 6.144, 6.048, 6.048, 6.048, 4.192 | 14.779% |
| gelu_tanh_fp16_020 | [4, 64] | 6.112, 6.048, 6.048, 4.160, 6.048 | 14.991% |
| gelu_tanh_fp32_020 | [4, 64] | 4.128, 6.112, 6.080, 4.192, 4.192 | 21.351% |
| gelu_tanh_fp16_021 | [4, 128] | 4.160, 4.224, 4.192, 6.048, 6.112 | 20.913% |
| gelu_tanh_fp32_021 | [4, 128] | 6.112, 4.192, 4.224, 4.192, 6.080 | 20.911% |
| silu_fp32_022 | [4, 256] | 4.160, 6.144, 6.112, 6.144, 4.128 | 20.416% |
| silu_fp16_023 | [4, 512] | 6.144, 6.048, 4.192, 6.112, 6.112 | 14.957% |
| gelu_tanh_fp16_030 | [8, 64] | 4.128, 4.528, 4.160, 6.112, 6.112 | 20.367% |
| gelu_tanh_fp32_030 | [8, 64] | 4.128, 6.080, 4.128, 4.128, 6.080 | 21.780% |
| gelu_tanh_fp16_031 | [8, 128] | 4.160, 4.128, 6.112, 5.984, 6.112 | 19.926% |
| silu_fp32_031 | [8, 128] | 6.016, 4.192, 6.144, 6.112, 6.048 | 14.833% |
| gelu_tanh_fp32_031 | [8, 128] | 6.080, 4.288, 6.144, 6.144, 6.112 | 14.247% |
| gelu_tanh_fp16_040 | [16, 64] | 6.112, 6.016, 4.160, 6.048, 4.160 | 19.635% |
| silu_fp32_040 | [16, 64] | 6.112, 4.192, 6.144, 6.112, 6.112 | 15.038% |
| silu_fp16_041 | [16, 128] | 6.144, 4.160, 6.112, 6.016, 6.144 | 15.239% |
| silu_fp16_050 | [32, 64] | 6.048, 4.224, 6.048, 6.048, 6.048 | 14.353% |
| silu_fp32_096 | [512, 1536] | 6.176, 8.064, 6.176, 6.208, 6.144 | 12.888% |

## Next measurement decision

Preserve this run as pilot evidence. Before final manuscript timings, diagnose event resolution, cold-cache launch cadence and balanced order; specify an appropriate stability gate prospectively, then take a fresh independent run without replacing or selectively choosing the fastest pilot results. Any altered timer/cache policy must also be matched on A5. No additional GPU experiment was launched by this audit.

Source validation: the two operator files byte-match official FlagGems v5.0.2 commit 30bfcc0735b8ebd48bd7339ef752ec64df69663f; the compiled PTX targets sm_100a. This is compatibility/target compilation evidence, not evidence of an operator optimization designed specifically for B200.
