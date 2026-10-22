# Compiled precision audit (CPU/read-only)

Snapshot UTC: 2026-10-21T07:08:08.067994+00:00

This snapshot reads completed compiler-cache files only; it launches no device work and imports no GPU library.

| Kernel / input dtype | Compiled variants | Result |
|---|---:|---|
| _attn_fwd / f16 | 8 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| _attn_fwd / f32 | 8 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| _attn_fwd / f8E4M3FN | 8 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| _attn_fwd / f8E5M2 | 8 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| bmm_kernel / f16 | 4 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| bmm_kernel / f32 | 4 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| bmm_kernel / f8E4M3FN | 4 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| bmm_kernel / f8E5M2 | 4 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| causal_conv1d_fwd_kernel / f16 | 20 | all targets sm_100a; storage/vector kernel; no tensor precision claim |
| causal_conv1d_fwd_kernel / f32 | 20 | all targets sm_100a; storage/vector kernel; no tensor precision claim |
| grouped_mm_kernel / f16 | 20 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| grouped_mm_kernel / f32 | 20 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| grouped_mm_kernel / f8E4M3FN | 20 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| grouped_mm_kernel / f8E5M2 | 20 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| mm_kernel_general / f16 | 1 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| mm_kernel_general / f32 | 1 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| mm_kernel_general / f8E4M3FN | 1 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| mm_kernel_general / f8E5M2 | 1 | all targets sm_100a; per-dot IEEE, no PTX TF32 |
| reshape_and_cache_flash_kernel / f16 | 8 | all targets sm_100a; storage/vector kernel; no tensor precision claim |
| reshape_and_cache_flash_kernel / f32 | 8 | all targets sm_100a; storage/vector kernel; no tensor precision claim |
| reshape_and_cache_flash_kernel / f8E4M3FN | 8 | all targets sm_100a; storage/vector kernel; no tensor precision claim |
| reshape_and_cache_flash_kernel / f8E5M2 | 8 | all targets sm_100a; storage/vector kernel; no tensor precision claim |

Pending/uncompiled kernels: 

default_dot_input_precision=tf32 in JSON is a backend default. Per-dot TTIR IEEE mode, including an elided default attribute, overrides it. A backend default alone does not prove TF32 execution.

TT_DotOp inputPrecision is DefaultValuedAttr<TT_InputPrecisionAttr, InputPrecision::IEEE>; optional custom assembly omits the default. Missing printed inputPrecision therefore means ieee, not unspecified.

Schema source: https://github.com/triton-lang/triton/blob/v3.6.0/include/triton/Dialect/Triton/IR/TritonOps.td#L626

Only listed compiled f32 variants with all TTIR dots ieee and no PTX TF32 instructions qualify as IEEE input dot mode; missing operators remain pending.

Only actual f8 compiled cache variants with direct f8 load/store and no f16 arithmetic are eligible for storage-copy claim; absence of compiled FP8 artifacts is not proof of support.

FP8 compiler-path findings: {"cache_completed_variants": 16, "cache_all_have_no_fp16_arithmetic_or_float_conversion": true, "cache_direct_transfer_dtype_groups": {"f8E5M2": 8, "f8E4M3FN": 8}, "dot_completed_variants": 66, "dot_all_have_fp8_TTIR_operands": true, "dot_all_have_tensor_mma_instructions": true, "interpretation": "Completed FP8 dot artifacts require actual f8 TTIR operands and tensor-MMA PTX; completed FP8 cache artifacts require direct load/store and no floating conversion. These are compiler-path findings, not timing or numerical result claims."}

Issues: []

Representative .ttir/.ttgir/.llir/.ptx and metadata are archived beside this report; all scanned artifacts have SHA256 in JSON.
Initial parser snapshot is retained separately; it incorrectly treated elided IEEE defaults as unspecified. That parser issue was corrected against the exact v3.6.0 upstream dialect schema and is not an experiment failure.
