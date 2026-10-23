"""CPU source/contract audit; does not initialize CUDA or import Triton."""
import ast,datetime,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'source_manifest.json').read_text());extraction=json.loads((ROOT/'extraction_manifest.json').read_text())
assert all('error' not in r for r in manifest)
for r in manifest:assert hashlib.sha256((ROOT/r['archive']).read_bytes()).hexdigest()==r['sha256']
for r in extraction:assert hashlib.sha256((ROOT/r['extracted_file']).read_bytes()).hexdigest()==r['extracted_sha256']
spec=importlib.util.spec_from_file_location('source_only_adapter',ROOT/'sm100_ops.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
plans={op:adapter.shape_plan(op) for op in adapter.OPS}
baseline=ROOT.parent/'b200_remaining/matrix_attention_ops.py';bspec=importlib.util.spec_from_file_location('baseline_source_only',baseline);base=importlib.util.module_from_spec(bspec);bspec.loader.exec_module(base)
for op,plan in plans.items():
 assert len(plan)==100 and len({tuple(s) for s in plan})==100
 assert plan==base.shape_plan(op)
for s in plans['sdpa']:assert s[2]%512==0 and s[3] in [64,128]
# Static descriptor coordinate proof, no tensor values/GPU. v3.6's FP8
# descriptor's second dimension is B*H*S but head offsets advance by S*D.
old_violations=[]
for i,(b,h,s,d) in enumerate(plans['sdpa']):
 bad_heads=[head for head in range(b*h) if head*s*d+s-1>=b*h*s]
 if bad_heads:old_violations.append({'index':i,'shape':[b,h,s,d],'invalid_head_ids':bad_heads})
 # Fixed main descriptor [B*H*D,S] and coordinates[head*D,start_n] have
 # every tile in bounds for this selected exact-block shape grid.
 assert (b*h-1)*d+d<=b*h*d and s%64==0
runtime=json.loads((ROOT/'runtime_cpu_api_audit.json').read_text());opts=runtime['files'][0]['CUDAOptions_fields']
assert 'default_dot_input_precision' in opts and 'maxnreg' in opts
assert 'clc' not in opts
def triton_api_calls(path):
 tree=ast.parse(path.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_attn_fwd','_attn_fwd_inner','_maybe_make_tensor_desc']]
 return sorted({ast.unparse(n.func) for f in functions for n in ast.walk(f) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith('tl.')})
old_calls=triton_api_calls(ROOT/'triton_v3.6.0/python/tutorials/06-fused-attention.py');new_calls=triton_api_calls(ROOT/'triton_main/python/tutorials/06-fused-attention.py')
assert old_calls==new_calls
for p in ROOT.rglob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
results={'verification_status':'SOURCE_AUDITED_NOT_GPU_TESTED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_gpu_execution':True,'upstream_files_verified':len(manifest),'exact_ast_function_count':sum(len(r['definitions']) for r in extraction),'plans':plans,'plan_equality_with_existing_cohort':True,'runtime_static_api':runtime,'v360_fp8_attention_descriptor_static_oob':{'affected_shape_count':len(old_violations),'cases':old_violations,'scope':'Coordinate proof against upstream descriptor shape, not a GPU reproduction or performance assertion'},'candidates':[
 {'op':'dense_matmul','source':'official Triton09 v3.6.0','architecture':'Explicit Blackwell WS; host TMA persistent or device TMA descriptor persistent','dtypes':{'fp16':'official demonstrated source candidate','fp8_e4m3fn':'official demonstrated unscaled source candidate','fp32':'generic pointer-output device descriptor source candidate, probe required, IEEE option explicit','fp8_e5m2':'generic pointer-output device descriptor source candidate, native dot compilation probe required'},'replacement':'Root smoke/100shape numeric+timing+IR validation first; retain prior generic cohort'}
,{'op':'sdpa','source':'official Triton06 v3.6.0 FP16; pinned main93035c4 E5M2','architecture':'Explicit major10 is_blackwell; WS inner loop, accumulator split and maxnreg168/80','dtypes':{'fp16':'source candidate','fp8_e5m2':'corrected upstream V layout source candidate; compatibility probe needed','fp32':'UNSUPPORTED: kernel hardcodes FP16/E5 output/probability','fp8_e4m3fn':'UNSUPPORTED: kernel hardcodes FP16/E5 output/probability'},'replacement':'Only qualifying precision subsets; FP32/E4 stay generic unless another authoritative source is found'}
,{'op':'batched_matmul','source':'official Triton08 grouped TMA','architecture':'SM90+ TMA portable path; no SM100-specific WS claim','dtypes':{'fp16':'source candidate','fp8_e4m3fn':'source candidate','fp32':'UNSUPPORTED: hardcoded pointer type','fp8_e5m2':'UNSUPPORTED: hardcoded pointer type'},'replacement':'Supplemental TMA-family implementation only; cannot label B200-exclusive optimized BMM'}
,{'op':'group_gemm','source':'official Triton08 grouped TMA','architecture':'SM90+ TMA portable path; no SM100-specific WS claim','dtypes':{'fp16':'source candidate','fp8_e4m3fn':'source candidate','fp32':'UNSUPPORTED: hardcoded pointer type','fp8_e5m2':'UNSUPPORTED: hardcoded pointer type'},'replacement':'Supplemental TMA-family implementation; same uniform row-count groups as baseline, metadata layout changed explicitly'}
],'flag_gems_native_route':{'installed_commit':'30bfcc0735b8ebd48bd7339ef752ec64df69663f','newer_audited_commit':'32e4a4cf9b8618185a13eb82d72687fcdbbfb982','arch_maps':{'8':'ampere','9':'hopper'},'major10_arch_route':False,'newer_generic_cluster_path':'HAS_TLE and capability.major>=9, only FP16, size/layout gates; not SM100-exclusive; cannot silently transfer results or require new environment','hopper_mm_ws':'H20-specific selector requires properties.major==9; cannot present its Hopper path as native B200 dispatch'},'main_clc':{'commit':'93035c4472eb820f78074de0e28341fa321a759d','source':'09-persistent-matmul.py lines454-537, explicit SM100+ CLC and clc=True','installed_runtime_has_clc_option':False,'selected_for_execution':False,'reason':'Installed3.6 CUDAOptions has no clc field; no environment changes authorized; archive/reference only'},'licenses':{'Triton09_06':'MIT, full license archived','Triton08':'MIT with NVIDIA2023-2025 notice retained in full original source','FlagGems':'Apache2.0, full license archived'},'not_covered':'No claim that all25 ops have Blackwell-exclusive upstream implementations or that fixed tutorial configs are fastest. CUTLASS/CuTeDSL and blockscaled MXFP8 excluded from primary cohort.'}
results['attention_corrected_kernel_triton_api_calls_equal_to_v360']=True
results['attention_triton_api_call_names']=old_calls
(ROOT/'source_audit.json').write_bytes((json.dumps(results,indent=2)+'\n').encode())
print(json.dumps({k:results[k] for k in ['verification_status','upstream_files_verified','exact_ast_function_count','plan_equality_with_existing_cohort']},indent=2));print('old FP8 layout static affected shapes:',len(old_violations))
