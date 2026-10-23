"""UNRUN candidates: restore upstream tuning / SM90+ TMA, never claim B200-only.

Imports and shape_plan are CPU/source-only. Only root's serial runner may call
build_case. Existing baseline inputs, reference, tolerance and outputs are reused.
No environment/package modification and no timed allocation/native Torch compute.
"""
import functools,hashlib,importlib,importlib.util,os,sys
from pathlib import Path

COMMIT='30bfcc0735b8ebd48bd7339ef752ec64df69663f'
ROOT=Path(__file__).resolve().parent
OPS=('dense_matmul','batched_matmul','group_gemm')
EXPECTED={
 'mm':'b2a5eb649a6c2ae251f340130249b3b4b5dd862bbcd06df0c3d1349043bfd91c',
 'bmm':'7ed3048cc3b4960b5890fd0df98a6ec44d0655719d773243a347acc2ed1c9cb6',
 'group_gemm':'863269a05e42fd07d839880c3f4de65f2312f2763226d7a4ce834241766534b1'}

@functools.lru_cache(None)
def _baseline():
 path=ROOT.parent/'b200_remaining'/'matrix_attention_ops.py'
 spec=importlib.util.spec_from_file_location('_b200_audit_existing_matrix_baseline',path)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return module

def shape_plan(op):
 if op not in OPS:raise ValueError(op)
 return _baseline().shape_plan(op)

def _check_caches():
 allowed=Path('/home/ubuntu/workspace/oyhj/papers2').resolve()
 for key in ('TRITON_CACHE_DIR','FLAGGEMS_CACHE_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR'):
  value=os.environ.get(key)
  if not value or not Path(value).resolve().is_relative_to(allowed):raise RuntimeError('Set restricted cache before imports: '+key)
 if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('PYTHONDONTWRITEBYTECODE=1 required')

def build_case(op,dtype,index,device='cuda:0'):
 if op not in OPS:raise ValueError(op)
 _check_caches()
 # Root owns the device. This call allocates the exact existing baseline case,
 # performs CPU fp64 references, and does not launch its old fn.
 case=_baseline().build_case(op,dtype,index,device)
 return wrap_case(case,op)

def wrap_case(case,op):
 """Replace only callable/dispatch labels; all measurement tensors stay identical."""
 _check_caches()
 if op not in OPS:raise ValueError(op)
 import torch,triton
 name={'dense_matmul':'mm','batched_matmul':'bmm','group_gemm':'group_gemm'}[op]
 mod=importlib.import_module('flag_gems.ops.'+name);source=Path(mod.__file__)
 if hashlib.sha256(source.read_bytes()).hexdigest()!=EXPECTED[name]:raise RuntimeError('Pinned upstream source mismatch')
 A=case['device_inputs']['a'];B=case['device_inputs']['b'];C=case['outputs']['y']
 if op=='dense_matmul':
  M,N,K=case['shape']
  def grid(meta):return (triton.cdiv(M,meta['BLOCK_M'])*triton.cdiv(N,meta['BLOCK_N']),)
  def fn():mod.mm_kernel_general[grid](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride(),GROUP_M=8)
  entry='flag_gems.ops.mm.mm_kernel_general with original LibTuner wrappers'
  dispatch='Native public general_mm configuration selection; runtime.get_tuned_config("mm"). SYRK is inapplicable to independent A/B; StreamK requires major==8 and is inapplicable on SM100.'
  feature='generic NVIDIA autotuning, no B200-specific path'
 elif op=='batched_matmul':
  batch,M,N,K=case['shape']
  def grid(meta):return (triton.cdiv(M,meta['TILE_M']),triton.cdiv(N,meta['TILE_N']),batch)
  def fn():mod.bmm_kernel[grid](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride())
  entry='flag_gems.ops.bmm.bmm_kernel with original LibTuner + heuristics'
  dispatch='Native bmm candidate set plus DIVISIBLE_M/N/K heuristics retained. USE_FLAGTUNE recorded exactly, never changed here.'
  feature='generic NVIDIA autotuning, no B200-specific path'
 else:
  G,R,K,N=case['shape'];M=G*R;offs=case['device_inputs']['group_ends']
  capability=torch.cuda.get_device_capability(A.device)
  if capability[0]<9 or not mod._support_host_tensor_descriptor:raise NotImplementedError('Upstream group_mm host TensorDescriptor TMA gate not satisfied')
  # Matches public group_mm descriptor construction and NUM_SMS launch exactly.
  dummy=[1,1]
  a_desc=mod.TensorDescriptor(A,A.shape,A.stride(),dummy)
  b_desc=mod.TensorDescriptor(B,[G*K,N],[B.stride(1),B.stride(2)],dummy)
  c_desc=mod.TensorDescriptor(C,C.shape,C.stride(),dummy)
  sms=torch.cuda.get_device_properties(A.device).multi_processor_count
  def fn():mod.grouped_mm_tma_kernel[(sms,)](a_desc,b_desc,c_desc,C,offs,G,M,N,K,*C.stride())
  entry='flag_gems.ops.group_gemm.grouped_mm_tma_kernel with original 7-config LibTuner and descriptor block-shape pre-hook'
  dispatch='Exact public group_mm gate: supports_tma(capability major>=9) and host TensorDescriptor available. Preallocated C replaces public output allocation; descriptors created outside timing.'
  feature='SM90+ TMA hardware-feature path, not B200-exclusive; actual TMA lowering must be evidenced by root compile/IR gate'
  case['contract']={**case['contract'],'detected_capability':list(capability),'native_grid_sms':sms,
   'native_TMA_metadata':'Three host descriptors; block shapes set by original per-config pre-hook; construction outside timing; launch-side pre-hook remains original. No staging FP16 conversion.'}
 old_contract=case['contract'];case['fn']=fn
 case['contract']={**old_contract,'baseline':'UNRUN candidate: native upstream selection / hardware feature path; not a B200-specific optimization claim',
  'previous_fixed_launch':old_contract.get('launch'),'launch':'Upstream tuner selects config at first correctness/warmup call; record actual compiled metadata after serial gate',
  'entrypoint':entry,'dispatch_evidence':dispatch,'hardware_feature_classification':feature,
  'source_transform':'No source edits and no .fn unwrapping; original decorators/tuning/pre-hooks retained',
  'USE_FLAGTUNE':os.environ.get('USE_FLAGTUNE'),
  'timed_scope':'Only original wrapped Triton kernel launch into existing preallocated C; no public Python allocator/native Torch GPU compute/backward',
  'candidate_status':'CPU/source-only prepared; not compiled, numerically validated, or timed by audit agent',
  'same_contract':'Existing baseline cpu_inputs/device_inputs/outputs/refs/tolerances/traffic and FP8 storage retained exactly. FP32 dot explicitly allow_tf32=False in unchanged upstream kernels.'}
 case['sources']=[*case['sources'],{'path':str(source),'sha256':EXPECTED[name],'upstream_commit':COMMIT,
  'url':f'https://github.com/flagos-ai/FlagGems/blob/{COMMIT}/src/flag_gems/ops/{name}.py',
  'role':'Native wrapper/config selection, exact official source; no claim of B200-specific tuning'}]
 return case
