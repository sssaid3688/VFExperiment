"""UNRUN FLA native pure-Triton branch candidates, preallocated output.

No Torch/Triton/FLA import at module import or shape_plan. Root alone calls
build_case in its serial GPU owner. Math kernels AND native decorators unchanged.
"""
import functools,hashlib,importlib.util,json,os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'fla_native_sources'
COMMIT='9d478facb282410dc150051d5f57205cd3812f31'
OPS=('group_rmsnorm','causal_conv1d')

@functools.lru_cache(None)
def _baseline(op):
 name='norm_rope_ops.py' if op=='group_rmsnorm' else 'matrix_attention_ops.py'
 path=ROOT.parent/'b200_remaining'/name
 spec=importlib.util.spec_from_file_location('_exp3_fla_native_baseline_'+op,path)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return module

def shape_plan(op):
 if op not in OPS:raise ValueError(op)
 return _baseline(op).shape_plan(op)

def _check_policy():
 allowed=Path('/home/ubuntu/workspace/oyhj/papers2').resolve()
 for key in ('TRITON_CACHE_DIR','FLAGGEMS_CACHE_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR'):
  value=os.environ.get(key)
  if not value or not Path(value).resolve().is_relative_to(allowed):raise RuntimeError('Set restricted cache before imports: '+key)
 if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('PYTHONDONTWRITEBYTECODE=1 required')
 if os.environ.get('FLA_CACHE_MODE','disabled')!='disabled':
  raise NotImplementedError('FLA_CACHE_MODE is nondefault; this candidate preserves the official default native search, never silently changes requested cache/dispatch policy')
 if any(os.environ.get(k,'0')!='0' for k in ('FLA_GLUON','FLA_CONV_GLUON')):
  raise NotImplementedError('Optional Gluon backend explicitly requested; candidate is limited to original pure-Triton fallback, not a complete backend-registry implementation')
 if os.environ.get('FLA_CONV_BACKEND','triton')!='triton':
  raise NotImplementedError('Non-Triton convolution backend requested; no CUDA/mix fallback is admitted')

def _manifest():return json.loads((BUNDLE/'manifest.json').read_text())

@functools.lru_cache(None)
def _module(op):
 manifest=_manifest();name='group_native_kernels.py' if op=='group_rmsnorm' else 'conv_native_kernels.py'
 # Validate whole official files and isolated decorated sources before execution.
 for entry in list(manifest['files'].values())+list(manifest['extractions'].values()):
  path=BUNDLE/entry['path']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('FROZEN_SOURCE_MISMATCH '+str(path))
 path=BUNDLE/name;spec=importlib.util.spec_from_file_location('_exp3_fla_native_'+op,path)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return module

def _sources(op):
 m=_manifest();name='group_native_kernels.py' if op=='group_rmsnorm' else 'conv_native_kernels.py'
 original=m['extractions'][name]['original_path']
 relevant=[original,'fla/utils/env.py','fla/utils/hardware.py','fla/backends.py']
 if op=='causal_conv1d':relevant+=['fla/ops/utils/cache.py','fla/modules/causal_conv1d/__init__.py']
 out=[]
 for rel in relevant:
  e=m['files'][rel];p=BUNDLE/rel
  if hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:raise RuntimeError('Source bytes changed '+rel)
  out.append(dict(path=str(p),sha256=e['sha256'],upstream_commit=COMMIT,
   url=e['url'].replace('raw.githubusercontent.com','github.com').replace('/'+COMMIT+'/', '/blob/'+COMMIT+'/'),
   role='Exact official source; math/dispatch/tuning or transitive helper evidence',license='MIT'))
 e=m['extractions'][name];p=BUNDLE/name
 out.append(dict(path=str(p),sha256=e['sha256'],upstream_commit=COMMIT,
  url=e['functions'][0]['url'],role='Exact original function AST including original decorators; isolated import plumbing only; generated module bytes differ from full upstream file',
  functions=e['functions'],license='MIT'))
 return out

def build_case(op,dtype,index,device='cuda:0'):
 if op not in OPS:raise ValueError(op)
 if dtype not in ('fp32','fp16','fp8_e4m3fn','fp8_e5m2'):raise NotImplementedError('Unknown storage dtype; no dtype emulation')
 _check_policy()
 case=_baseline(op).build_case(op,dtype,index,device)
 return wrap_case(case,op)

def wrap_case(case,op):
 if op not in OPS:raise ValueError(op)
 _check_policy()
 import torch,triton
 target=triton.runtime.driver.active.get_current_target()
 if target.backend!='cuda':raise NotImplementedError('CUDA-only native NVIDIA rule adapter; Intel/AMD/native backend policies must not be guessed')
 native=_module(op);old=case['contract'];x=case['device_inputs']['x']
 if not x.is_contiguous():raise NotImplementedError('Original selected layout contract requires contiguous input; no hidden timed conversion')
 if op=='group_rmsnorm':
  t,g,d=case['shape'];rows=t*g;w=case['device_inputs']['weight'];y=case['outputs']['normalized'];rstd=case['outputs']['rstd']
  if tuple(w.shape)!=(g,d) or tuple(y.shape)!=(t,g,d) or tuple(rstd.shape)!=(rows,) or rstd.dtype!=torch.float32:
   raise NotImplementedError('Group weights/output/stat contract differs from audited original case')
  block=triton.next_power_of_2(d);max_fused=65536//x.element_size()
  if d>min(max_fused,block):raise NotImplementedError('Native FLA feature dimension limit exceeded')
  # Exact CUDA case of original layer_norm_fwd: IS_INTEL=False.
  multiline=d<=512;nb=triton.cdiv(rows,2048)
  args=dict(x=x,y=y,w=w,b=None,res=None,res_out=None,mean=None,rstd=rstd,eps=1e-6,
   G=g,D=d,BD=block,IS_RMS_NORM=True,HAS_RESIDUAL=False,STORE_RESIDUAL_OUT=False,HAS_WEIGHT=True,HAS_BIAS=False)
  if multiline:
   kernel=native.layer_norm_fwd_kernel
   def grid(meta):return (triton.cdiv(rows,meta['BT']),)
   def fn():kernel[grid](**args,T=rows,NB=nb)
   path='layer_norm_fwd_kernel';search={'BT':[32,64,128],'num_warps':[2,4,8],'configs':9}
  else:
   kernel=native.layer_norm_fwd_kernel_row
   def fn():kernel[(rows,)](**args)
   path='layer_norm_fwd_kernel_row';search={'num_warps':[2,4,8,16],'configs':4}
  details=dict(native_dispatch_condition='D<=512 and not(D==512 and IS_INTEL); CUDA IS_INTEL=False',
   native_multiline_selected=multiline,native_kernel_path=path,T_rows=rows,NB=nb,BD=block,
   native_search=search,stats='Existing FP32 rstd[T*G] checked with unchanged per-output FP32 tolerance; no mean/residual output for RMS',
   group_semantics='Contiguous [T,G,D] viewed by pointer as T*G rows; original row_id%G selects original [G,D] weights inside each kernel. No weighting or normalization outside timing.')
 else:
  b,t,d,width=case['shape'];weight=case['device_inputs']['weight'];bias=case['device_inputs']['bias'];y=case['outputs']['y']
  if tuple(weight.shape)!=(d,width) or tuple(y.shape)!=(b,t,d):raise NotImplementedError('Causal convolution layout/weight contract mismatch')
  bt=64;bw=triton.next_power_of_2(width);nt=triton.cdiv(t,bt);kernel=native.causal_conv1d_fwd_kernel
  def grid(meta):return (triton.cdiv(d,meta['BD']),nt,b)
  args=dict(x=x,y=y,weight=weight,bias=bias,residual=None,cu_seqlens=None,initial_state=None,chunk_indices=None,
   B=b,T=t,stride_x_n=x.stride(0),stride_x_t=x.stride(1),stride_x_d=x.stride(2),D=d,W=width,BT=bt,BW=bw,ACTIVATION=None)
  def fn():kernel[grid](**args)
  details=dict(native_dispatch_condition='Original public backend=triton, no CP/varlen/initial_state/final_state/activation/residual; chunk_size64',
   native_kernel_path='causal_conv1d_fwd_kernel',native_grid='[ceil(D/native_BD),ceil(T/64),B]',BT=bt,BW=bw,
   native_search={'BD':[16,32,64,128],'num_warps':[4,8,16,32],'configs':16},
   native_feature_heuristics='Original HAS_WEIGHT/HAS_BIAS/HAS_RESIDUAL/USE_INITIAL_STATE/IS_VARLEN decorators retained; derived from actual pointers',
   layout='Original [B,T,D] input and contiguous [D,W] packed weight retained, prepared outside fn',
   rounding='Original FP32 accumulation and final rtne requested-dtype store unchanged; no fp16 substitution for FP8')
 case['fn']=fn;case['sources']=[*case['sources'],*_sources(op)]
 case['contract']={**old,**details,'baseline':'UNRUN candidate: original native pure-Triton shape/tuning selection, not B200-specific optimization',
  'previous_fixed_launch':old.get('launch'),'launch':'Original decorated kernel chooses native configuration; capture actual best_config/compiled metadata after first correctness call',
  'source_policy':'Fixed official FLA9d478f. Kernel function AST, signature, body and complete original decorators unchanged. Original env.py/cache.py helpers loaded in isolation; no complete FLA import or installation.',
  'dispatcher_scope':'Reproduce original CUDA pure-Triton default shape selection into preallocated outputs. Optional multi-backend registry/CUDA/mix/Gluon replacements are not reproduced and explicit requests are rejected.',
  'FLA_CACHE_MODE':os.environ.get('FLA_CACHE_MODE','disabled'),'FLA_CACHE_RESULTS':os.environ.get('FLA_CACHE_RESULTS','1'),
  'autotune_cache_kwargs':native.autotune_cache_kwargs,
  'same_contract':'All old100 shapes/index/CPU seed/actual storage tensors/independent CPUfp64 refs/all returned outputs/stats/tolerances/useful+additional bytes retained exactly, only fn and dispatch metadata change.',
  'timed_scope':'One direct original decorated forward Triton kernel after initial autotune. Outputs preallocated; no Torch native GPU compute, allocation, hidden casts, backward, copy or state helper in fn.',
  'candidate_status':'SOURCE_PREPARED_ONLY: audit agent never built GPU case, compiled, checked numeric output or timed it'}
 def tuning_snapshot():
  obj=kernel;chain=[]
  for _ in range(8):
   item={'type':type(obj).__name__}
   configs=getattr(obj,'configs',None)
   if configs is not None:item['declared_configs']=len(configs)
   best=getattr(obj,'best_config',None)
   if best is not None:item['last_observed_best_config']={'kwargs':best.kwargs,'num_warps':best.num_warps,'num_stages':best.num_stages,'num_ctas':getattr(best,'num_ctas',None)}
   chain.append(item)
   if not hasattr(obj,'fn'):break
   obj=obj.fn
  return {'kernel_path':details['native_kernel_path'],'wrapper_chain':chain,
   'scope':'CPU object snapshot after root correctness call; capture immediately per-case. Compiled cache/IR remains authoritative for actual launch.'}
 case['native_tuning_metadata']=tuning_snapshot
 return case
