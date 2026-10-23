"""Unmeasured authoritative Triton architecture candidates, separate cohort.

Import is CPU safe. build_case is deliberately reserved for the root GPU runner.
Never import whole tutorials (their top-level code initializes CUDA/cuBLAS).
"""
import functools,hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OPS=['dense_matmul','batched_matmul','group_gemm','sdpa']

def shape_plan(op):
 if op=='dense_matmul':return [[m,n,[64,96,128,192,256][j%5]] for m in [16,32,48,64,96,128,192,256,384,512] for j,n in enumerate([32,64,96,128,192,256,320,384,448,512])]
 if op=='batched_matmul':return [[b,m,n,[64,128,256,512][j]] for b in [1,2,4,8,16] for m in [16,32,64,96,128] for j,n in enumerate([32,64,96,128])]
 if op=='group_gemm':return [[g,r,k,n] for g in [2,4,8,16,32] for r in [8,16,32,64,128] for k in [64,128] for n in [64,128]]
 if op=='sdpa':return [[b,h,s,d] for b in [1,2,3,4,5] for h in [1,2,3,4,5] for s in [512,1024] for d in [64,128]]
 raise ValueError(op)

@functools.lru_cache(None)
def _provenance(name):
 return next(r for r in json.loads((ROOT/'extraction_manifest.json').read_text()) if r['name']==name)

@functools.lru_cache(None)
def _module(name):
 r=_provenance(name);p=ROOT/r['extracted_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['extracted_sha256']
 module_name='_exp3_sm100_'+name;spec=importlib.util.spec_from_file_location(module_name,p);mod=importlib.util.module_from_spec(spec);sys.modules[module_name]=mod;spec.loader.exec_module(mod);return mod

def _sources(name):
 r=_provenance(name);src=r['original'];p=ROOT/src['archive'];assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256']
 common={'commit':src['commit'],'upstream_commit':src['commit'],'url':src['url'],'source_url':src['url']}
 return [{'path':str(p),'sha256':src['sha256'],'role':'byte-exact official upstream source',**common},{'path':str(ROOT/r['extracted_file']),'sha256':r['extracted_sha256'],'role':'exact function AST; decorators only replaced','transform':r['transform'],**common}]

def _scratch(torch,triton,device):
 # Device descriptor construction requests workspace on every launch. This
 # callback always returns a single preallocated aligned buffer: no torch
 # allocation, view, arithmetic, or transfer occurs in the timed callable.
 # Root runner is serial; sharing the workspace among sequential launches on
 # one stream follows the upstream allocator protocol. Concurrent calls are
 # intentionally unsupported. Fail rather than allocate during timing.
 buffer=torch.empty(64*1024*1024,dtype=torch.uint8,device=device)
 def allocate(size,alignment,stream):
  assert size<=buffer.numel(),('Preallocated descriptor workspace too small',size)
  assert buffer.data_ptr()%alignment==0,('Workspace alignment',alignment)
  return buffer
 triton.set_allocator(allocate)
 return buffer

def build_case(op,dtype,index,device='cuda:0',variant=None):
 if op not in OPS or not 0<=index<100:raise ValueError((op,index))
 if op=='sdpa' and dtype not in ('fp16','fp8_e5m2'):
  raise NotImplementedError('Official 06 kernel explicitly computes/casts FP16 or E5M2 output and probabilities; FP32/E4M3FN are different contracts, not substituted.')
 if op in ('batched_matmul','group_gemm') and dtype not in ('fp16','fp8_e4m3fn'):
  raise NotImplementedError('Official 08 TMA kernel hardcodes tl.pointer_type(float16 or float8e4nv); FP32/E5M2 cannot be reinterpreted safely.')
 import torch,triton
 from triton.tools.tensor_descriptor import TensorDescriptor
 if torch.cuda.get_device_capability(device)[0]!=10:raise NotImplementedError('This separate candidate cohort requires SM100-family Blackwell (major10).')
 dt={'fp32':torch.float32,'fp16':torch.float16,'fp8_e4m3fn':torch.float8_e4m3fn,'fp8_e5m2':torch.float8_e5m2}[dtype]
 f8=dtype.startswith('fp8');shape=shape_plan(op)[index];gen=torch.Generator(device='cpu').manual_seed(20261020+index);cpu={};gpu={};keepers=[]
 def rand(name,size):
  c=torch.randn(size,generator=gen,dtype=torch.float32).to(dt);cpu[name]=c;gpu[name]=c.to(device);return c,gpu[name]
 def meta(name,c):cpu[name]=c;gpu[name]=c.to(device);return gpu[name]
 rtol,atol=(2e-5,1e-4) if dtype=='fp32' else ((2e-3,2e-3) if dtype=='fp16' else ((.07,.05) if dtype=='fp8_e4m3fn' else (.14,.1)))
 contract={'cohort':'Separate unmeasured upstream architecture candidate; no performance or numeric qualification inferred from baseline','source_transform':'Exact function AST; remove autotune/profiler decorators only; fixed prespecified launch config','timed_scope':'Direct Triton launch only. Packing, allocation, references, metadata and descriptors prepared outside timer. No torch arithmetic inside fn.','fp8_policy':'True unscaled float8 storage; no MXFP8/blockscale substitution; FP32 accumulation; requested output storage.','config_selection':'A legal member of the upstream tutorial config list chosen before any GPU measurements; not a best-achievable tuning claim.','input_generation':'CPU FP32 seed20261020+index, then cast to actual dtype; CPU FP64 reference uses these rounded inputs.'}
 outputs={};refs={};extra=0;tols={};workspace=None
 if op=='dense_matmul':
  M,N,K=shape;ac,A=rand('a',(M,K));bc,B=rand('b',(K,N));packed=bc.T.contiguous();cpu['b']=packed;gpu['b']=packed.to(device);B=gpu['b'];C=torch.empty((M,N),device=device,dtype=dt)
  # Host descriptor kernel has explicit E4M3FN/FP16 output; device descriptor
  # kernel obtains output dtype from pointer and can be probed for FP32/E5.
  variant=variant or ('host_tma_ws' if dtype in ('fp16','fp8_e4m3fn') else 'device_tma_ws')
  if variant not in ('host_tma_ws','device_tma_ws'):raise ValueError(variant)
  if variant=='host_tma_ws' and dtype not in ('fp16','fp8_e4m3fn'):raise NotImplementedError('Host TMA output is hardcoded FP16/E4M3FN')
  sm=torch.cuda.get_device_properties(device).multi_processor_count
  cfg=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=64,GROUP_SIZE_M=8,EPILOGUE_SUBTILE=False,NUM_SMS=sm,WARP_SPECIALIZE=True,num_warps=4,num_stages=3,default_dot_input_precision='ieee')
  grid=(min(sm,triton.cdiv(M,128)*triton.cdiv(N,128)),);mod=_module('mm_ws')
  if variant=='host_tma_ws':
   args=(TensorDescriptor(A,[M,K],[K,1],[128,64]),TensorDescriptor(B,[N,K],[K,1],[128,64]),TensorDescriptor(C,[M,N],[N,1],[128,128]),M,N,K)
   cfg['FP8_OUTPUT']=f8;ker=mod.matmul_kernel_tma_persistent
  else:
   workspace=_scratch(torch,triton,device);args=(A,B,C,M,N,K);cfg['FLATTEN']=True;ker=mod.matmul_kernel_descriptor_persistent
  fn=lambda:ker[grid](*args,**cfg)
  refs={'y':ac.double()@bc.double()};outputs={'y':C};useful=(M*K+K*N+M*N)*A.element_size();source='mm_ws'
  contract.update(formula='C[M,N]=A[M,K] @ B_logical[K,N]',variant=variant,architecture_evidence='Official09 lines398-401: warp specialization only works on Blackwell, with async warp scheduling; descriptor flatten=True and TMA persistent tile scheduler',layout_adapter='B[K,N] transposed to contiguous B_T[N,K] outside timing, same numerical inputs; packing excluded',launch={**cfg,'grid':grid},input_dtype=str(dt),output_dtype=str(dt),accumulator_dtype='fp32',fp32_policy='Backend default_dot_input_precision=ieee supplied explicitly. Must confirm generated TTIR/PTX after root smoke; torch TF32 flags alone insufficient.',upstream_demonstrated_dtypes=['fp16','fp8_e4m3fn'],experimental_dtype_probes=['fp32','fp8_e5m2'],flops=2*M*N*K)
 elif op in ('batched_matmul','group_gemm'):
  if variant not in (None,'group_tma'):raise ValueError(variant)
  if op=='batched_matmul':
   G,M,N,K=shape;ac,A=rand('a',(G,M,K));bc,B=rand('b',(G,K,N));C=torch.empty((G,M,N),device=device,dtype=dt);ref=ac.double()@bc.double()
  else:
   G,M,K,N=shape;ac,A=rand('a',(G*M,K));bc,B=rand('b',(G,K,N));C=torch.empty((G*M,N),device=device,dtype=dt);ref=(ac.reshape(G,M,K).double()@bc.double()).reshape(G*M,N)
  packed=bc.transpose(-1,-2).contiguous();cpu['b']=packed;gpu['b']=packed.to(device);B=gpu['b'];esize=A.element_size()
  ptr_a=meta('a_ptrs',torch.tensor([A.data_ptr()+g*M*K*esize for g in range(G)],dtype=torch.int64))
  ptr_b=meta('b_ptrs',torch.tensor([B.data_ptr()+g*N*K*esize for g in range(G)],dtype=torch.int64))
  ptr_c=meta('c_ptrs',torch.tensor([C.data_ptr()+g*M*N*esize for g in range(G)],dtype=torch.int64))
  sizes=meta('gemm_sizes',torch.tensor([M,N,K]*G,dtype=torch.int32));lds=meta('gemm_lds',torch.tensor([K,K,N]*G,dtype=torch.int32))
  sm=torch.cuda.get_device_properties(device).multi_processor_count;workspace=_scratch(torch,triton,device)
  cfg=dict(NUM_SM=sm,BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=64,FP8=f8,num_warps=4,num_stages=3,default_dot_input_precision='ieee')
  ker=_module('group_tma').grouped_matmul_tma_kernel;fn=lambda:ker[(sm,)](ptr_a,ptr_b,ptr_c,sizes,lds,G,**cfg)
  outputs={'y':C};refs={'y':ref};useful=G*(M*K+K*N+M*N)*esize+G*(3*8+6*4);source='group_tma'
  contract.update(formula='Independent C[g]=A[g]@B[g]; grouped row-count layout reinterpreted as disjoint row views without arithmetic',variant='official08_group_tma',architecture_evidence='Official08 TMA requires CUDA>=SM90; compatible Blackwell candidate, NOT Blackwell-specific warp-specialized BMM/GroupGEMM',architecture_specificity='TMA-family optimization only; cannot close strict SM100-exclusive coverage claim for BMM/GroupGEMM',metadata='Device pointer arrays int64, GEMM sizes and strides int32 constructed before timing; all metadata reads included in useful bytes',layout_adapter='B[G,K,N] packed B_T[G,N,K] outside timing; A/C group pointers are contiguous disjoint views',launch=cfg,input_dtype=str(dt),output_dtype=str(dt),accumulator_dtype='fp32',flops=2*G*M*N*K)
 else:
  if variant not in (None,'host_attention_ws'):raise ValueError(variant)
  B,H,S,D=shape;qc,Q=rand('q',(B,H,S,D));kc,K=rand('k',(B,H,S,D));vc,V=rand('v',(B,H,S,D));O=torch.empty((B,H,S,D),device=device,dtype=dt);LSE=torch.empty((B,H,S),device=device,dtype=torch.float32)
  bm,bn=128,64;ydim=B*H*S
  dq=TensorDescriptor(Q,[ydim,D],[D,1],[bm,D]);dk=TensorDescriptor(K,[ydim,D],[D,1],[bn,D]);do=TensorDescriptor(O,[ydim,D],[D,1],[bm,D])
  if f8:
   packed=vc.transpose(-1,-2).contiguous();cpu['v']=packed;gpu['v']=packed.to(device);V=gpu['v']
   dv=TensorDescriptor(V,[B*H*D,S],[S,1],[D,bn]);source='attention_ws_corrected'
  else:dv=TensorDescriptor(V,[ydim,D],[D,1],[bn,D]);source='attention_ws_v360'
  cfg=dict(HEAD_DIM=D,BLOCK_M=bm,BLOCK_N=bn,FP8_OUTPUT=f8,STAGE=1,warp_specialize=True,IS_HOPPER=False,num_warps=4,num_stages=2,maxnreg=168 if D==128 and not f8 else 80,default_dot_input_precision='ieee')
  ker=_module(source)._attn_fwd;fn=lambda:ker[(triton.cdiv(S,bm),B*H,1)](D**-.5,LSE,B,H,dq,dk,dv,do,N_CTX=S,**cfg)
  scores=(qc.double()@kc.double().transpose(-1,-2))*(D**-.5);ref=torch.softmax(scores,-1)@vc.double();reflse=torch.logsumexp(scores,-1)/math.log(2);del scores
  outputs={'y':O,'lse_log2':LSE};refs={'y':ref,'lse_log2':reflse};useful=4*B*H*S*D*Q.element_size();extra=4*B*H*S
  tols={'lse_log2':{'rtol':2e-5,'atol':2e-6}}
  if f8:rtol,atol=.30,.05
  contract.update(formula='O=softmax(Q@K.T/sqrt(D))@V; dense noncausal, noGQA/dropout/mask',variant='host_attention_ws',architecture_evidence='Official06 is_blackwell major==10 + maxnreg168/80; inner warp_specialize loop and Blackwell accumulator reshape branch',launch=cfg,input_dtype=str(dt),output_dtype=str(dt),accumulator_dtype='fp32 QK and PV; probability p rounded to input/output dtype before PV',fp8_layout='E5M2 V packed as [B,H,D,S], corrected upstream descriptor[B*H*D,S]; no input/output scaling',fp8_source_reason='v3.6.0 transposed-V descriptor can index outside descriptor bounds for multihead; fixed main commit selected unchanged for E5 only',shape_constraints='D in64/128 here, S512/1024 multiple of BLOCK_M128 and BLOCK_N64; all100 shapes paired to baseline',flops=4*B*H*S*S*D)
 if workspace is not None:keepers.append(workspace)
 contract['workspace']={'preallocated_bytes':64*1024*1024 if workspace is not None else 0,'timed_allocation':False,'scope':'Descriptor scratch traffic and descriptor encoding are not included in useful algorithmic bytes. Device-descriptor candidates may have extra descriptor work; report separately.'}
 # Keep descriptors/workspace and the global allocator callback alive throughout
 # timing. Rebinding allocator between cases is allowed only in serial runner.
 return dict(fn=fn,outputs=outputs,refs=refs,cpu_inputs=cpu,device_inputs=gpu,useful_bytes=useful,additional_bytes=extra,shape=shape,contract=contract,sources=_sources(source),rtol=rtol,atol=atol,output_tolerances=tols,_keepers=keepers)
