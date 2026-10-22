"""Fixed-configuration authoritative Triton leaves; no native compute in fn.

Selected FlagGems files are byte-identical to official v5.0.2. FLA convolution
keeps the exact upstream function body with explicit configuration. Fixed
tiles are a disclosed study baseline, not a best-achievable B200 claim.
"""
import functools,hashlib,importlib,importlib.util,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT/'matrix_upstream'
OPS=['dense_matmul','batched_matmul','group_gemm','causal_conv1d','sdpa','kv_cache_store']

def shape_plan(op):
 if op=='dense_matmul':
  return [[m,n,[64,96,128,192,256][j%5]] for m in [16,32,48,64,96,128,192,256,384,512] for j,n in enumerate([32,64,96,128,192,256,320,384,448,512])]
 if op=='batched_matmul':
  return [[b,m,n,[64,128,256,512][j]] for b in [1,2,4,8,16] for m in [16,32,64,96,128] for j,n in enumerate([32,64,96,128])]
 if op=='group_gemm':
  return [[g,rows,k,n] for g in [2,4,8,16,32] for rows in [8,16,32,64,128] for k in [64,128] for n in [64,128]]
 if op=='causal_conv1d':
  return [[b,t,d,w] for b in [1,2] for t in [16,32,64,128,256] for d in [32,64,96,128,256] for w in [3,4]]
 if op=='sdpa':
  return [[b,h,s,d] for b in [1,2,3,4,5] for h in [1,2,3,4,5] for s in [512,1024] for d in [64,128]]
 if op=='kv_cache_store':
  return [[t,h,d,p] for t in [16,32,64,128,256] for h in [1,2,4,8,16] for d in [64,128] for p in [16,32]]
 raise ValueError(op)

@functools.lru_cache(None)
def _manifest():return json.loads((UPSTREAM/'provenance.json').read_text())

@functools.lru_cache(None)
def _module(name):
 if name=='causal_conv1d':
  path=UPSTREAM/'fla_causal_conv1d_kernel.py';module_name='_exp3_exact_fla_causal_conv1d'
  spec=importlib.util.spec_from_file_location(module_name,path);mod=importlib.util.module_from_spec(spec);sys.modules[module_name]=mod;spec.loader.exec_module(mod)
  return mod
 return importlib.import_module('flag_gems.'+('fused.reshape_and_cache_flash' if name=='reshape_and_cache_flash' else 'ops.'+name))

def _jit(obj):
 import triton
 for _ in range(10):
  if isinstance(obj,triton.runtime.jit.JITFunction):return obj
  obj=obj.fn
 raise RuntimeError('Could not select upstream JIT leaf without autotune wrappers')

def _sources(name):
 m=_manifest()[name];path=UPSTREAM/m['archive_file'] if name=='causal_conv1d' else Path(_module(name).__file__)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256'],('Pinned source differs',name,path)
 out=[{'path':str(path),'sha256':m['sha256'],'commit':m['commit'],'upstream_commit':m['commit'],'url':m['source_url'],'source_url':m['source_url'],'role':'byte-verified exact upstream source'}]
 if name=='causal_conv1d':
  p=UPSTREAM/m['extracted_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m['extracted_file_sha256']
  out.append({'path':str(p),'sha256':m['extracted_file_sha256'],'commit':m['commit'],'upstream_commit':m['commit'],'url':m['source_url'],'source_url':m['source_url'],'role':'exact function body with @triton.jit; only wrapper decorators replaced by explicit launch metadata','function_body_sha256':m['function_body_sha256']})
 return out

def build_case(op,dtype,index,device='cuda:0'):
 import torch,triton
 assert op in OPS and 0<=index<100
 dt={'fp16':torch.float16,'fp32':torch.float32,'fp8_e4m3fn':torch.float8_e4m3fn,'fp8_e5m2':torch.float8_e5m2}[dtype]
 f8=dtype.startswith('fp8');gen=torch.Generator(device='cpu').manual_seed(20261020+index)
 shape=shape_plan(op)[index];cpu={};gpu={}
 def rand(name,size):
  c=torch.randn(size,generator=gen,dtype=torch.float32).to(dt);cpu[name]=c;gpu[name]=c.to(device);return c,gpu[name]
 def meta(name,c):cpu[name]=c;gpu[name]=c.to(device);return gpu[name]
 rtol,atol=(2e-5,1e-4) if dtype=='fp32' else ((2e-3,2e-3) if dtype=='fp16' else ((.07,.05) if dtype=='fp8_e4m3fn' else (.14,.1)))
 contract={'baseline':'Fixed-configuration authoritative generic NVIDIA Triton, not B200-exclusive tuning','input_generation':'CPU fp32 generator seed20261020+shape_index, then CPU cast to true requested input dtype','timed_scope':'Only selected upstream Triton leaf; all allocations, CPU references, metadata and layout preparation excluded','source_transform':'Strip tuning/heuristic wrappers by selecting their existing JITFunction; original kernel body unchanged','fp8_policy':'True float8 input and output. Dot operators use native float8 tl.dot with fp32 accumulator; no harness fp16 substitution. Convolution computes fp32 from float8 storage; cache only copies storage.','precision_validation':'FP32 IEEE tl.dot allow_tf32=False explicit at every selected dot; independent CPU fp64 reference. FP32 matrix tolerance rtol2e-5/atol1e-4 prespecified for K<=512, not changed in response to observations.'}
 extra=0;out_tols={}
 if op=='dense_matmul':
  M,N,K=shape;ac,A=rand('a',(M,K));bc,B=rand('b',(K,N));C=torch.empty((M,N),dtype=dt,device=device)
  ker=_jit(_module('mm').mm_kernel_general);cfg={'BLOCK_M':64,'BLOCK_N':64,'BLOCK_K':32,'GROUP_M':8,'num_warps':4,'num_stages':3}
  fn=lambda:ker[(triton.cdiv(M,64)*triton.cdiv(N,64),)](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride(),**cfg)
  outputs={'y':C};refs={'y':ac.double()@bc.double()};useful=(M*K+K*N+M*N)*A.element_size();source='mm'
  contract.update(formula='C[M,N]=A[M,K]@B[K,N]',entrypoint='flag_gems.ops.mm.mm_kernel_general JIT leaf; bypasses public SYRK/StreamK branches',launch=cfg,output_dtype=str(dt),accumulator_dtype='fp32',flops=2*M*N*K)
 elif op=='batched_matmul':
  batch,M,N,K=shape;ac,A=rand('a',(batch,M,K));bc,B=rand('b',(batch,K,N));C=torch.empty((batch,M,N),dtype=dt,device=device)
  ker=_jit(_module('bmm').bmm_kernel);cfg={'TILE_M':64,'TILE_N':64,'TILE_K':32,'GROUP_M':1,'DIVISIBLE_M':M%64==0,'DIVISIBLE_N':N%64==0,'DIVISIBLE_K':K%32==0,'num_warps':4,'num_stages':3}
  fn=lambda:ker[(triton.cdiv(M,64),triton.cdiv(N,64),batch)](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride(),**cfg)
  outputs={'y':C};refs={'y':ac.double()@bc.double()};useful=batch*(M*K+K*N+M*N)*A.element_size();source='bmm'
  contract.update(formula='C[B,M,N]=batch_matmul(A[B,M,K],B[B,K,N]); no batch broadcast',entrypoint='flag_gems.ops.bmm.bmm_kernel JIT leaf',launch=cfg,output_dtype=str(dt),accumulator_dtype='fp32',flops=2*batch*M*N*K)
 elif op=='group_gemm':
  G,R,K,N=shape;M=G*R;ac,A=rand('a',(M,K));bc,B=rand('b',(G,K,N));ends=torch.arange(1,G+1,dtype=torch.int32)*R;offs=meta('group_ends',ends);C=torch.empty((M,N),dtype=dt,device=device)
  ker=_jit(_module('group_gemm').grouped_mm_kernel);cfg={'BLOCK_M':128,'BLOCK_N':64,'BLOCK_K':32,'GROUP_M':4,'num_warps':4,'num_stages':4}
  tiles=G*triton.cdiv(R,128)*triton.cdiv(N,64);grid=(min(tiles,148),)
  fn=lambda:ker[grid](A,B,C,offs,G,M,N,K,*A.stride(),B.stride(1),B.stride(2),*C.stride(),**cfg)
  refs={'y':torch.cat([ac[g*R:(g+1)*R].double()@bc[g].double() for g in range(G)],0)};outputs={'y':C};useful=(M*K+G*K*N+M*N)*A.element_size()+G*4;source='group_gemm'
  contract.update(formula='Y rows[gR:(g+1)R]=A rows[gR:(g+1)R] @ B[g,K,N]',metadata='A5 group row counts converted to cumulative row ends outside timing; int32 ends counted as reads',entrypoint='flag_gems.ops.group_gemm.grouped_mm_kernel non-TMA JIT leaf',launch={**cfg,'grid':grid},output_dtype=str(dt),accumulator_dtype='fp32',flops=2*M*N*K)
 elif op=='causal_conv1d':
  B,T,D,W=shape;xc,X=rand('x',(B,T,D));wc,Wlogical=rand('weight',(W,D));bc,Bias=rand('bias',(D,))
  # Preserve the logical A5 input [W,D] in the archive; both CPU and device
  # inputs below use the actual contiguous FLA layout [D,W] before timing.
  packed=wc.T.contiguous();cpu['weight']=packed;gpu['weight']=packed.to(device);Weight=gpu['weight']
  Y=torch.empty((B,T,D),dtype=dt,device=device);ref=bc.double().view(1,1,D).expand(B,T,D).clone()
  for w in range(W):
   lag=W-1-w;ref[:,lag:,:]+=xc[:,:T-lag,:].double()*wc[w].double()
  ker=_jit(_module('causal_conv1d').causal_conv1d_fwd_kernel)
  cfg={'D':D,'W':W,'BT':64,'BW':triton.next_power_of_2(W),'BD':64,'ACTIVATION':None,'HAS_WEIGHT':True,'HAS_BIAS':True,'HAS_RESIDUAL':False,'USE_INITIAL_STATE':False,'IS_VARLEN':False,'num_warps':4,'num_stages':3}
  fn=lambda:ker[(triton.cdiv(D,64),triton.cdiv(T,64),B)](X,Y,Weight,Bias,None,None,None,None,B,T,*X.stride(),**cfg)
  outputs={'y':Y};refs={'y':ref};useful=(2*B*T*D+D*W+D)*X.element_size();source='causal_conv1d'
  if dtype=='fp32':rtol,atol=2e-5,2e-6
  if f8:rtol,atol=(.07,.02) if dtype=='fp8_e4m3fn' else (.14,.04)
  contract.update(formula='y[b,t,d]=bias[d]+sum_w x[b,t-(W-1-w),d]*weight[w,d], zero causal left padding',entrypoint='Exact FLA causal_conv1d_fwd_kernel body, explicit launch metadata',layout_adapter='A5 logical weight[W,D] transposed to FLA contiguous weight[D,W] before timing; no activation/residual/state',launch=cfg,output_dtype=str(dt),accumulator_dtype='fp32; FP8 storage does not imply FP8 TensorCore arithmetic',flops=B*T*D*(2*W+1))
 elif op=='sdpa':
  B,H,S,D=shape;qc,Q=rand('q',(B,H,S,D));kc,K=rand('k',(B,H,S,D));vc,V=rand('v',(B,H,S,D));O=torch.empty((B,H,S,D),device=device,dtype=dt);LSE=torch.empty((B,H,S),device=device,dtype=torch.float32)
  scores=(qc.double()@kc.double().transpose(-1,-2))*(D**-.5)
  ref_lse=torch.logsumexp(scores,-1)/math.log(2);ref=torch.softmax(scores,-1)@vc.double();del scores
  ker=_jit(_module('attention')._attn_fwd);cfg={'GROUP_HEAD':1,'HEAD_DIM':D,'BLOCK_M':64,'BLOCK_N':32,'STAGE':1,'HAS_ATTN_MASK':False,'PRE_LOAD_V':False,'num_warps':4,'num_stages':2}
  # GROUP_HEAD and HEAD_DIM are passed as constexpr keyword metadata; Z,H,H,Q_CTX,KV_CTX remain positional.
  fn=lambda:ker[(triton.cdiv(S,64),B*H,1)](Q,K,V,None,D**-.5,LSE,O,*Q.stride(),*K.stride(),*V.stride(),1,1,1,1,*O.stride(),B,H,H,Q_CTX=S,KV_CTX=S,**cfg)
  outputs={'y':O,'lse_log2':LSE};refs={'y':ref,'lse_log2':ref_lse};useful=4*B*H*S*D*Q.element_size();extra=4*B*H*S;source='attention'
  if dtype=='fp32':rtol,atol=2e-5,2e-6
  if f8:rtol,atol=(.15,.025) if dtype=='fp8_e4m3fn' else (.30,.05)
  out_tols={'lse_log2':{'rtol':2e-5,'atol':2e-6}}
  contract.update(formula='O=softmax(Q@K.T/sqrt(D),lastdim)@V; dense noncausal no mask/no GQA/no dropout',entrypoint='flag_gems.ops.attention._attn_fwd JIT leaf with _attn_fwd_inner dependency',launch=cfg,output_dtype=str(dt),accumulator_dtype='fp32; qk and p-v tl.dot both explicitly allow_tf32=False',probability_rounding='p cast to value dtype before p@V, including float8 formats; precision contract includes that upstream rounding',additional_outputs='LSE_log2[B,H,S] FP32 checked independently with stricter FP32 threshold',fp8_tolerance_reason='FP8 quantizes both unnormalized probabilities before PV and output. Prespecified output gates include those two quantizations; LSE retains FP32 gate.',flops=4*B*H*S*S*D)
 elif op=='kv_cache_store':
  T,H,D,P=shape;blocks=triton.cdiv(T,P)+2;capacity=blocks*P;kc,K=rand('key',(T,H,D));vc,V=rand('value',(T,H,D));slots=torch.randperm(capacity,generator=gen,dtype=torch.int64)[:T].contiguous();Slots=meta('slot_mapping',slots)
  cpu_k=torch.zeros((blocks,P,H,D),dtype=dt);cpu_v=torch.zeros_like(cpu_k);KC=cpu_k.to(device);VC=cpu_v.to(device)
  refk=cpu_k.double();refv=cpu_v.double();refk.view(-1,H,D)[slots]=kc.double();refv.view(-1,H,D)[slots]=vc.double()
  ker=_jit(_module('reshape_and_cache_flash').reshape_and_cache_flash_kernel)
  fn=lambda:ker[(T,)](K,V,KC,VC,Slots,KC.stride(0),K.stride(0),V.stride(0),H,D,P,1.,1.,H*D,num_warps=4)
  outputs={'key_cache':KC,'value_cache':VC};refs={'key_cache':refk,'value_cache':refv};useful=4*T*H*D*K.element_size()+T*8;source='reshape_and_cache_flash';rtol=atol=0.
  contract.update(formula='Scatter K,V tokens to preassigned physical paged-cache slots; remaining slots unchanged',entrypoint='flag_gems.fused.reshape_and_cache_flash.reshape_and_cache_flash_kernel direct Triton leaf, avoids optional C extension',layout='B200 physical cache[blocks,page,heads,D], A5 [blocks,heads,page,D]; equivalent logical cache but physical layout differs',metadata='Unique shuffled slots from CPU seed; conversion of page metadata excluded from timed scope; metadata read bytes counted',state='Caches initialized once to zero outside timing; repeated idempotent writes of identical K/V; whole caches checked against reference',fp8_scope='Exact float8 storage copy only, no scale/quantization and no FP8 arithmetic claim. Source TODO refers to dtype conversion, not asserted as supported until actual direct-leaf smoke.',output_dtype=str(dt),flops=0)
 else:raise ValueError(op)
 return {'fn':fn,'outputs':outputs,'refs':refs,'cpu_inputs':cpu,'device_inputs':gpu,'useful_bytes':useful,'additional_bytes':extra,'shape':shape,'contract':contract,'sources':_sources(source),'rtol':rtol,'atol':atol,'output_tolerances':out_tols}
