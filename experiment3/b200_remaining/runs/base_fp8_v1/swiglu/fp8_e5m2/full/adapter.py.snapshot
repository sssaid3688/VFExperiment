"""Existing six forward kernels, extended with genuine FP8 storage probes."""
import hashlib,importlib,math
from pathlib import Path
SHAPES=[[m,n] for m in [1,2,4,8,16,32,64,128,256,512] for n in [64,128,256,512,768,1024,1536,2048,4096,8192]]
def shape_plan(op):return SHAPES
def build_case(op,dtype,index,device='cuda:0'):
 import torch
 M,N=SHAPES[index];dt={'fp32':torch.float32,'fp16':torch.float16,'fp8_e4m3fn':torch.float8_e4m3fn,'fp8_e5m2':torch.float8_e5m2}[dtype]
 gen=torch.Generator(device='cpu').manual_seed(20261020+index)
 cpu={'x':torch.randn(M,N,generator=gen,dtype=torch.float32).to(dt)}
 if op=='swiglu':cpu['b']=torch.randn(M,N,generator=gen,dtype=torch.float32).to(dt)
 if op in ['rms_norm','layer_norm']:
  cpu['weight']=torch.randn(N,generator=gen,dtype=torch.float32).to(dt)
  if op=='layer_norm':cpu['bias']=torch.randn(N,generator=gen,dtype=torch.float32).to(dt)
 inp={k:v.to(device) for k,v in cpu.items()};x=inp['x'];y=torch.empty_like(x);out={'y':y};xf=cpu['x'].double();refs={};tol={};extra=0
 path={'silu':'flag_gems.ops.silu','gelu_tanh':'flag_gems.ops.gelu','swiglu':'flag_gems.fused.silu_and_mul','rms_norm':'liger_kernel.ops.rms_norm','layer_norm':'liger_kernel.ops.layer_norm','softmax':'flag_gems.ops.softmax'}[op]
 mod=importlib.import_module(path)
 if op=='silu':fn=lambda:mod.silu_forward(x,out0=y);refs['y']=xf/(1+torch.exp(-xf));useful=2*M*N*x.element_size()
 elif op=='gelu_tanh':fn=lambda:mod.gelu_tanh(x,out0=y);refs['y']=.5*xf*(1+torch.tanh(math.sqrt(2/math.pi)*(xf+.044715*xf**3)));useful=2*M*N*x.element_size()
 elif op=='swiglu':fn=lambda:mod.silu_and_mul_kernel(x,inp['b'],out0=y);refs['y']=xf/(1+torch.exp(-xf))*cpu['b'].double();useful=3*M*N*x.element_size()
 elif op=='softmax':fn=lambda:mod.softmax_kernel_inner[(M,1,1)](y,x,M,N);refs['y']=torch.softmax(xf,-1);useful=2*M*N*x.element_size()
 else:
  utils=importlib.import_module('liger_kernel.ops.utils');block,warps=utils.calculate_settings(N)
  out['rstd']=torch.empty(M,device=device,dtype=torch.float32)
  if op=='rms_norm':
   refs['rstd']=torch.rsqrt(xf.square().mean(-1)+1e-6);refs['y']=xf*refs['rstd'][:,None]*cpu['weight'].double()
   fn=lambda:mod._rms_norm_forward_kernel[(M,)](y,N,x,N,inp['weight'],1,out['rstd'],1,N,1e-6,0.,1,elementwise_affine=True,BLOCK_SIZE=block,num_warps=warps)
   useful=(2*M*N+N)*x.element_size();extra=4*M
  else:
   out['mean']=torch.empty(M,device=device,dtype=torch.float32);refs['mean']=xf.mean(-1);cent=xf-refs['mean'][:,None];refs['rstd']=torch.rsqrt(cent.square().mean(-1)+1e-6);refs['y']=cent*refs['rstd'][:,None]*cpu['weight'].double()+cpu['bias'].double()
   fn=lambda:mod._layer_norm_forward_kernel[(M,)](y,N,x,N,inp['weight'],1,inp['bias'],1,out['mean'],1,out['rstd'],1,N,1e-6,BLOCK_SIZE=block,num_warps=warps)
   useful=(2*M*N+2*N)*x.element_size();extra=8*M
  for name in out:
   if name!='y':tol[name]={'rtol':2e-5,'atol':2e-6}
 rt,at={'fp32':(2e-5,2e-6),'fp16':(2e-3,2e-3),'fp8_e4m3fn':(.063,.00098),'fp8_e5m2':(.126,.000008)}[dtype]
 source=Path(mod.__file__);commit='43293d84d760e402e4fd974b12410a1d47c31761' if path.startswith('liger') else '30bfcc0735b8ebd48bd7339ef752ec64df69663f'
 url=('https://github.com/linkedin/Liger-Kernel/blob/'+commit+'/src/'+path.replace('.','/')+'.py') if path.startswith('liger') else ('https://github.com/flagos-ai/FlagGems/blob/'+commit+'/src/'+path.replace('.','/')+'.py')
 return dict(fn=fn,outputs=out,refs=refs,cpu_inputs=cpu,device_inputs=inp,useful_bytes=useful,additional_bytes=extra,shape=SHAPES[index],rtol=rt,atol=at,output_tolerances=tol,sources=[{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'upstream_commit':commit,'url':url}],contract={'operator':op,'input_generation':'CPU fp32 seed20261020+index then cast to genuine requested dtype','timed_scope':'one direct upstream Triton forward, preallocated output','fp8_policy':'FP8 storage, source vector arithmetic promoted as kernel defines, not FP8 TensorCore throughput; no host fp16 substitution','tolerance_policy':'FP8 output rounding bound: E4M3 half-relative-ulp6.25% plus half-subnormal-step2^-10; E5M2 12.5% plus2^-17; slight float32 arithmetic margin','norm_mode':'GEMMA offset0 eps1e-6; stats outputfp32 independently checked' if op in ['rms_norm','layer_norm'] else None})
