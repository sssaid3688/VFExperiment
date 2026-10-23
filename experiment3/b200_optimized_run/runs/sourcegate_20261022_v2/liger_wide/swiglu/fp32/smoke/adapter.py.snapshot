"""UNRUN FP32-only Liger native Blackwell/row SwiGLU matched-cohort candidate.

Frozen before timing. Import/shape_plan require only stdlib and never touch GPU.
Root's serial runner alone may call build_case. No old shape or result is changed.
"""
import hashlib,importlib,os
from pathlib import Path

COMMIT='43293d84d760e402e4fd974b12410a1d47c31761'
MS=[1,2,4,8,16,32,64,128,256,512]
NS=[64,128,256,512,1024,4096,8192,14336,16384,28672]
SHAPES=[[m,n] for m in MS for n in NS]
EXPECTED={
 'liger_kernel.ops.swiglu':('ops/swiglu.py','7397029e8fefac0de42e5eeafcbbac147644a91c357bd30b9ba019fa5df3ce0a'),
 'liger_kernel.ops.utils':('ops/utils.py','9c566fbe7444d434daf86f12a99c2f4a526b7b7ee77f622849eb865e741cbe2c'),
 'liger_kernel.utils':('utils.py','3757f7531ac7868db662b41f471cb21f107ea493317aacca2041d3b0954e8b25')}

def shape_plan(op):
 if op not in ('swiglu','swiglu_blackwell_wide'):raise ValueError(op)
 return [s.copy() for s in SHAPES]

def _check_caches():
 allowed=Path('/home/ubuntu/workspace/oyhj/papers2').resolve()
 for key in ('TRITON_CACHE_DIR','FLAGGEMS_CACHE_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR'):
  value=os.environ.get(key)
  if not value or not Path(value).resolve().is_relative_to(allowed):raise RuntimeError('Set restricted cache before imports: '+key)
 if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('PYTHONDONTWRITEBYTECODE=1 required')

def build_case(op,dtype,index,device='cuda:0'):
 if op not in ('swiglu','swiglu_blackwell_wide'):raise ValueError(op)
 if dtype not in ('fp32','float32','torch.float32'):
  raise NotImplementedError('PRECISION_CONTRACT_MISMATCH: Liger casts SiLU to b.dtype before multiply; FP16/FP8 are excluded from the all-FP32-intermediate matched cohort, not replaced or emulated.')
 if not 0<=index<100:raise IndexError(index)
 _check_caches()
 import torch,triton
 mod=importlib.import_module('liger_kernel.ops.swiglu');sources=[]
 for name,(relative,expected) in EXPECTED.items():
  srcmod=importlib.import_module(name);p=Path(srcmod.__file__);actual=hashlib.sha256(p.read_bytes()).hexdigest()
  if actual!=expected:raise RuntimeError('PINNED_SOURCE_MISMATCH '+name+': '+actual)
  sources.append(dict(path=str(p),sha256=actual,upstream_commit=COMMIT,
   url=f'https://github.com/linkedin/Liger-Kernel/blob/{COMMIT}/src/liger_kernel/{relative}',license='BSD-2-Clause',
   role='Exact official native Blackwell dispatcher/kernel/settings source, not a modified mathematical kernel'))
 M,N=SHAPES[index];gen=torch.Generator(device='cpu').manual_seed(20261020+index)
 cpu={'x':torch.randn((M,N),generator=gen,dtype=torch.float32),
      'b':torch.randn((M,N),generator=gen,dtype=torch.float32)}
 inputs={name:value.to(device) for name,value in cpu.items()}
 A,B=inputs['x'],inputs['b'];C=torch.empty((M,N),dtype=torch.float32,device=device)
 # This is the exact upstream dispatcher condition and corresponding settings.
 should_tile=bool(mod._should_tile(N));arch=mod.infer_device_arch()
 if should_tile:
  block,warps=mod._swiglu_tile_settings(N);grid=(M,triton.cdiv(N,block))
  kernel=mod._swiglu_forward_kernel_tiled;kernel_name='_swiglu_forward_kernel_tiled'
 else:
  block,warps=mod.calculate_settings(N);grid=(M,)
  kernel=mod._swiglu_forward_kernel;kernel_name='_swiglu_forward_kernel'
 def fn():kernel[grid](A,B,C,A.stride(0),1.0,N,BLOCK_SIZE=block,num_warps=warps)
 x64=cpu['x'].double();b64=cpu['b'].double();ref=x64/(1+torch.exp(-x64))*b64
 return dict(fn=fn,outputs={'y':C},refs={'y':ref},math_refs={'y':ref},cpu_inputs=cpu,device_inputs=inputs,
  useful_bytes=3*M*N*4,additional_bytes=0,shape=[M,N],rtol=2e-5,atol=2e-6,sources=sources,
  contract=dict(operator=op,cohort='prespecified_blackwell_swiglu_fp32_native_20261022',
   formula='y=silu(x)*b; FP32 SiLU and multiply, final FP32 store',
   input_generation='CPU fp32 torch.Generator seed20261020+case_index, x[M,N] then b[M,N]; actual rounded CPU inputs used by independent CPUfp64 reference',
   frozen_plan=dict(M=MS,N=NS,unique_shapes=100,selection='Source-driven FFN widths and native gate below/above transition; frozen before timing; old100shape grid unchanged'),
   paired_comparison='A5 must use this exact new shape plan/seed and FP32 contract. Do not compare different old/new shapes by matching case_index.',
   native_should_tile=should_tile,detected_device_arch=arch,
   native_gate='infer_device_arch().startswith("blackwell") and next_power_of_2(N)>=16384',
   native_kernel_path='liger_kernel.ops.swiglu.'+kernel_name,
   native_tuning='No autotuner on these original JIT kernels. Preserve native dispatcher and calculate_settings/_swiglu_tile_settings; do not invent an autotune/B200 claim for row path.',
   native_gate_multiplier=1.0,launch=dict(grid=list(grid),BLOCK_SIZE=block,num_warps=warps),
   baseline='Exact pinned Liger native Blackwell tiled path only on eligible shapes; native row path otherwise; performance untested',
   targeted_Blackwell_branch_selected=should_tile,
   precision='All inputs/output FP32; source silu.cast(b.dtype) is an identity FP32 cast; FP16/FP8 explicitly excluded because their intermediate rounding changes the contract',
   timed_scope='One original Triton forward kernel into preallocated C. Input generation, CPU references, allocation, descriptor/shape decisions and device transfer outside fn; no native Torch GPU compute/backward',
   mutation='Inputs read-only; output overwritten each invocation',candidate_status='UNRUN: no compile, correctness, timing or optimality claims'))
