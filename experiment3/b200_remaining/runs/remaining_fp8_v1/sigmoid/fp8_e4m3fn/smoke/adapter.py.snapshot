"""Audited FlagGems v5.0.2 adapters. Import/build only in the serial GPU owner.

No GPU work is launched by this module until build_case is called by the runner.
Each returned fn only launches original Triton forward kernels into fixed buffers.
"""
import hashlib
import importlib
from functools import lru_cache
from pathlib import Path

COMMIT = '30bfcc0735b8ebd48bd7339ef752ec64df69663f'
BASE_URL = f'https://github.com/flagos-ai/FlagGems/blob/{COMMIT}/src/flag_gems/'
OPS = ['log_softmax','embedding','cumsum','groupnorm','sigmoid','sum_reduce','max_pool2d','relu']
MS = [1,2,4,8,16,32,64,128,256,512]
NS = [64,128,256,512,768,1024,1536,2048,4096,8192]
EXPECTED_SHA = {
 'log_softmax':'0dfb3b80530050b206e6a21f19142be957727575e1e3d3af13dd1e4c9751cb66',
 'embedding':'58ac369551bf30ba6bdc82145985a10914e5f033d67fa5ede06ac7e23b148e9a',
 'cumsum':'84cef7e5c350f9cc2ab73a8ab05cf3419684f3dced0f7c60b307da33f4960300',
 'groupnorm':'de1832c6d1540429d6138bb82e74f7fa69c4bb76a14f9a6da42f7f079f47bc1d',
 'sigmoid':'11a5e2e7777a5600fdedc135295171f6d38b9e9fada0fe6a4bca54615f8d864b',
 'sum':'3e4dc0dbf86349d1cc6d069f7a60ec793e7070d3e5107330acf04e9c5084583d',
 'max_pool2d_with_indices':'8d3ebd0e2bdfe3c4868590bc62c000de57162165f3d3b32245869b83d07a0491',
 'relu':'64a509380f6c7ba83d14eedb45caa1f1797e367c880e488e692d5679ea4aff0b',
 'limits':'6610b3ba16ade4be2544f51a208b9d6877ac1b0e6d3a065bcf9375afbdb18339',
}

def shape_plan(op):
    """Return fresh lists; no torch/Triton import or device initialization."""
    if op not in OPS: raise ValueError(op)
    if op == 'embedding':
        return [[t,4096,h] for t in MS for h in NS]
    if op == 'groupnorm':
        return [[b,c,h,h] for b in [1,2,4,8,16] for c in [8,16,32,64] for h in [4,8,16,24,32]]
    if op == 'max_pool2d':
        return [[b,c,h,h] for b in [1,2,4,8,16] for c in [1,4,8,16] for h in [8,16,24,32,64]]
    if op == 'cumsum':
        return [[m,n] for m in MS for n in [32,64,128,256,512,1024,2048,4096,8192,16384]]
    return [[m,n] for m in MS for n in NS]

@lru_cache(None)
def _module(name):
    mod = importlib.import_module('flag_gems.ops.' + name)
    digest = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    if digest != EXPECTED_SHA[name]:
        raise RuntimeError(f'SOURCE_MISMATCH {name}: {digest}, expected {EXPECTED_SHA[name]}')
    return mod

def _source(name):
    mod = _module(name)
    return {'path':mod.__file__,'sha256':EXPECTED_SHA[name], 'upstream_commit':COMMIT,
            'url':BASE_URL+'ops/'+name+'.py','license':'Apache-2.0',
            'classification':'official generic Triton; B200-specific tuning not established'}

def _dtype(dtype):
    import torch
    if isinstance(dtype, torch.dtype): dt=dtype
    else:
        key=str(dtype).replace('torch.','')
        table={'fp16':torch.float16,'float16':torch.float16,'fp32':torch.float32,'float32':torch.float32,
               'fp8_e4m3fn':torch.float8_e4m3fn,'float8_e4m3fn':torch.float8_e4m3fn,
               'fp8_e5m2':torch.float8_e5m2,'float8_e5m2':torch.float8_e5m2}
        if key not in table: raise NotImplementedError('UNSUPPORTED dtype '+key)
        dt=table[key]
    if dt not in [torch.float16,torch.float32,torch.float8_e4m3fn,torch.float8_e5m2]:
        raise NotImplementedError('UNSUPPORTED dtype '+str(dt))
    return dt

def tensor_bytes(t):
    """CPU or device tensor hash bytes; float8 has no direct NumPy dtype."""
    import torch
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

def _is_fp8(dt):
    import torch
    return dt in [torch.float8_e4m3fn,torch.float8_e5m2]

def _ref(value,dt):
    """FP8 compares to declared final quantized storage output, never fp16."""
    import torch
    if _is_fp8(dt): return value.to(dt).to(torch.float64)
    return value.to(torch.float64)

def _tolerances(dt,op):
    import torch
    if op in ['embedding','relu','max_pool2d']: return 0.0,0.0
    if dt == torch.float8_e4m3fn: return 0.125,2.0**-9
    if dt == torch.float8_e5m2: return 0.25,2.0**-16
    if dt == torch.float16: return 2e-3,2e-3
    return 2e-5,(2e-5 if op in ['cumsum','sum_reduce'] else 2e-6)

def _common(op,dt,index,shape,cpu,dev,outputs,refs,fn,useful,extra,contract,sources,math_refs=None,output_tolerances=None):
    rtol,atol=_tolerances(dt,op)
    contract.update({'seed':20261020+index,'input_generation':'CPU torch.randn fp32 with one CPU generator, then actual input dtype cast on CPU; integer indices use same generator',
       'baseline':'official generic Triton; no B200-specific optimization claimed',
       'precision_regime':'FP8 storage, vector/scalar compute; no FP8 TensorCore claim' if _is_fp8(dt) else str(dt),
       'reference':'independent CPU fp64 formula on actual rounded inputs',
       'reference_output_cast':'FP8: quantize final mathematical result directly to actual output FP8 dtype then convert to fp64. FP16/FP32: unrounded CPU fp64 result.',
       'math_refs':'Unquantized CPU fp64 mathematical results on actual rounded input values; log separately from storage-quantized refs',
       'fp8_tolerance':'E4M3FN rtol=1/8 atol=2^-9; E5M2 rtol=1/4 atol=2^-16, one-output-ULP envelope against quantized reference; exact for gather/ReLU/max selection',
       'fp8_qualification':'candidate only; serial runner must compile and pass two-shape gate before support is claimed',
       'timed_scope':'preallocated outputs; only original Triton kernel launches; no native Torch compute, copies, allocation, or backward inside fn',
       'mutation':'read-only inputs; output overwritten each call', 'kernel_launches':1})
    return {'fn':fn,'outputs':outputs,'refs':refs,'math_refs':refs if math_refs is None else math_refs,
      'output_tolerances':{} if output_tolerances is None else output_tolerances,'cpu_inputs':cpu,'device_inputs':dev,
      'useful_bytes':int(useful),'additional_bytes':int(extra),'shape':list(shape),
      'contract':contract,'sources':sources,'rtol':rtol,'atol':atol}

def _pool_case(dt,index,device,probe=None):
    import torch
    import torch.nn.functional as F
    import triton
    shape=shape_plan('max_pool2d')[index];B,C,H,W=shape
    gen=torch.Generator(device='cpu').manual_seed(20261020+index)
    raw=torch.randn(shape,generator=gen,dtype=torch.float32)
    raw[:,:,:2,:2]=7.0  # every case has an explicit max tie; first scan index must win
    if probe=='nan':raw[:,:,0,1]=float('nan')
    elif probe=='all_negative_infinity':raw[:,:,:2,:2]=float('-inf')
    elif probe is not None:raise ValueError(probe)
    cpu={'x':raw.to(dt)};dev={'x':cpu['x'].to(device)}
    OH,OW=H//2,W//2
    out=torch.empty((B,C,OH,OW),device=device,dtype=dt)
    inds=torch.empty((B,C,OH,OW),device=device,dtype=torch.int64)
    ref,refidx=F.max_pool2d(cpu['x'].double(),2,stride=2,return_indices=True)
    mod=_module('max_pool2d_with_indices')
    x=dev['x'];strides=x.stride()
    grid=lambda meta:(B*C,triton.cdiv(OH,meta['BLOCK_H'])*triton.cdiv(OW,meta['BLOCK_W']))
    def fn():
        mod.max_pool2d_forward_kernel[grid](x,out,inds,*strides,C,H,W,OH,OW,2,2,2,2,0,0,1,1)
    limit_path=Path(mod.__file__).parents[1]/'utils'/'limits.py'
    actual=hashlib.sha256(limit_path.read_bytes()).hexdigest()
    if actual!=EXPECTED_SHA['limits']:raise RuntimeError('SOURCE_MISMATCH limits.py')
    source_limits={'path':str(limit_path),'sha256':actual,'upstream_commit':COMMIT,'url':BASE_URL+'utils/limits.py','license':'Apache-2.0'}
    contract={'formula':'NCHW max_pool2d kernel=2,stride=2,padding=0,dilation=1,ceil_mode=False,with int64 flattened spatial indices',
       'finite_input_contract':probe is None,'ties':'strict > chooses first row-major kh,kw occurrence; explicit 2x2 tie injected in every batch/channel',
       'integer_ref_outputs':['indices'],'nan_semantics':'UNSUPPORTED relative to PyTorch: kernel strict > never propagates NaN; independent nan probe expected to expose mismatch',
       'infinity_semantics':'all -inf windows leave initial index=-1; independent all_negative_infinity probe expected to expose mismatch',
       'semantic_probe':probe,'reference_indices':'CPU F.max_pool2d float64, exact int64; per-channel flattened H*W',
       'fp8_source_caveat':'Kernel initializes max accumulator to -inf in input dtype; E4M3FN has no infinity, so it may compile-fail or fail numerical gate. E5M2 must be tested separately.',
       'autotune':'original installed 12-config Triton autotuner; tuning outside timed groups after first validation launch'}
    return _common('max_pool2d',dt,index,shape,cpu,dev,{'output':out,'indices':inds},{'output':ref,'indices':refidx},fn,
       cpu['x'].numel()*cpu['x'].element_size()+out.numel()*(out.element_size()+8),0,contract,[_source('max_pool2d_with_indices'),source_limits])

def build_max_pool2d_probe(dtype,probe,device='cuda:0'):
    """Optional diagnostic only: 'nan' or 'all_negative_infinity'; do not time."""
    return _pool_case(_dtype(dtype),0,device,probe=probe)

def build_case(op,dtype,index,device='cuda:0'):
    import torch
    import triton
    if op not in OPS:raise ValueError(op)
    if not 0<=index<100:raise ValueError('index must be 0..99')
    dt=_dtype(dtype)
    if op=='max_pool2d':return _pool_case(dt,index,device)
    shape=shape_plan(op)[index]
    gen=torch.Generator(device='cpu').manual_seed(20261020+index)
    module_name='sum' if op=='sum_reduce' else op
    mod=_module(module_name)
    sources=[_source(module_name)]
    contract={};additional=0;output_tolerances={};math_refs={}
    if op=='embedding':
        T,V,H=shape
        cpu={'weight':torch.randn(V,H,generator=gen,dtype=torch.float32).to(dt),
             'indices':torch.randint(0,V,(T,),generator=gen,dtype=torch.int64)}
        dev={k:v.to(device) for k,v in cpu.items()}
        output=torch.empty((T,H),device=device,dtype=dt)
        ref=cpu['weight'].double()[cpu['indices']]
        def fn():mod.embedding_kernel[(T,)](output,dev['indices'],dev['weight'],H,triton.next_power_of_2(H))
        outputs={'output':output};refs={'output':ref}
        math_refs={'output':ref}
        useful=8*T+2*T*H*cpu['weight'].element_size()
        contract.update(formula='table[indices], no backward, padding_idx unused in forward',layout='weight[V,H], indices[T], output[T,H]',
          traffic_model='minimum useful gather read+write plus int64 indices, not full table bytes',fp8_source_caveat='direct load/store without arithmetic; true FP8 input/output eligible for gate')
    elif op=='groupnorm':
        B,C,H,W=shape;G=C//8;GS=8;HW=H*W
        cpu={'x':torch.randn(shape,generator=gen,dtype=torch.float32).to(dt),
             'weight':torch.randn(C,generator=gen,dtype=torch.float32).to(dt),
             'bias':torch.randn(C,generator=gen,dtype=torch.float32).to(dt)}
        dev={k:v.to(device) for k,v in cpu.items()}
        output=torch.empty_like(dev['x']);mean=torch.empty((B,G),device=device,dtype=dt);rstd=torch.empty_like(mean)
        xf=cpu['x'].double().reshape(B,G,GS,H,W)
        rm=xf.mean((2,3,4));center=xf-rm[:,:,None,None,None]
        rs=torch.rsqrt(center.square().mean((2,3,4))+1e-5)
        ry=(center*rs[:,:,None,None,None]).reshape(shape)*cpu['weight'].double()[None,:,None,None]+cpu['bias'].double()[None,:,None,None]
        def fn():
            mod.group_norm_kernel[(B*G,)](dev['x'],output,dev['weight'],dev['bias'],mean,rstd,GS,C,HW,G,1e-5,
              BLOCK_GROUP_SIZE=8,BLOCK_HW_SIZE=triton.next_power_of_2(HW))
        outputs={'output':output,'mean':mean,'rstd':rstd};refs={'output':_ref(ry,dt),'mean':_ref(rm,dt),'rstd':_ref(rs,dt)}
        math_refs={'output':ry,'mean':rm,'rstd':rs}
        useful=(2*cpu['x'].numel()+2*C)*cpu['x'].element_size();additional=2*B*G*cpu['x'].element_size()
        contract.update(formula='affine groupnorm biased population variance, eps1e-5',groups=G,group_size=GS,
          accumulation='source explicitly casts X to fp32; weight/bias mixed operations promote with fp32 normalized X',
          additional_outputs='Mean[B,G],Rstd[B,G] in actual input dtype, exactly as upstream wrapper',
          shape_constraint='C divisible by G, fixed C/G=8 power-of-two; block8*nextpow2(HW)<=8192',
          fp8_source_caveat='X load explicitly fp32; true FP8 input/output/stat storage must pass gate')
    else:
        M,N=shape
        raw=torch.randn(M,N,generator=gen,dtype=torch.float32)
        if op in ['cumsum','sum_reduce']:
            raw=raw*0.125  # prospective finite-range inputs for FP8 prefixes/reductions
            contract['input_distribution']='CPU fp32 normal(0,1)*0.125 before CPU dtype cast, for all precisions'
        cpu={'x':raw.to(dt)}
        dev={'x':cpu['x'].to(device)};x=dev['x'];xf=cpu['x'].double()
        output=torch.empty((M,1) if op=='sum_reduce' else (M,N),device=device,dtype=dt)
        outputs={'output':output}
        if op=='log_softmax':
            ref=xf-torch.logsumexp(xf,-1,keepdim=True)
            def fn():mod.log_softmax_kernel[(triton.cdiv(M,8),1)](output,x,M,N,1,BLOCK_M=8,BLOCK_N=256,num_warps=8)
            contract.update(formula='log_softmax(x,dim=-1), same input/output storage dtype',accumulation='upstream explicitly fp32 online logsumexp; original M8,N256 launch',
              fp8_source_caveat='explicit fp32 load and true FP8 store, no harness input cast')
        elif op=='sigmoid':
            ref=torch.sigmoid(xf)
            def fn():mod.sigmoid_forward(x,out0=output)
            contract.update(formula='sigmoid(x)',accumulation='source converts input to fp32 and uses exp2',fp8_source_caveat='pointwise INT_TO_FLOAT promotion; FP8 remains floating input and explicit out0 is FP8')
        elif op=='relu':
            ref=torch.where(xf>0,xf,0.)
            def fn():mod.relu_forward(x,out0=output)
            contract.update(formula='where(x>0,x,0), finite inputs',fp8_source_caveat='FP8 load/compare/select/store; no native TensorCore claim')
        elif op=='sum_reduce':
            ref=xf.sum(dim=-1,keepdim=True)
            def fn():mod.sum_dim_kernel_inner[(M,1,1)](output,x,M,N)
            contract.update(formula='sum(x,dim=-1,keepdim=True)',dim=-1,keepdim=True,
              accumulation='upstream explicitly promotes fp16/bf16 to fp32; other input dtypes use original dtype before reduction',
              fp8_source_caveat='FP8 is not explicitly promoted by source cdtype dispatch; compile/numeric gate required; unsupported must be recorded rather than emulating through fp16',
              tolerance_reason='FP32 atol2e-5 permits ordinary FP32 reduction error near zero against CPU fp64; no reference change')
        elif op=='cumsum':
            ref=xf.cumsum(dim=-1)
            if _is_fp8(dt):
                # Original generic scan kernel with an explicit fp32 load path.
                # Single block per row: same mathematical inclusive prefix, no fp16 staging.
                partial=torch.empty((M,1,1),device=device,dtype=torch.float32)
                def fn():mod.scan_part_sum_abc_kernel[(M,1,1)](x,output,partial,N,1,1,triton.next_power_of_2(N))
                outputs['partial_sum']=partial;ref_sum=xf.sum(-1).reshape(M,1,1)
                refs={'output':_ref(ref,dt),'partial_sum':ref_sum};additional=4*M
                math_refs={'output':ref,'partial_sum':ref_sum}
                output_tolerances['partial_sum']={'rtol':2e-5,'atol':2e-5}
                contract.update(entrypoint='scan_part_sum_abc_kernel, one full-row block, part_num1, explicit fp32 internal scan',
                  additional_outputs='partial_sum[M,1,1] fp32, unused helper output; one kernel total',
                  fp8_source_caveat='alternate existing upstream scan kernel explicitly converts any floating input to fp32; output remains real FP8')
            else:
                def fn():mod.reduce_then_scan_root_scan_kernel_row[(M,1,1)](x,output,N,triton.next_power_of_2(N),num_warps=8 if N>2048 else 4)
                contract.update(entrypoint='reduce_then_scan_root_scan_kernel_row, original N<=16384 row path',accumulation='fp16 promoted to fp32; fp32 unchanged')
            contract.update(formula='inclusive cumsum(x,dim=-1), same input/output dtype',dim=-1,
              shape_restriction='power-of-two scan widths 32..16384; original FP8 helper does not define masked partial-sum padding, so no partial padding is used',
              tolerance_reason='FP32 atol2e-5 for prefix cancellation versus CPU fp64')
        if op!='cumsum' or not _is_fp8(dt):
            refs={'output':_ref(ref,dt)}
            math_refs={'output':ref}
        useful=(M*N+output.numel())*cpu['x'].element_size()
    return _common(op,dt,index,shape,cpu,dev,outputs,refs,fn,useful,additional,contract,sources,math_refs,output_tolerances)

def audit_shape_plans():
    for op in OPS:
        plan=shape_plan(op)
        assert len(plan)==100 and len({tuple(p) for p in plan})==100,op
    return {op:{'cases':100,'first':shape_plan(op)[0],'last':shape_plan(op)[99]} for op in OPS}

if __name__=='__main__':
    import json
    print(json.dumps(audit_shape_plans(),indent=2))
