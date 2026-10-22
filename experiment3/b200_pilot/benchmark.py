#!/usr/bin/env python3
"""Read-only package baseline; all generated artifacts confined by launch env."""
import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/ubuntu/workspace/oyhj/papers2/experiment3').resolve()
PID = os.getpid()
MS = [1,2,4,8,16,32,64,128,256,512]
NS = [64,128,256,512,768,1024,1536,2048,4096,8192]
SHAPES = [(m,n) for m in MS for n in NS]

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def smi():
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,clocks.mem','--format=csv'],text=True)
    proc = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    pids = [int(line.split(',')[0].strip()) for line in proc.splitlines() if line.strip()]
    return {'timestamp_utc':stamp(),'gpu_csv':gpu,'process_csv':proc,'foreign_pids':[p for p in pids if p != PID]}

def emit(path,obj):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,allow_nan=False)+'\n')
        f.flush()

def quantile(xs,q):
    s=sorted(xs); pos=(len(s)-1)*q; lo=int(pos); hi=math.ceil(pos)
    return s[lo]+(s[hi]-s[lo])*(pos-lo)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','full'],required=True)
    args=ap.parse_args()
    folder=ROOT/'b200_pilot'/args.mode
    folder.mkdir(parents=True,exist_ok=False)
    initial=smi(); (folder/'initial_gpu.json').write_text(json.dumps(initial,indent=2))
    if initial['foreign_pids']:
        raise RuntimeError('GPU occupied before initialization: '+str(initial['foreign_pids']))
    import torch
    import triton
    import triton.testing
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    silu_mod=importlib.import_module('flag_gems.ops.silu')
    gelu_mod=importlib.import_module('flag_gems.ops.gelu')
    sources={name:{'path':mod.__file__,'sha256':sha(mod.__file__)} for name,mod in [('silu',silu_mod),('gelu_tanh',gelu_mod)]}
    sources['timer']={'path':triton.testing.__file__,'sha256':sha(triton.testing.__file__)}
    package_versions={n:importlib.metadata.version(n) for n in ['torch','triton','flag-gems']}
    manifest={'created_utc':stamp(),'pid':PID,'mode':args.mode,'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'versions':package_versions,'baseline':'FlagGems portable generic Triton kernels; not claimed B200 optimized','upstream_commit':None,'sources':sources,'harness_sha256':sha(__file__),'seed_base':20261020,'shapes':SHAPES,'case_order':'shape, dtype, operator; 5 repeated independent do_bench invocations per case','input_generation':'CPU torch.randn fp32 generator seed_base+zero-based shape index; cast on CPU to requested dtype, identical seed between dtypes/operators','reference':'CPU fp64 mathematical formula on actual rounded CPU inputs','timer':'triton.testing.do_bench(return_mode=all,warmup=25,rep=100)','groups':5 if args.mode=='full' else 0,'timed_scope':'direct upstream Triton pointwise entrypoint; preallocated out0; one kernel; allocations/validation excluded','cache_policy':'Triton do_bench native runtime-driver clear-cache before each timed iteration; cache outside event pair','fp8_status':'not requested for generic elementwise code: unverified/no matching supported precision contract; no fake cast path','environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','TRITON_CACHE_DIR','FLAGGEMS_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','CUDA_CACHE_PATH','PYTHONDONTWRITEBYTECODE']}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Input precision is scalar IEEE-like fp32 arithmetic, no tl.dot/TF32.
    indices=[0,99] if args.mode=='smoke' else list(range(100))
    output=folder/'cases.jsonl'
    failures=0;successful=0
    try:
        with torch.no_grad():
            for i in indices:
                M,N=SHAPES[i]
                guard=smi()
                if guard['foreign_pids']:
                    emit(folder/'interference.jsonl',guard)
                    raise RuntimeError('Foreign GPU process detected; final timings deferred')
                for dtype_name,dt in [('fp16',torch.float16),('fp32',torch.float32)]:
                    gen=torch.Generator(device='cpu').manual_seed(20261020+i)
                    cpu=torch.randn(M,N,generator=gen,dtype=torch.float32).to(dt)
                    cpu_hash=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
                    refx=cpu.to(torch.float64)
                    x=cpu.to('cuda:0');y=torch.empty_like(x)
                    rtol,atol=(2e-3,2e-3) if dtype_name=='fp16' else (2e-5,2e-6)
                    for op in ['silu','gelu_tanh']:
                        row={'case_id':f'{op}_{dtype_name}_{i:03d}','operator':op,'dtype':dtype_name,'shape_index':i,'shape':[M,N],'seed':20261020+i,'input_sha256':cpu_hash,'n_elements':M*N,'useful_bytes':2*M*N*cpu.element_size(),'input_dtype':str(dt),'output_dtype':str(dt),'accumulator_arithmetic':'fp32 scalar arithmetic','reference':'CPU fp64 on actual rounded input','atol':atol,'rtol':rtol,'pid':PID,'timestamp_utc':stamp(),'source':sources[op],'baseline':'portable generic Triton','upstream_commit':None,'groups':[]}
                        try:
                            fn=(lambda:silu_mod.silu_forward(x,out0=y)) if op=='silu' else (lambda:gelu_mod.gelu_tanh(x,out0=y))
                            ref=refx/(1+torch.exp(-refx)) if op=='silu' else 0.5*refx*(1+torch.tanh(math.sqrt(2/math.pi)*(refx+0.044715*refx.pow(3))))
                            fn();torch.cuda.synchronize()
                            got=y.cpu().to(torch.float64)
                            diff=(got-ref).abs(); limit=atol+rtol*ref.abs()
                            valid=bool(torch.isfinite(got).all() and (diff<=limit).all())
                            row.update({'max_abs_error':diff.max().item(),'rms_error':diff.square().mean().sqrt().item(),'max_tolerance_ratio':(diff/limit).max().item(),'correct':valid})
                            if not valid:
                                raise ValueError('Numerical tolerance failed')
                            if args.mode=='full':
                                for group in range(5):
                                    group_start=stamp()
                                    vals=triton.testing.do_bench(fn,warmup=25,rep=100,return_mode='all')
                                    vals=[float(v)*1000 for v in vals]
                                    group_info={'group':group,'start_utc':group_start,'samples_us':vals,'samples':len(vals),'median_us':statistics.median(vals),'q25_us':quantile(vals,.25),'q75_us':quantile(vals,.75),'iqr_us':quantile(vals,.75)-quantile(vals,.25)}
                                    row['groups'].append(group_info)
                                meds=[g['median_us'] for g in row['groups']]
                                row.update({'latency_us_median_of_group_medians':statistics.median(meds),'group_median_min_us':min(meds),'group_median_max_us':max(meds),'group_median_iqr_us':quantile(meds,.75)-quantile(meds,.25),'group_median_cv':statistics.stdev(meds)/statistics.mean(meds),'effective_GBps':row['useful_bytes']/statistics.median(meds)/1000})
                                # Verify final output and input integrity after repetitions.
                                finalgot=y.cpu().to(torch.float64)
                                if not bool(torch.isfinite(finalgot).all() and ((finalgot-ref).abs()<=limit).all()):
                                    raise ValueError('Post-timing numerical check failed')
                                if hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()!=cpu_hash:
                                    raise ValueError('Timed input was mutated')
                            row['status']='PASS';successful+=1
                        except Exception as e:
                            row.update({'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()});failures+=1
                        emit(output,row)
                        print(json.dumps({k:row.get(k) for k in ['case_id','status','max_abs_error','latency_us_median_of_group_medians','error']}),flush=True)
                    del cpu,refx,x,y
                emit(folder/'gpu_guards.jsonl',guard)
        final=smi();(folder/'final_gpu.json').write_text(json.dumps(final,indent=2))
        if final['foreign_pids']:
            raise RuntimeError('Foreign GPU process present at completion; timing needs review')
    finally:
        summary={'completed_utc':stamp(),'mode':args.mode,'successful':successful,'failed':failures,'expected':len(indices)*4,'pid':PID}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    if failures:
        raise SystemExit(2)

if __name__=='__main__':
    main()
