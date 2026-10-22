#!/usr/bin/env python3
"""Direct official Triton forward kernels, preallocated output/statistics."""
import argparse,datetime,hashlib,importlib,importlib.metadata,json,math,os,statistics,subprocess,traceback
from pathlib import Path

ROOT=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3').resolve()
PID=os.getpid()
SHAPES=[(m,n) for m in [1,2,4,8,16,32,64,128,256,512] for n in [64,128,256,512,768,1024,1536,2048,4096,8192]]
EPS=1e-6
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,r):
    with p.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
def smi():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,clocks.mem','--format=csv'],text=True)
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return {'timestamp_utc':stamp(),'gpu_csv':gpu,'process_csv':proc,'foreign_pids':[int(s.split(',')[0]) for s in proc.splitlines() if s.strip() and int(s.split(',')[0])!=PID]}
def quantile(xs,q):
    xs=sorted(xs);p=(len(xs)-1)*q;i=int(p);j=math.ceil(p);return xs[i]+(xs[j]-xs[i])*(p-i)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','full'],required=True);args=ap.parse_args()
    folder=ROOT/'b200_norms'/args.mode;folder.mkdir(parents=True,exist_ok=False)
    initial=smi();(folder/'initial_gpu.json').write_text(json.dumps(initial,indent=2))
    if initial['foreign_pids']:raise RuntimeError('GPU occupied: '+str(initial['foreign_pids']))
    import torch,triton,triton.testing
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sw=importlib.import_module('flag_gems.fused.silu_and_mul')
    rms=importlib.import_module('liger_kernel.ops.rms_norm')
    ln=importlib.import_module('liger_kernel.ops.layer_norm')
    utils=importlib.import_module('liger_kernel.ops.utils')
    modules={'swiglu':sw,'rms_norm':rms,'layer_norm':ln,'liger_utils':utils,'timer':triton.testing}
    sources={n:{'path':m.__file__,'sha256':sha(m.__file__)} for n,m in modules.items()}
    contracts={
      'swiglu':{'formula':'a/(1+exp(-a))*b','intermediate':'fp32 through SiLU and final multiply; cast only at output','entrypoint':'flag_gems.fused.silu_and_mul.silu_and_mul_kernel(a,b,out0=y)','additional_outputs':[],'baseline':'FlagGems generic Triton; not B200 specialized','upstream_commit':'30bfcc0735b8ebd48bd7339ef752ec64df69663f'},
      'rms_norm':{'formula':'x*rsqrt(mean(x*x,lastdim)+1e-6)*w','intermediate':'GEMMA casting_mode=1, offset=0; X,W,eps,offset and all arithmetic fp32; cast only output','entrypoint':'Liger _rms_norm_forward_kernel','additional_outputs':['RSTD[M] fp32'],'baseline':'Liger generic row Triton forward; not B200 specialized','upstream_commit':'43293d84d760e402e4fd974b12410a1d47c31761'},
      'layer_norm':{'formula':'(x-mean(x))*rsqrt(mean((x-mean(x))^2)+1e-6)*w+b','intermediate':'X,W,b and all arithmetic fp32; cast only output','entrypoint':'Liger _layer_norm_forward_kernel','additional_outputs':['Mean[M] fp32','RSTD[M] fp32'],'baseline':'Liger generic row Triton forward; not B200 specialized','upstream_commit':'43293d84d760e402e4fd974b12410a1d47c31761'}}
    manifest={'created_utc':stamp(),'mode':args.mode,'pid':PID,'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'versions':{n:importlib.metadata.version(n) for n in ['torch','triton','flag-gems','liger-kernel']},'sources':sources,'contracts':contracts,'harness_sha256':sha(__file__),'shapes':SHAPES,'seed_base':20261020,'input_generation':'Per operator reset CPU fp32 torch.Generator seed20261020+shape_index; x[M,N] first. swiglu then b[M,N]; RMS then weight[N]; LN then weight[N],bias[N]. Generate each source in fp32 then cast CPU to dtype.','reference':'Independent CPU fp64 formulas on actual rounded inputs, not altered to reproduce low precision','timer':'triton.testing.do_bench(warmup=25,rep=100,return_mode=all), 5 invocations','timed_scope':'direct single Triton forward kernel, preallocated output/statistics; no Torch compute or backward inside callable','cache_policy':'native do_bench driver cache-clear outside event pair','fp8_status':'no verified matching supported implementation; not run and no fake cast','environment':{k:os.environ.get(k) for k in ['TRITON_CACHE_DIR','FLAGGEMS_CACHE_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR','PYTHONDONTWRITEBYTECODE']}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    indices=[0,99] if args.mode=='smoke' else list(range(100))
    successful=failed=0;terminal_status='RUNNING'
    try:
      with torch.no_grad():
       for i in indices:
        M,N=SHAPES[i];guard=smi();emit(folder/'gpu_guards.jsonl',guard)
        if guard['foreign_pids']:raise RuntimeError('Foreign GPU process; defer measurement')
        for dtype_name,dt in [('fp16',torch.float16),('fp32',torch.float32)]:
         for op in ['swiglu','rms_norm','layer_norm']:
          gen=torch.Generator(device='cpu').manual_seed(20261020+i)
          cpu={'x':torch.randn(M,N,generator=gen,dtype=torch.float32).to(dt)}
          if op=='swiglu':cpu['b']=torch.randn(M,N,generator=gen,dtype=torch.float32).to(dt)
          else:
           cpu['weight']=torch.randn(N,generator=gen,dtype=torch.float32).to(dt)
           if op=='layer_norm':cpu['bias']=torch.randn(N,generator=gen,dtype=torch.float32).to(dt)
          inp={k:v.to('cuda:0') for k,v in cpu.items()};x=inp['x'];y=torch.empty_like(x)
          input_hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in cpu.items()}
          rtol,atol=(2e-3,2e-3) if dtype_name=='fp16' else (2e-5,2e-6)
          block,warps=utils.calculate_settings(N)
          stats={};refs={};xf=cpu['x'].double()
          if op=='swiglu':
           fn=lambda:sw.silu_and_mul_kernel(x,inp['b'],out0=y)
           ref=xf/(1+torch.exp(-xf))*cpu['b'].double()
           useful_bytes=3*M*N*cpu['x'].element_size();extra_bytes=0
           launch={'kind':'FlagGems pointwise_dynamic; exact generated metadata in compiler archive'}
          elif op=='rms_norm':
           stats['rstd']=torch.empty(M,device='cuda:0',dtype=torch.float32)
           refs['rstd']=torch.rsqrt(xf.square().mean(-1)+EPS)
           ref=xf*refs['rstd'].unsqueeze(-1)*cpu['weight'].double()
           fn=lambda:rms._rms_norm_forward_kernel[(M,)](y,N,x,N,inp['weight'],1,stats['rstd'],1,N,EPS,0.0,1,elementwise_affine=True,BLOCK_SIZE=block,num_warps=warps)
           useful_bytes=(2*M*N+N)*cpu['x'].element_size();extra_bytes=4*M
           launch={'grid':[M],'BLOCK_SIZE':block,'num_warps':warps,'casting_mode':1,'casting_mode_name':'gemma','offset':0,'elementwise_affine':True}
          else:
           stats['mean']=torch.empty(M,device='cuda:0',dtype=torch.float32);stats['rstd']=torch.empty_like(stats['mean'])
           refs['mean']=xf.mean(-1);cent=xf-refs['mean'].unsqueeze(-1)
           refs['rstd']=torch.rsqrt(cent.square().mean(-1)+EPS)
           ref=cent*refs['rstd'].unsqueeze(-1)*cpu['weight'].double()+cpu['bias'].double()
           fn=lambda:ln._layer_norm_forward_kernel[(M,)](y,N,x,N,inp['weight'],1,inp['bias'],1,stats['mean'],1,stats['rstd'],1,N,EPS,BLOCK_SIZE=block,num_warps=warps)
           useful_bytes=(2*M*N+2*N)*cpu['x'].element_size();extra_bytes=8*M
           launch={'grid':[M],'BLOCK_SIZE':block,'num_warps':warps}
          row={'case_id':f'{op}_{dtype_name}_{i:03d}','operator':op,'dtype':dtype_name,'shape_index':i,'shape':[M,N],'seed':20261020+i,'input_sha256':input_hashes['x'],'input_hashes':input_hashes,'n_elements':M*N,'useful_bytes':useful_bytes,'additional_stat_write_bytes':extra_bytes,'kernel_minimum_bytes':useful_bytes+extra_bytes,'output_dtype':str(dt),'atol':atol,'rtol':rtol,'source':sources[op],'contract':contracts[op],'launch':launch,'timestamp_utc':stamp(),'pid':PID,'groups':[]}
          try:
           fn();torch.cuda.synchronize();got=y.cpu().double();diff=(got-ref).abs();limit=atol+rtol*ref.abs()
           correct=bool(torch.isfinite(got).all() and (diff<=limit).all())
           checks={}
           for k,v in stats.items():
            sg=v.cpu().double();err=(sg-refs[k]).abs();lim=2e-6+2e-5*refs[k].abs()
            checks[k]={'dtype':'fp32','max_abs_error':err.max().item(),'max_tolerance_ratio':(err/lim).max().item(),'correct':bool(torch.isfinite(sg).all() and (err<=lim).all())}
           correct=correct and all(z['correct'] for z in checks.values())
           row.update({'correct':correct,'max_abs_error':diff.max().item(),'rms_error':diff.square().mean().sqrt().item(),'max_tolerance_ratio':(diff/limit).max().item(),'statistic_correctness':checks})
           if not correct:raise ValueError('Numeric tolerance failure')
           if args.mode=='full':
            for g in range(5):
             began=stamp();vals=[float(v)*1000 for v in triton.testing.do_bench(fn,warmup=25,rep=100,return_mode='all')]
             row['groups'].append({'group':g,'start_utc':began,'samples_us':vals,'samples':len(vals),'median_us':statistics.median(vals),'q25_us':quantile(vals,.25),'q75_us':quantile(vals,.75),'iqr_us':quantile(vals,.75)-quantile(vals,.25)})
            meds=[g['median_us'] for g in row['groups']]
            row.update({'latency_us_median_of_group_medians':statistics.median(meds),'group_median_cv':statistics.stdev(meds)/statistics.mean(meds),'group_median_min_us':min(meds),'group_median_max_us':max(meds),'group_median_iqr_us':quantile(meds,.75)-quantile(meds,.25),'effective_GBps':useful_bytes/statistics.median(meds)/1000})
            if not bool(torch.isfinite(y).all() and ((y.cpu().double()-ref).abs()<=limit).all()):raise ValueError('Post-timing numeric failure')
            for k,t in inp.items():
             if hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()!=input_hashes[k]:raise ValueError('Input mutated: '+k)
           row['status']='PASS';successful+=1
          except Exception as exc:
           row.update({'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc()});failed+=1
          emit(folder/'cases.jsonl',row)
          print(json.dumps({k:row.get(k) for k in ['case_id','status','max_abs_error','latency_us_median_of_group_medians','error']}),flush=True)
          del inp,x,y,cpu,stats,refs,ref,xf,fn
      final=smi();(folder/'final_gpu.json').write_text(json.dumps(final,indent=2))
      if final['foreign_pids']:raise RuntimeError('Foreign GPU process at completion')
      terminal_status='COMPLETE' if failed==0 else 'NUMERIC_OR_RUNTIME_FAILURE'
    except BaseException as exc:
      terminal_status='TERMINATED_WITH_EXCEPTION';(folder/'terminal_exception.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),'timestamp_utc':stamp()},indent=2));raise
    finally:
      (folder/'summary.json').write_text(json.dumps({'completed_utc':stamp(),'mode':args.mode,'successful':successful,'failed':failed,'expected':len(indices)*6,'pid':PID,'terminal_status':terminal_status},indent=2))
    if failed:raise SystemExit(2)
if __name__=='__main__':main()
