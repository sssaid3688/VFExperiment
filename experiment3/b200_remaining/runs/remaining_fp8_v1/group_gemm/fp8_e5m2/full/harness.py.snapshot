"""Serial, audited B200 forward-only execution. Raw evidence is append-only."""
import argparse,collections,datetime,hashlib,importlib,importlib.metadata,json,math,os,statistics,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODULES={**dict.fromkeys(['silu','gelu_tanh','swiglu','rms_norm','layer_norm','softmax'],'base_ops'),**dict.fromkeys(['log_softmax','embedding','cumsum','groupnorm','sigmoid','sum_reduce','max_pool2d','relu'],'basic_ops'),**dict.fromkeys(['residual_rmsnorm','residual_layernorm','rope','group_rmsnorm','mrope'],'norm_rope_ops'),**dict.fromkeys(['dense_matmul','batched_matmul','group_gemm','causal_conv1d','sdpa','kv_cache_store'],'matrix_attention_ops')}
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,r):
 with p.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
def guard(folder):
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem','--format=csv'],text=True)
 proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 foreign=[int(x.split(',')[0]) for x in proc.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 r={'timestamp_utc':stamp(),'gpu_csv':gpu,'processes':proc,'foreign_pids':foreign};emit(folder/'gpu_guards.jsonl',r)
 if foreign:raise RuntimeError('GPU_INTERFERENCE: '+str(foreign))
def tensorhash(t):
 import torch
 return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def check(case):
 import torch
 metrics={}
 assert set(case['outputs'])==set(case['refs']),'Output/reference key mismatch'
 for name,t in case['outputs'].items():
  ref=case['refs'][name];assert ref.device.type=='cpu','Reference must be CPU'
  got=t.detach().cpu();assert got.shape==ref.shape,(name,got.shape,ref.shape)
  if not got.is_floating_point():
   ok=bool(torch.equal(got,ref.to(got.dtype)));metrics[name]={'correct':ok,'exact_integer_check':True,'output_dtype':str(t.dtype)}
  else:
   got=got.double();ref=ref.double();err=(got-ref).abs()
   tol=case.get('output_tolerances',{}).get(name,{})
   rt=float(tol.get('rtol',case['rtol']));at=float(tol.get('atol',case['atol']))
   limit=at+rt*ref.abs();finite=bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
   ok=finite and bool((err<=limit).all())
   maxabs=float(err.max()) if finite else None
   rawref=case.get('math_refs',{}).get(name,ref).double();rawerr=(got-rawref).abs()
   metrics[name]={'correct':ok,'output_dtype':str(t.dtype),'max_abs_error':maxabs,'mathematical_max_abs_error':float(rawerr.max()) if finite else None,'reference_output_quantized':name in case.get('math_refs',{}) and not torch.equal(rawref,ref),'relative_l2':float(err.norm()/ref.norm().clamp_min(1e-30)) if finite else None,'max_tolerance_ratio':float((err/limit.clamp_min(1e-300)).max()) if finite else None,'rtol':rt,'atol':at,'finite':finite}
  if not metrics[name]['correct']:return False,metrics
 return True,metrics
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--op',choices=list(MODULES),required=True);ap.add_argument('--dtype',choices=['fp32','fp16','fp8_e4m3fn','fp8_e5m2'],required=True);ap.add_argument('--mode',choices=['smoke','full'],required=True);ap.add_argument('--batch',required=True);a=ap.parse_args()
 folder=ROOT/'runs'/a.batch/a.op/a.dtype/a.mode;folder.mkdir(parents=True,exist_ok=False)
 counts=collections.Counter();indices=[0,99] if a.mode=='smoke' else list(range(100))
 summary={'operator':a.op,'dtype':a.dtype,'mode':a.mode,'pid':os.getpid(),'expected':len(indices),'status':'STARTED','start_utc':stamp()}
 def save():
  summary.update(counts=dict(counts),updated_utc=stamp());(folder/'summary.json').write_text(json.dumps(summary,indent=2))
 save()
 try:
  guard(folder)
  import torch,triton,triton.testing
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  module=importlib.import_module(MODULES[a.op]);plan=module.shape_plan(a.op)
  (folder/'harness.py.snapshot').write_bytes(Path(__file__).read_bytes())
  (folder/'adapter.py.snapshot').write_bytes(Path(module.__file__).read_bytes())
  assert len(plan)==100 and len({json.dumps(x,sort_keys=True) for x in plan})==100,'100 unique prespecified shapes required'
  manifest={'created_utc':stamp(),'operator':a.op,'dtype':a.dtype,'mode':a.mode,'pid':os.getpid(),'plan':plan,'module_path':module.__file__,'module_sha256':sha(module.__file__),'harness_sha256':sha(__file__),'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'versions':{k:importlib.metadata.version(k) for k in ['torch','triton','flag-gems','liger-kernel']},'timer':'triton.testing.do_bench(warmup=25,rep=100,return_mode=all), five invocations; native cache clear outside event pair','baseline':'Authoritative NVIDIA Triton implementation, specialization only where individually evidenced','precision_rule':'True requested storage dtype. FP8 vector operations may compute in FP32; explicitly labelled, never attributed FP8 TensorCore peak. No harness fp16 substitution.'}
  (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
  for i in indices:
   guard(folder)
   row={'case_id':f'{a.op}_{a.dtype}_{i:03d}','operator':a.op,'dtype':a.dtype,'shape_index':i,'shape':plan[i],'seed':20261020+i,'pid':os.getpid(),'timestamp_utc':stamp(),'groups':[]}
   case=None
   try:
    case=module.build_case(a.op,a.dtype,i,'cuda:0')
    assert case['shape']==plan[i],('Case vs plan mismatch',case['shape'],plan[i])
    for key in ['outputs','refs','cpu_inputs','device_inputs','useful_bytes','additional_bytes','contract','sources','rtol','atol']:assert key in case,key
    row.update(useful_bytes=int(case['useful_bytes']),additional_stat_write_bytes=int(case['additional_bytes']),kernel_minimum_bytes=int(case['useful_bytes']+case['additional_bytes']),contract=case['contract'],sources=case['sources'],rtol=case['rtol'],atol=case['atol'],input_hashes={k:tensorhash(v) for k,v in case['cpu_inputs'].items()},actual_input_dtypes={k:str(v.dtype) for k,v in case['device_inputs'].items()})
    row['input_sha256']=row['input_hashes'].get('x',next(iter(row['input_hashes'].values())))
    for s in case['sources']:
     p=Path(s['path']);assert p.is_file(),s
     actual=sha(p);assert actual==s['sha256'],('Source hash mismatch',s)
     dest=ROOT/'source_archive'/actual/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists():dest.write_bytes(p.read_bytes())
    with torch.no_grad():case['fn']()
    torch.cuda.synchronize();ok,metrics=check(case);row['output_checks']=metrics
    if not ok:raise ValueError('NUMERIC_TOLERANCE_FAILURE')
    if a.mode=='full':
     with torch.no_grad():
      for g in range(5):
       vals=[float(v)*1000 for v in triton.testing.do_bench(case['fn'],warmup=25,rep=100,return_mode='all')]
       assert vals and min(vals)>0
       row['groups'].append({'group':g,'samples_us':vals,'samples':len(vals),'median_us':statistics.median(vals)})
     med=[g['median_us'] for g in row['groups']];lat=statistics.median(med)
     row.update(latency_us_median_of_group_medians=lat,group_median_cv=statistics.stdev(med)/statistics.mean(med),effective_GBps=row['useful_bytes']/lat/1000)
     torch.cuda.synchronize();ok,post=check(case);row['post_timing_checks']=post
     if not ok:raise ValueError('POST_TIMING_NUMERIC_FAILURE')
    for k,t in case['device_inputs'].items():
     assert k in row['input_hashes'],('Missing CPU input',k)
     if tensorhash(t)!=row['input_hashes'][k]:raise ValueError('INPUT_MUTATED: '+k)
    row['max_abs_error']=max((m.get('mathematical_max_abs_error',m.get('max_abs_error',0)) or 0) for m in metrics.values());row['status']='PASS'
   except NotImplementedError as e:row.update(status='UNSUPPORTED',error=str(e),traceback=traceback.format_exc())
   except Exception as e:row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
   emit(folder/'cases.jsonl',row);counts[row['status']]+=1;save()
   print(json.dumps({k:row.get(k) for k in ['case_id','status','error','max_abs_error','latency_us_median_of_group_medians']}),flush=True)
   if row['status']=='UNSUPPORTED':
    summary['scope_wide_unsupported']=True;break
   if 'illegal memory access' in row.get('error','').lower():raise RuntimeError('Abort after device memory error')
   del case
  guard(folder);summary['status']='COMPLETE'
 except BaseException as e:
  summary.update(status='CRASHED',error=repr(e),traceback=traceback.format_exc());raise
 finally:save()
 if counts['FAIL'] or counts['UNSUPPORTED']:raise SystemExit(2)
if __name__=='__main__':main()
