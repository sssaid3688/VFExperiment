"""Separate source/full qualification runner. Never started by preparation."""
import argparse,collections,datetime,hashlib,importlib,importlib.metadata,json,math,os,statistics,subprocess,sys,traceback
from pathlib import Path
from capture_scope import CompileCapture
from qualification_contract import dependency_records,dependency_hashes,assert_dependencies_unchanged,validate_ir_evidence,validate_queue_binding,logical_input_record
ROOT=Path(__file__).resolve().parent
PROVIDERS={'native':'native_dispatch_candidates','sm100':'sm100_ops','sm100_stage2':'sm100_ops_stage2','liger_wide':'liger_blackwell_swiglu_candidate','fla':'fla_native_candidates'}
BASE_RUNNER_SHA256='82c9f96cc2f85216c1945f2f026161d8ccd2d1c8d83c071a9a684cd9d5eaea7b'
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

def plan_hash(plan):return hashlib.sha256(json.dumps(plan,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def quantile(values,q):
 vals=sorted(values);position=(len(vals)-1)*q;lo=int(position);hi=min(lo+1,len(vals)-1)
 return vals[lo]+(vals[hi]-vals[lo])*(position-lo)
def assert_lock_lease():
 import fcntl
 value=os.environ.get('EXPERIMENT3_DEVICE0_LOCK_FD')
 if not value:raise RuntimeError('Run through the locked supervisor; direct device execution forbidden')
 fd=int(value);actual=Path(os.readlink('/proc/self/fd/'+str(fd))).resolve()
 expected=ROOT.parents[1]/'b200_remaining/device0.lock'
 if actual!=expected.resolve():raise RuntimeError('Unexpected device0 lock path')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stop_check():
 value=os.environ.get('EXPERIMENT3_STOP_REQUEST')
 if value and Path(value).exists():raise RuntimeError('GPU_INTERFERENCE_MONITOR_STOP_REQUEST: '+Path(value).read_text())
def review_gate(args,module,plan,dependencies):
 if args.mode!='full':return None
 if (args.provider.startswith('sm100') and args.op=='dense_matmul' and args.dtype=='fp32'):
  raise NotImplementedError('SM100_DENSE_FP32_EXCLUDED: observed TF32 semantics/precision or shared-memory failure; use qualified native IEEE baseline')
 if not args.qualification:raise RuntimeError('FULL requires root-reviewed source/IR qualification file; queue not auto-approved')
 p=Path(args.qualification);doc=json.loads(p.read_text())
 if doc.get('status')!='APPROVED':raise RuntimeError('Source/IR review pending; FULL forbidden')
 queue_path=os.environ.get('EXPERIMENT3_QUEUE_PATH')
 if not queue_path or validate_queue_binding(doc,queue_path)!=os.environ.get('EXPERIMENT3_APPROVED_QUEUE_SHA256'):raise RuntimeError('Full runner lacks root-approved exact queue lease')
 matches=[r for r in doc.get('pairs',[]) if [r.get('provider'),r.get('op'),r.get('dtype')]==[args.provider,args.op,args.dtype]]
 if len(matches)!=1:raise RuntimeError('Missing or duplicate pair qualification')
 r=matches[0]
 if not all(r.get(k) is True for k in ['source_review_passed','ir_review_passed','precision_contract_review_passed']):raise RuntimeError('Source/IR/precision qualification incomplete')
 for label,actual in [('adapter_sha256',sha(module.__file__)),('plan_sha256',plan_hash(plan)),('runner_sha256',sha(__file__)),('capture_helper_sha256',sha(ROOT/'capture_scope.py')),('contract_helper_sha256',sha(ROOT/'qualification_contract.py'))]:
  if r.get(label)!=actual:raise RuntimeError('Reviewed '+label+' differs from current source/plan')
 if r.get('dependency_sha256')!=dependency_hashes(dependencies):raise RuntimeError('Reviewed baseline adapter/config/manifest dependency hashes differ')
 validate_ir_evidence(r)
 return {'path':str(p),'sha256':sha(p),'pair':r}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--provider',choices=list(PROVIDERS),required=True);ap.add_argument('--op',required=True)
 ap.add_argument('--dtype',choices=['fp32','fp16','fp8_e4m3fn','fp8_e5m2'],required=True);ap.add_argument('--mode',choices=['smoke','full'],required=True)
 ap.add_argument('--batch',required=True);ap.add_argument('--qualification');a=ap.parse_args()
 assert_lock_lease();stop_check()
 module=importlib.import_module(PROVIDERS[a.provider]);plan=module.shape_plan(a.op)
 assert len(plan)==100 and len({json.dumps(x,sort_keys=True) for x in plan})==100,'100 prespecified unique shapes required'
 dependencies=dependency_records(module,a.provider,a.op,ROOT.parents[1]);qualification=review_gate(a,module,plan,dependencies)
 folder=ROOT/'runs'/a.batch/a.provider/a.op/a.dtype/a.mode;folder.mkdir(parents=True,exist_ok=False)
 counts=collections.Counter();indices=[0,99] if a.mode=='smoke' else list(range(100))
 summary={'provider':a.provider,'operator':a.op,'dtype':a.dtype,'mode':a.mode,'pid':os.getpid(),'expected':len(indices),'status':'STARTED','start_utc':stamp()}
 def save():
  summary.update(counts=dict(counts),updated_utc=stamp());(folder/'summary.json').write_text(json.dumps(summary,indent=2))
 save()
 try:
  guard(folder);stop_check()
  import torch,triton,triton.testing
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  for name,path in [('harness.py.snapshot',Path(__file__)),('capture_scope.py.snapshot',ROOT/'capture_scope.py'),('adapter.py.snapshot',Path(module.__file__))]:
   (folder/name).write_bytes(path.read_bytes())
  (folder/'qualification_contract.py.snapshot').write_bytes((ROOT/'qualification_contract.py').read_bytes())
  for dependency in dependencies:
   target=folder/'dependency_snapshots'/dependency['sha256']/Path(dependency['path']).name;target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(Path(dependency['path']).read_bytes())
  manifest={'created_utc':stamp(),'provider':a.provider,'operator':a.op,'dtype':a.dtype,'mode':a.mode,'pid':os.getpid(),
   'plan':plan,'plan_sha256':plan_hash(plan),'module_path':module.__file__,'module_sha256':sha(module.__file__),'harness_sha256':sha(__file__),
   'base_runner_sha256':BASE_RUNNER_SHA256,'capture_helper_sha256':sha(ROOT/'capture_scope.py'),'contract_helper_sha256':sha(ROOT/'qualification_contract.py'),'source_ir_qualification':qualification,
   'dependencies':dependencies,'root_approved_queue_sha256':os.environ.get('EXPERIMENT3_APPROVED_QUEUE_SHA256'),
   'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),
   'versions':{k:importlib.metadata.version(k) for k in ['torch','triton','flag-gems','liger-kernel']},
   'timer':'do_bench(warmup=25,rep=100,return_mode=all), 5 same-input/same-process repeated measurements, not statistically independent trials',
   'capture_policy':'Correctness only: capture original JIT/LibEntry returned compiled kernels; restore original class run methods in finally BEFORE any timer/warmup; reenable fresh for next case',
   'baseline':'Separate candidate batch; source/IR-qualified only when review file approved; no best-B200 or TF32/IEEE claim from backend metadata alone',
   'environment':{k:os.environ.get(k) for k in ['TRITON_CACHE_DIR','FLAGGEMS_CACHE_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR','FLA_CACHE_MODE','FLA_CACHE_RESULTS','FLA_GLUON','FLA_CONV_GLUON','FLA_CONV_BACKEND','ARCH','USE_FLAGTUNE']}}
  (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
  for i in indices:
   guard(folder);stop_check();assert_dependencies_unchanged(dependencies);capture=CompileCapture()
   row={'case_id':f'{a.op}_{a.dtype}_{i:03d}','provider':a.provider,'operator':a.op,'dtype':a.dtype,'shape_index':i,'shape':plan[i],
    'seed':20261020+i,'pid':os.getpid(),'timestamp_utc':stamp(),'groups':[],'performance_eligible':False};case=None
   try:
    case=module.build_case(a.op,a.dtype,i,'cuda:0')
    assert case['shape']==plan[i],('Case vs plan mismatch',case['shape'],plan[i])
    for key in ['outputs','refs','cpu_inputs','device_inputs','useful_bytes','additional_bytes','contract','sources','rtol','atol']:assert key in case,key
    row.update(useful_bytes=int(case['useful_bytes']),additional_stat_write_bytes=int(case['additional_bytes']),kernel_minimum_bytes=int(case['useful_bytes']+case['additional_bytes']),
     contract=case['contract'],sources=case['sources'],rtol=case['rtol'],atol=case['atol'],input_hashes={k:tensorhash(v) for k,v in case['cpu_inputs'].items()},
     actual_input_dtypes={k:str(v.dtype) for k,v in case['device_inputs'].items()})
    row['input_sha256']=row['input_hashes'].get('x',next(iter(row['input_hashes'].values())))
    row.update(logical_input_record(case,a.provider,a.op,tensorhash))
    for source in case['sources']:
     path=Path(source['path']);assert path.is_file() and sha(path)==source['sha256'],('Source mismatch',source)
     dest=ROOT/'source_archive'/source['sha256']/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists():dest.write_bytes(path.read_bytes())
    import triton.runtime.jit
    libentry_module=sys.modules.get('flag_gems.utils.libentry');libentry_class=getattr(libentry_module,'LibEntry',None)
    capture.install(triton.runtime.jit.JITFunction,libentry_class)
    try:
     with torch.no_grad():case['fn']()
     torch.cuda.synchronize()
    finally:capture.restore()
    capture.assert_restored();row['original_run_restored_before_timer']=True
    row['compilation']=capture.archive(folder/('compiled_'+str(i)))
    if not row['compilation']['unique_kernels']:raise RuntimeError('NO_COMPILED_KERNEL_OBSERVED_DURING_CORRECTNESS')
    if callable(case.get('native_tuning_metadata')):row['native_tuning_metadata']=case['native_tuning_metadata']()
    ok,metrics=check(case);row['output_checks']=metrics;capture.clear()
    if not ok:raise ValueError('NUMERIC_TOLERANCE_FAILURE')
    if a.mode=='full':
     with torch.no_grad():
      for g in range(5):
       guard(folder);stop_check();capture.assert_restored()
       vals=[float(v)*1000 for v in triton.testing.do_bench(case['fn'],warmup=25,rep=100,return_mode='all')]
       capture.assert_restored();assert vals and min(vals)>0
       q25,q75=quantile(vals,.25),quantile(vals,.75)
       row['groups'].append({'group':g,'samples_us':vals,'samples':len(vals),'median_us':statistics.median(vals),'q25_us':q25,'q75_us':q75,'iqr_us':q75-q25})
     med=[g['median_us'] for g in row['groups']];lat=statistics.median(med)
     row.update(latency_us_median_of_group_medians=lat,group_median_cv=statistics.stdev(med)/statistics.mean(med),effective_GBps=row['useful_bytes']/lat/1000)
     torch.cuda.synchronize();ok,post=check(case);row['post_timing_checks']=post
     if not ok:raise ValueError('POST_TIMING_NUMERIC_FAILURE')
    for name,tensor in case['device_inputs'].items():
     if tensorhash(tensor)!=row['input_hashes'][name]:raise ValueError('INPUT_MUTATED: '+name)
    stop_check();assert_dependencies_unchanged(dependencies);row['timing_session_contaminated']=False
    row['max_abs_error']=max((m.get('mathematical_max_abs_error',m.get('max_abs_error',0)) or 0) for m in metrics.values())
    row['status']='PASS';row['performance_eligible']=a.mode=='full'
   except NotImplementedError as error:row.update(status='UNSUPPORTED',error=str(error),traceback=traceback.format_exc())
   except Exception as error:row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
   finally:
    capture.restore();capture.assert_restored();row['capture_inactive_at_case_exit']=True
    if 'compilation' not in row and capture.observations:row['partial_failed_compilation']=capture.archive(folder/('failed_compiled_'+str(i)))
    capture.clear()
   if os.environ.get('EXPERIMENT3_STOP_REQUEST') and Path(os.environ['EXPERIMENT3_STOP_REQUEST']).exists():row['timing_session_contaminated']=True;row['performance_eligible']=False
   emit(folder/'cases.jsonl',row);counts[row['status']]+=1;save()
   print(json.dumps({k:row.get(k) for k in ['case_id','status','error','max_abs_error','latency_us_median_of_group_medians']}),flush=True)
   if row['status']=='UNSUPPORTED':summary['scope_wide_unsupported']=True;break
   if 'illegal memory access' in row.get('error','').lower():raise RuntimeError('Abort after device memory error')
   stop_check();del case
  guard(folder);stop_check();summary['status']='COMPLETE'
 except BaseException as error:
  summary.update(status='CRASHED',error=repr(error),traceback=traceback.format_exc());raise
 finally:save()
 if counts['FAIL'] or counts['UNSUPPORTED']:raise SystemExit(2)
if __name__=='__main__':main()
