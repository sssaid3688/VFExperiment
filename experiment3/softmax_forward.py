"""Unmodified FlagGems forward kernel; CPU fp64 reference; bounded launcher."""
import argparse,datetime,hashlib,importlib,json,math,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--machine',choices=['a5','b200'],required=True);p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--index',type=int);p.add_argument('--dtype',choices=['fp16','fp32']);a=p.parse_args()
base=Path(__file__).resolve().parent
folder=base/('softmax_'+a.machine)/a.mode;folder.mkdir(parents=True,exist_ok=True)
tag=f'{a.dtype}_{a.index}' if a.index is not None else 'all'
recfile=folder/(tag+'.jsonl');assert not recfile.exists(),recfile
def emit(x):
 with recfile.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
 print(json.dumps({k:v for k,v in x.items() if k not in ['groups','traceback']}),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
 if a.machine=='a5':return {'performance_eligible':False,'reason':'Other pre-existing device0 process; correctness only'}
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 other=[int(x.split(',')[0]) for x in s.splitlines() if x.strip() and int(x.split(',')[0])!=os.getpid()]
 r={'timestamp':time.time(),'processes':s,'foreign_pids':other};
 with (folder/'gpu_guards.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 if other:raise RuntimeError('GPU occupied '+str(other))
 return {'performance_eligible':True}
initial=guard()
if a.machine=='a5':
 assert a.mode=='smoke'
 import legacy_smoke_helpers as helpers
 helpers.get_flag_gems()
import torch,triton,triton.testing
torch.set_num_threads(4)
device='npu:0' if a.machine=='a5' else 'cuda:0'
module=importlib.import_module('flag_gems.runtime.backend._ascend.ops.softmax' if a.machine=='a5' else 'flag_gems.ops.softmax')
(folder/'source.py').write_bytes(Path(module.__file__).read_bytes())
source={'path':module.__file__,'sha256':sha(module.__file__),'release':'FlagGems v5.0.2','upstream_commit':'30bfcc0735b8ebd48bd7339ef752ec64df69663f','classification':'Ascend-specific baseline; 950PR tuning not established' if a.machine=='a5' else 'Generic NVIDIA Triton baseline; B200-specific tuning not established'}
(folder/'manifest.json').write_text(json.dumps({'source':source,'harness_sha256':sha(__file__),'device':device,'python':sys.executable,'torch':torch.__version__,'triton':triton.__version__,'input':'CPU fp32 generator seed20261020+shape_index then CPU cast to dtype','formula':'softmax(x,lastdim), same input/output dtype, no mask','arithmetic_policy':'Unmodified source casts load to output dtype in one-tile branch; do not describe as all-fp32 accumulation without generated-IR audit','timed_scope':'direct Triton softmax_kernel_inner, preallocated output','fp8':'not qualified; not run'},indent=2))
shapes=[(m,n) for m in [1,2,4,8,16,32,64,128,256,512] for n in [64,128,256,512,768,1024,1536,2048,4096,8192]]
indices=[a.index] if a.index is not None else ([32,65] if a.mode=='smoke' else range(100))
dtypes=[a.dtype] if a.dtype else ['fp16','fp32']
def sync():return torch.npu.synchronize() if a.machine=='a5' else torch.cuda.synchronize()
failed=passed=0
for i in indices:
 eligibility=guard()
 for dn in dtypes:
  dt={'fp16':torch.float16,'fp32':torch.float32}[dn];M,N=shapes[i]
  row={'operator':'softmax','case_id':f'softmax_{dn}_{i:03d}','shape':[M,N],'shape_index':i,'dtype':dn,'seed':20261020+i,'source':source,'groups':[],**eligibility}
  try:
   gen=torch.Generator(device='cpu').manual_seed(20261020+i)
   cpu=torch.randn(M,N,generator=gen,dtype=torch.float32).to(dt);x=cpu.to(device);y=torch.empty_like(x)
   row['input_sha256']=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
   ref=torch.softmax(cpu.double(),-1);rt,at=(2e-5,2e-6) if dn=='fp32' else (2e-3,2e-3)
   fn=lambda:module.softmax_kernel_inner[(M,1,1)](y,x,M,N)
   fn();sync();got=y.cpu().double();err=(got-ref).abs();limit=at+rt*ref.abs()
   row.update(rtol=rt,atol=at,max_abs_error=err.max().item(),max_tolerance_ratio=(err/limit).max().item(),row_sum_max_error=(got.sum(-1)-1).abs().max().item(),useful_bytes=2*M*N*cpu.element_size())
   torch.testing.assert_close(got,ref,rtol=rt,atol=at)
   if a.mode=='full':
    for g in range(5):
     vals=[float(v)*1000 for v in triton.testing.do_bench(fn,warmup=25,rep=100,return_mode='all')]
     row['groups'].append({'group':g,'samples_us':vals,'samples':len(vals),'median_us':statistics.median(vals)})
    meds=[g['median_us'] for g in row['groups']];lat=statistics.median(meds)
    row.update(latency_us_median_of_group_medians=lat,group_median_cv=statistics.stdev(meds)/statistics.mean(meds),effective_GBps=row['useful_bytes']/lat/1000)
    sync();torch.testing.assert_close(y.cpu().double(),ref,rtol=rt,atol=at)
    assert hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()==row['input_sha256']
   row['status']='PASS';passed+=1
  except Exception as e:row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());failed+=1
  emit(row)
guard()
(folder/(tag+'_summary.json')).write_text(json.dumps({'successful':passed,'failed':failed,'expected':len(indices)*len(dtypes),'terminal_status':'COMPLETE','pid':os.getpid()},indent=2))
if failed:raise SystemExit(2)
