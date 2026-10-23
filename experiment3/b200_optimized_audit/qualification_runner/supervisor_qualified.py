"""One bounded serial queue, original device0.lock, periodic process evidence."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from qualification_contract import validate_queue_binding,validate_ir_evidence
ROOT=Path(__file__).resolve().parent
LOCK=ROOT.parents[1]/'b200_remaining/device0.lock'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scan_processes():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 return text,[int(line.split(',')[0]) for line in text.splitlines() if line.strip()]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True);parser.add_argument('--batch',required=True)
 parser.add_argument('--mode',choices=['smoke','full'],required=True);parser.add_argument('--qualification');parser.add_argument('--pair-timeout',type=int,default=900);a=parser.parse_args()
 queue_path=Path(a.queue).resolve();queue=json.loads(queue_path.read_text())
 if not isinstance(queue,list) or not queue or any(not isinstance(x,list) or len(x)!=3 for x in queue):raise ValueError('Queue requires [provider,op,dtype] triples')
 if a.pair_timeout<=0:raise ValueError('Positive hard timeout required')
 if a.mode=='full':
  if not a.qualification:raise RuntimeError('Root source/IR review pending; FULL not started')
  doc=json.loads(Path(a.qualification).read_text());validate_queue_binding(doc,queue_path)
  if any(provider.startswith('sm100') and op=='dense_matmul' and dtype=='fp32' for provider,op,dtype in queue):raise RuntimeError('SM100 DenseFP32 expressly excluded')
  for provider,op,dtype in queue:
   matches=[p for p in doc.get('pairs',[]) if [p.get('provider'),p.get('op'),p.get('dtype')]==[provider,op,dtype]]
   if len(matches)!=1 or not all(matches[0].get(k) is True for k in ['source_review_passed','ir_review_passed','precision_contract_review_passed']):raise RuntimeError('Queue contains a pair without root source/IR/precision approval')
   validate_ir_evidence(matches[0])
 lock=LOCK.open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('DEVICE0_BUSY_LOCK: no queue started; root may retry after current device owner finishes')
 folder=ROOT/'runs'/a.batch;folder.mkdir(parents=True,exist_ok=False);logs=folder/'logs';logs.mkdir()
 ledger=folder/'supervisor.jsonl';stop=folder/'STOP_REQUEST.json';monitor=folder/'gpu_monitor.jsonl'
 def emit(path,row):
  with path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
 def event(row):emit(ledger,dict(time=time.time(),**row));print(json.dumps(row),flush=True)
 metadata={'batch':a.batch,'mode':a.mode,'queue':queue,'queue_sha256':sha(queue_path),'runner_sha256':sha(ROOT/'runner_qualified.py'),
  'capture_helper_sha256':sha(ROOT/'capture_scope.py'),'contract_helper_sha256':sha(ROOT/'qualification_contract.py'),'supervisor_sha256':sha(__file__),'lock_path':str(LOCK),
  'pair_timeout_seconds':a.pair_timeout,'monitor_period_seconds':5,'qualification':a.qualification,
  'qualification_sha256':sha(a.qualification) if a.qualification else None,'status':'RUNNING'}
 (folder/'queue_manifest.json').write_text(json.dumps(metadata,indent=2));(folder/'queue.json').write_text(json.dumps(queue,indent=2))
 text,pids=scan_processes();emit(monitor,{'time':time.time(),'stage':'before_queue','processes':text,'foreign_pids':pids})
 if pids:
  metadata.update(status='STOPPED_BUSY_BEFORE_QUEUE',finished_unix=time.time())
  (folder/'queue_manifest.json').write_text(json.dumps(metadata,indent=2));event({'event':'STOP_FOREIGN_GPU_BEFORE_QUEUE','pids':pids});return
 for provider,op,dtype in queue:
  if stop.exists():break
  cmd=[os.sys.executable,'-B','-u',str(ROOT/'runner_qualified.py'),'--provider',provider,'--op',op,'--dtype',dtype,'--mode',a.mode,'--batch',a.batch]
  if a.qualification:cmd+=['--qualification',str(Path(a.qualification).resolve())]
  env=os.environ.copy();env['EXPERIMENT3_DEVICE0_LOCK_FD']=str(lock.fileno());env['EXPERIMENT3_STOP_REQUEST']=str(stop)
  env['EXPERIMENT3_QUEUE_PATH']=str(queue_path)
  if a.mode=='full':env['EXPERIMENT3_APPROVED_QUEUE_SHA256']=sha(queue_path)
  began=time.monotonic();event({'event':'START','provider':provider,'op':op,'dtype':dtype})
  timed_out=False;contaminated=False
  with (logs/f'{provider}_{op}_{dtype}.log').open('w') as stream:
   process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
   last_scan=0
   while process.poll() is None:
    elapsed=time.monotonic()-began
    if elapsed>=a.pair_timeout:
     timed_out=True;event({'event':'HARD_TIMEOUT_TERM','pid':process.pid,'elapsed_s':elapsed})
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
     break
    if time.monotonic()-last_scan>=5:
     last_scan=time.monotonic()
     try:
      text,pids=scan_processes();foreign=[p for p in pids if p!=process.pid]
      emit(monitor,{'time':time.time(),'stage':'during_pair','pid':process.pid,'processes':text,'foreign_pids':foreign})
      if foreign:
       contaminated=True;stop.write_text(json.dumps({'reason':'FOREIGN_GPU_PROCESS','pid':process.pid,'foreign_pids':foreign,'time':time.time()}))
       event({'event':'STOP_REQUEST_FOREIGN_GPU','pid':process.pid,'foreign_pids':foreign})
     except Exception as error:
      contaminated=True;stop.write_text(json.dumps({'reason':'MONITOR_ERROR','error':repr(error),'time':time.time()}))
      event({'event':'STOP_REQUEST_MONITOR_ERROR','error':repr(error)})
    time.sleep(.25)
   rc=process.wait()
  target=folder/provider/op/dtype/a.mode;summary_path=target/'summary.json'
  summary=json.loads(summary_path.read_text()) if summary_path.exists() else {'status':'NO_SUMMARY'}
  status={'provider':provider,'op':op,'dtype':dtype,'returncode':rc,'elapsed_s':time.monotonic()-began,'hard_timeout':timed_out,
   'timing_session_contaminated':contaminated,'summary':summary}
  target.mkdir(parents=True,exist_ok=True);(target/'process_exit.json').write_text(json.dumps(status,indent=2));event({'event':'END',**status})
  if contaminated or 'GPU_INTERFERENCE' in summary.get('error','') or 'illegal memory access' in summary.get('error','').lower():break
 metadata['status']='STOPPED_CONTAMINATED' if stop.exists() else 'QUEUE_TERMINAL';metadata['finished_unix']=time.time()
 (folder/'queue_manifest.json').write_text(json.dumps(metadata,indent=2));event({'event':metadata['status']})
if __name__=='__main__':main()
