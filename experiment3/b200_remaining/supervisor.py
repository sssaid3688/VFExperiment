"""Exactly one GPU worker at a time, finite jobs, source failures retained."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--batch',required=True);a=p.parse_args()
lock=(ROOT/'device0.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
folder=ROOT/'runs'/a.batch;folder.mkdir(parents=True,exist_ok=True)
queue=json.loads((ROOT/a.queue).read_text());ledger=folder/'supervisor.jsonl'
def record(x):
 with ledger.open('a') as f:f.write(json.dumps(x)+'\n');f.flush()
 print(json.dumps(x),flush=True)
def run(op,dtype,mode):
 target=folder/op/dtype/mode
 assert not target.exists(),'Explicit new revision needed; never silently retry: '+str(target)
 logs=folder/'logs';logs.mkdir(exist_ok=True);log=logs/f'{op}_{dtype}_{mode}.log'
 cmd=[os.sys.executable,'-B','-u',str(ROOT/'runner.py'),'--op',op,'--dtype',dtype,'--mode',mode,'--batch',a.batch]
 began=time.time();limit=300 if mode=='smoke' else 1200
 record({'event':'START','op':op,'dtype':dtype,'mode':mode,'hard_timeout_s':limit,'command':cmd,'time':began})
 with log.open('w') as f:
  proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=15s',str(limit)+'s',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  rc=proc.wait()
 summary=json.loads((target/'summary.json').read_text()) if (target/'summary.json').is_file() else {'status':'NO_SUMMARY'}
 record({'event':'END','op':op,'dtype':dtype,'mode':mode,'returncode':rc,'elapsed_s':time.time()-began,'summary':summary})
 return rc==0 and summary.get('status')=='COMPLETE' and summary.get('counts',{}).get('PASS')==(2 if mode=='smoke' else 100)
for op,dtype in queue:
 if run(op,dtype,'smoke'):run(op,dtype,'full')
 else:record({'event':'GATED_NO_FULL','op':op,'dtype':dtype})
record({'event':'QUEUE_COMPLETE','pairs':len(queue),'time':time.time()})
