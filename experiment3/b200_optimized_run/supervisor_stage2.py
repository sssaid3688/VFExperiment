import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BATCH='sm100_fp32_stage2_20261022_v1'
lock=(ROOT.parent/'b200_remaining/device0.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
folder=ROOT/'runs'/BATCH;folder.mkdir(parents=True,exist_ok=False)
queue=json.loads((ROOT/'queue_stage2.json').read_text());ledger=folder/'supervisor.jsonl'
def record(x):
 with ledger.open('a') as f:f.write(json.dumps(x)+'\n');f.flush()
 print(json.dumps(x),flush=True)
for provider,op,dtype in queue:
 target=folder/provider/op/dtype/'smoke';logs=folder/'logs';logs.mkdir(exist_ok=True)
 cmd=[os.sys.executable,'-B','-u',str(ROOT/'runner_stage2.py'),'--provider',provider,'--op',op,'--dtype',dtype,'--mode','smoke','--batch',BATCH]
 began=time.time();record({'event':'START','provider':provider,'op':op,'dtype':dtype,'time':began})
 with (logs/f'{provider}_{op}_{dtype}.log').open('w') as f:
  rc=subprocess.run(['timeout','--signal=TERM','--kill-after=15s','360s',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 summary=json.loads((target/'summary.json').read_text()) if (target/'summary.json').is_file() else {'status':'NO_SUMMARY'}
 record({'event':'END','provider':provider,'op':op,'dtype':dtype,'returncode':rc,'elapsed_s':time.time()-began,'summary':summary})
 if 'GPU_INTERFERENCE' in summary.get('error',''):
  record({'event':'STOP_FOREIGN_GPU_PROCESS'});break
record({'event':'QUEUE_COMPLETE','time':time.time()})
