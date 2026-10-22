"""CPU-only manifest/snapshot inventory, no GPU-library imports."""
import hashlib,json
from pathlib import Path
root=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_remaining')
records=[]
for p in sorted((root/'runs/base_fp8_v1/silu').glob('*/*/manifest.json')):
 j=json.loads(p.read_text());records.append({'manifest':str(p),'mode':j.get('mode'),'dtype':j.get('dtype'),'harness_sha256':j.get('harness_sha256'),'adapter_sha256':j.get('module_sha256'),'adapter_path':j.get('module_path'),'harness_snapshot_exists':(p.parent/'harness.py.snapshot').is_file(),'adapter_snapshot_exists':(p.parent/'adapter.py.snapshot').is_file()})
current=(root/'runner.py').read_bytes()
snapshots=[]
for p in (root/'runs').glob('*/*/*/*/harness.py.snapshot'):
 raw=p.read_bytes();snapshots.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
report={'records':records,'current_runner_sha256':hashlib.sha256(current).hexdigest(),'current_runner_bytes':len(current),'snapshot_digest_counts':{h:sum(s['sha256']==h for s in snapshots) for h in {s['sha256'] for s in snapshots}},'representative_snapshots':list({s['sha256']:s for s in snapshots}.values())}
dest=root/'cpu_audit/harness_recovery';dest.mkdir(parents=True,exist_ok=True)
(dest/'probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
