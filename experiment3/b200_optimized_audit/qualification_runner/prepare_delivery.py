"""Create source-only delivery manifest/archive. Does not execute any queue."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='delivery_manifest.json'}
doc={'kind':'SOURCE_ONLY_DELIVERY','GPU_execution':False,'full_approved':False,'files_sha256':files,
 'tests':'CPU-only compile/capture mocks/logical-layout and approval gates; no benchmark data generated',
 'smoke_launch':'bash /home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit/qualification_runner/launch_fla_source_smoke.sh'}
(ROOT/'delivery_manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
target=ROOT.parent/'qualification_runner_source_only.tar.gz'
with tarfile.open(target,'w:gz') as archive:
 for p in sorted(ROOT.iterdir()):
  if p.is_file():archive.add(p,arcname='qualification_runner/'+p.name)
print(json.dumps({'archive':str(target),'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'manifest_sha256':hashlib.sha256((ROOT/'delivery_manifest.json').read_bytes()).hexdigest(),
 'key_hashes':{key:files[key] for key in ['runner_qualified.py','capture_scope.py','qualification_contract.py','supervisor_qualified.py','queue_fla_source_smoke.json','queue_full_pending_review.json']},'GPU_execution':False},indent=2))
