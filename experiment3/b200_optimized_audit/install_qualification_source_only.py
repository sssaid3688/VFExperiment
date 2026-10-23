"""Deploy only new source files; stdlib and bash syntax checks, never GPU."""
import hashlib,json,tarfile,subprocess
from pathlib import Path
ROOT=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit')
archive=ROOT/'qualification_runner_source_only.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='dfadd1ece87765162e44d23b98f08c471804b0eb7d55222fe82539c5ec0dd86e'
target=ROOT/'qualification_runner'
if target.exists():raise RuntimeError('NEW_SOURCE_DIRECTORY_EXISTS: refuse overwrite; no action')
with tarfile.open(archive) as bundle:
 for member in bundle.getmembers():
  path=ROOT/member.name
  assert member.isfile() and path.resolve().is_relative_to(target.resolve())
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bundle.extractfile(member).read())
manifest=json.loads((target/'delivery_manifest.json').read_text())
for name,digest in manifest['files_sha256'].items():
 path=target/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
 if path.suffix=='.py':compile(path.read_text(encoding='utf-8'),str(path),'exec')
 if path.suffix=='.sh':subprocess.run(['bash','-n',str(path)],check=True,timeout=10)
doc=json.loads((target/'full_source_review_template.json').read_text());checked=0
for pair in doc['pairs']:
 for ref in pair['sourcegate_ir_evidence']:
  path=Path(ref['path']);assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256'],str(path);checked+=1
 for key,digest in pair['dependency_sha256'].items():
  path=(Path('/root/miniconda3/envs/kaixin/lib/python3.11/site-packages/flag_gems')/key.removeprefix('kaixin/flag_gems/')) if key.startswith('kaixin/flag_gems/') else ROOT/key if key.startswith('fla_native_sources/') else ROOT.parent/key
  assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,(key,str(path))
 module=Path(pair['adapter_path_as_smoked']);assert hashlib.sha256(module.read_bytes()).hexdigest()==pair['adapter_sha256'],str(module)
print(json.dumps({'source_delivery':'PASS','source_files':len(manifest['files_sha256']),'actual_remote_sourcegate_IR_manifests':checked,
 'all_actual_remote_dependencies_match':True,'python_syntax':'PASS','bash_syntax':'PASS','GPU_import_or_execution':False,'queue_started':False,'full_qualification':doc['status']},indent=2))
