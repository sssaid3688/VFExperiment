"""Read installed package bytes without importing Torch/FlagGems or any GPU API."""
import hashlib,json,tarfile
from pathlib import Path
OUT=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_audit')
PKG=Path('/root/miniconda3/envs/kaixin/lib/python3.11/site-packages/flag_gems')
OUT.mkdir(parents=True,exist_ok=True)
files=[]
with tarfile.open(OUT/'installed_flaggems_sources.tar.gz','w:gz') as tar:
 for p in sorted(PKG.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py','.yaml','.yml','.json','.toml'):continue
  rel=p.relative_to(PKG).as_posix();b=p.read_bytes();tar.add(p,arcname='flag_gems/'+rel,recursive=False)
  files.append(dict(relative_path=rel,path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
 metadata=[]
 for d in PKG.parent.glob('flag_gems-*.dist-info'):
  for name in ('METADATA','direct_url.json','LICENSE','licenses/LICENSE'):
   p=d/name
   if p.is_file():
    tar.add(p,arcname='metadata/'+name,recursive=False)
    metadata.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
archive=OUT/'installed_flaggems_sources.tar.gz'
report=dict(package_path=str(PKG),no_package_imports=True,no_gpu_calls=True,files=files,metadata=metadata,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(OUT/'installed_source_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(files=len(files),archive_bytes=archive.stat().st_size,archive_sha256=report['archive_sha256'])))
