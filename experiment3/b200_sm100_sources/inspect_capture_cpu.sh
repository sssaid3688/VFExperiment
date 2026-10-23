set -eu
/root/miniconda3/envs/kaixin/bin/python - <<'PY'
import ast,hashlib,json,sys
from pathlib import Path
site=next(p for p in (Path(sys.prefix)/'lib').glob('python*/site-packages') if (p/'triton').is_dir())
out=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_sm100_sources/cpu_capture_audit');out.mkdir(exist_ok=True)
for rel in ['triton/runtime/jit.py','triton/compiler/compiler.py','triton/runtime/_allocation.py','triton/backends/nvidia/driver.py']:
 p=site/rel
 if not p.exists():continue
 raw=p.read_bytes();dst=out/rel.replace('/','__');dst.write_bytes(raw)
 print(json.dumps({'path':str(p),'archive':str(dst),'sha256':hashlib.sha256(raw).hexdigest()}))
 text=raw.decode();lines=text.splitlines()
 keys=['self.run =','def run(', 'return kernel','return self','def launch(', 'global_scratch','profile_scratch','allocator','def __getitem__','def _make_launcher','def warmup','def __init__']
 for i,line in enumerate(lines):
  if any(k in line for k in keys):print(f'{i+1}: '+line)
PY
