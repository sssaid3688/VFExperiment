set -eu
/root/miniconda3/envs/kaixin/bin/python -B - <<'PY'
import os,signal,json
from pathlib import Path
pid=106828
expected=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_run')
p=Path('/proc')/str(pid)
assert p.exists()
assert (p/'cwd').resolve()==expected
assert b'supervisor.py' in (p/'cmdline').read_bytes()
os.kill(pid,signal.SIGSTOP)
def children(parent):
 f=Path('/proc')/str(parent)/'task'/str(parent)/'children'
 return [int(x) for x in f.read_text().split()] if f.exists() else []
def tree(parent):
 result=[]
 for child in children(parent):result+=tree(child)+[child]
 return result
targets=tree(pid)
details=[]
for child in targets:
 q=Path('/proc')/str(child)
 if not q.exists():continue
 details.append({'pid':child,'command':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')})
 try:os.kill(child,signal.SIGTERM)
 except ProcessLookupError:pass
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
(expected/'v1_stop.json').write_text(json.dumps({'reason':'Harness metadata JSON serialization failure; preserve all v1 evidence, restart as v2 with serialization-only fix','supervisor_pid':pid,'terminated_owned_descendants':details},indent=2))
print(json.dumps({'stopped_supervisor':pid,'descendants':targets}))
PY
