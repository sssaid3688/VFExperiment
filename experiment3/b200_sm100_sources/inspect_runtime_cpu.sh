set -eu
/root/miniconda3/envs/kaixin/bin/python - <<'PY'
import ast,hashlib,json,sys
from pathlib import Path
site=next(p for p in (Path(sys.prefix)/'lib').glob('python*/site-packages') if (p/'triton').is_dir())
records=[]
for relative in ['triton/backends/nvidia/compiler.py','triton/tools/tensor_descriptor.py','triton/language/core.py']:
 p=site/relative;s=p.read_text();t=ast.parse(s)
 r={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if relative.endswith('compiler.py'):
  node=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='CUDAOptions')
  r['CUDAOptions_fields']=[x.target.id for x in node.body if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name)]
  r['option_definitions']=[ast.get_source_segment(s,x) for x in node.body if isinstance(x,ast.AnnAssign) and getattr(x.target,'id','') in ['default_dot_input_precision','maxnreg','enable_fp_fusion','num_warps','num_stages','clc']]
 if relative.endswith('core.py'):
  r['has_tensor_descriptor']='class tensor_descriptor' in s;r['has_make_tensor_descriptor']='def make_tensor_descriptor' in s
  r['has_warp_specialize_keyword']='warp_specialize' in s
 records.append(r)
out=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_sm100_sources/runtime_cpu_api_audit.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'no_gpu_or_triton_import':True,'python_prefix':sys.prefix,'files':records},indent=2)+'\n')
print(out.read_text())
PY
