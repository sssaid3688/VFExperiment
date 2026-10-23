"""Pure metadata/source policy. No framework or GPU imports at import time."""
import hashlib
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dependency_records(module,provider,op,experiment_root,package_root=None):
 root=Path(experiment_root);audit=Path(module.__file__).resolve().parent;specs=[]
 if provider=='native' or (provider=='fla' and op=='causal_conv1d'):
  specs+=[('b200_remaining/matrix_attention_ops.py',root/'b200_remaining/matrix_attention_ops.py'),
   ('b200_remaining/matrix_upstream/provenance.json',root/'b200_remaining/matrix_upstream/provenance.json')]
 if provider=='fla' and op=='group_rmsnorm':
  specs+=[('b200_remaining/norm_rope_ops.py',root/'b200_remaining/norm_rope_ops.py'),
   ('b200_remaining/upstream/norm_rope/manifest.json',root/'b200_remaining/upstream/norm_rope/manifest.json')]
 if provider=='fla':specs+=[('fla_native_sources/manifest.json',audit/'fla_native_sources/manifest.json')]
 if provider.startswith('sm100'):specs+=[('b200_sm100_sources/extraction_manifest.json',audit/'extraction_manifest.json')]
 if provider=='native':
  pkg=Path(package_root) if package_root else Path('/root/miniconda3/envs/kaixin/lib/python3.11/site-packages/flag_gems')
  for rel in ['runtime/backend/_nvidia/tune_configs.yaml','runtime/backend/_nvidia/heuristics_config_utils.py','runtime/configloader.py','utils/libentry.py','runtime/__init__.py','runtime/backend/__init__.py','runtime/backend/_nvidia/__init__.py']:
   specs.append(('kaixin/flag_gems/'+rel,pkg/rel))
 return [{'key':key,'path':str(path),'sha256':sha(path)} for key,path in specs]
def dependency_hashes(records):return {r['key']:r['sha256'] for r in records}
def assert_dependencies_unchanged(records):
 for row in records:
  if sha(row['path'])!=row['sha256']:raise RuntimeError('FROZEN_DEPENDENCY_CHANGED: '+row['key'])
def validate_ir_evidence(pair):
 for field in ['sourcegate_ir_evidence','reviewed_ir_evidence']:
  refs=pair.get(field)
  if not isinstance(refs,list) or not refs:raise RuntimeError('Missing actual path/hash references: '+field)
  for ref in refs:
   if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256'):raise RuntimeError('IR evidence requires {path,sha256}: '+field)
   path=Path(ref['path'])
   if not path.is_file() or sha(path)!=ref['sha256']:raise RuntimeError('IR audit/evidence bytes differ from root approval: '+str(path))
def validate_queue_binding(doc,queue_path):
 if doc.get('status')!='APPROVED':raise RuntimeError('Root source/IR qualification is not approved')
 actual=sha(queue_path)
 if doc.get('approved_queue_sha256')!=actual:raise RuntimeError('ROOT_APPROVED_QUEUE_SHA256_MISMATCH')
 return actual
def logical_input_record(case,provider,op,tensorhash):
 """Hashes from actual CPU-rounded inputs, invert documented packing on CPU only."""
 inputs=case['cpu_inputs'];logical=dict(inputs);transforms={name:'identity' for name in inputs}
 pointer_metadata={'a_ptrs','b_ptrs','c_ptrs','gemm_sizes','gemm_lds'}
 for name,tensor in inputs.items():
  if tensor.device.type!='cpu':raise RuntimeError('Logical hashing requires actual CPU input: '+name)
 if provider.startswith('sm100') and op in ('dense_matmul','batched_matmul','group_gemm'):
  if 'b' not in inputs:raise RuntimeError('Documented packed B missing')
  shape=case['shape']
  expected=(shape[1],shape[2]) if op=='dense_matmul' else (shape[0],shape[2],shape[3]) if op=='batched_matmul' else (shape[0],shape[3],shape[2])
  if tuple(inputs['b'].shape)!=expected:raise RuntimeError('Unrecognized physical packed-B layout')
  logical['b']=inputs['b'].transpose(-1,-2).contiguous();transforms['b']='inverse documented B_T: CPU transpose(-1,-2).contiguous(), recover logical [...,K,N]'
 if provider.startswith('sm100') and op=='sdpa' and str(inputs['v'].dtype)=='torch.float8_e5m2':
  b,h,s,d=case['shape']
  if tuple(inputs['v'].shape)!=(b,h,d,s):raise RuntimeError('Unrecognized physical FP8 V packing')
  logical['v']=inputs['v'].transpose(-1,-2).contiguous();transforms['v']='inverse documented FP8 V packing: CPU transpose(-1,-2).contiguous(), recover [B,H,S,D]'
 if op=='causal_conv1d' and 'weight' in inputs:
  if tuple(inputs['weight'].shape)!=(case['shape'][2],case['shape'][3]):raise RuntimeError('Unrecognized physical FLA weight packing')
  logical['weight']=inputs['weight'].transpose(-1,-2).contiguous();transforms['weight']='inverse FLA physical [D,W] weight packing: CPU transpose(-1,-2).contiguous(), recover A5 logical [W,D]'
 mathematical={name:t for name,t in logical.items() if name not in pointer_metadata}
 return {'logical_input_hashes':{name:tensorhash(t) for name,t in mathematical.items()},
  'logical_input_shapes':{name:list(t.shape) for name,t in mathematical.items()},'logical_input_dtypes':{name:str(t.dtype) for name,t in mathematical.items()},
  'logical_input_transform':{name:transforms[name] for name in mathematical},
  'physical_metadata_hashes':{name:tensorhash(t) for name,t in inputs.items() if name in pointer_metadata},
  'logical_pairing_policy':'Compare exact logical shape, dtype, seed and all mathematical input hashes; never match different cohorts by index or compare device pointer metadata. Physical hashes remain separately preserved. Inverse packing is CPU-only and outside all timing.'}
