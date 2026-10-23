"""CPU metadata fixtures only; no framework imports or benchmark result files."""
import hashlib,json,tempfile,itertools
from pathlib import Path
from types import SimpleNamespace
from qualification_contract import logical_input_record,validate_queue_binding,validate_ir_evidence
class Tensor:
 def __init__(self,shape,values=None,dtype='torch.float16'):
  self.shape=tuple(shape);self.dtype=dtype;self.device=SimpleNamespace(type='cpu')
  self.values=list(range(__import__('math').prod(shape))) if values is None else values
 def transpose(self,a,b):
  n=len(self.shape);a%=n;b%=n;shape=list(self.shape);shape[a],shape[b]=shape[b],shape[a]
  values=[]
  for index in itertools.product(*(range(d) for d in shape)):
   source=list(index);source[a],source[b]=source[b],source[a];offset=0
   for i,d in zip(source,self.shape):offset=offset*d+i
   values.append(self.values[offset])
  return Tensor(shape,values,self.dtype)
 def contiguous(self):return self
def digest(t):return hashlib.sha256(json.dumps([t.shape,t.dtype,t.values]).encode()).hexdigest()
logical=Tensor((2,3));packed=logical.transpose(-1,-2)
record=logical_input_record({'cpu_inputs':{'a':Tensor((4,2)),'b':packed,'a_ptrs':Tensor((1,))},'shape':[4,3,2]},'sm100','dense_matmul',digest)
assert record['logical_input_hashes']['b']==digest(logical)
assert 'a_ptrs' not in record['logical_input_hashes'] and 'a_ptrs' in record['physical_metadata_hashes']
for dtype in ['torch.float16','torch.float8_e5m2']:
 v=Tensor((1,2,3,4),dtype=dtype);physical=v.transpose(-1,-2) if 'float8' in dtype else v
 record=logical_input_record({'cpu_inputs':{'v':physical},'shape':[1,2,3,4]},'sm100','sdpa',digest)
 assert record['logical_input_hashes']['v']==digest(v)
w=Tensor((3,5))
record=logical_input_record({'cpu_inputs':{'weight':w.transpose(-1,-2)},'shape':[1,8,5,3]},'fla','causal_conv1d',digest)
assert record['logical_input_hashes']['weight']==digest(w)
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'metadata_fixture.json';path.write_text('[]');sha=hashlib.sha256(path.read_bytes()).hexdigest()
 assert validate_queue_binding({'status':'APPROVED','approved_queue_sha256':sha},path)==sha
 for doc in [{'status':'PENDING','approved_queue_sha256':sha},{'status':'APPROVED','approved_queue_sha256':'0'*64}]:
  try:validate_queue_binding(doc,path)
  except RuntimeError:pass
  else:raise AssertionError('Unapproved or mismatched fixture accepted')
 ref={'path':str(path),'sha256':sha};validate_ir_evidence({'sourcegate_ir_evidence':[ref],'reviewed_ir_evidence':[ref]})
 try:validate_ir_evidence({'sourcegate_ir_evidence':[ref],'reviewed_ir_evidence':[]})
 except RuntimeError:pass
 else:raise AssertionError('Missing actual IR evidence accepted')
print('PASS CPU metadata fixtures: inverse packing, FP16 V identity, pointer exclusion, exact queue SHA and IR evidence gates; no GPU or benchmark data')
