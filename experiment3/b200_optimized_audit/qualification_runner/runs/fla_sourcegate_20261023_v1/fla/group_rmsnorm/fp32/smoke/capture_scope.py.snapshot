"""Scoped CPU-side compilation evidence capture. Import never touches GPU APIs.

Class run methods are patched ONLY across correctness calls and restored before
any timer/warmup. Optional LibEntry observation covers its compiled-kernel cache.
"""
import hashlib,json
from pathlib import Path

class CompileCapture:
 def __init__(self):
  self.patches=[];self.active=False;self.clear()
 def clear(self):
  if getattr(self,'active',False):raise RuntimeError('Cannot clear live capture')
  self.observations=0;self.by_hook={};self.kernels={};self.last=None
 def _observe(self,value,label):
  kernel=value[0] if isinstance(value,tuple) and value else value
  if kernel is None or not hasattr(kernel,'asm') or not hasattr(kernel,'metadata'):return
  order=self.observations;self.observations+=1;self.by_hook[label]=self.by_hook.get(label,0)+1
  self.kernels.setdefault(id(kernel),(order,kernel));self.last=kernel
 def install(self,jit_class,libentry_class=None):
  if self.active:raise RuntimeError('Capture is already active')
  self.clear();self.patches=[]
  for cls,label in [(jit_class,'JITFunction.run'),(libentry_class,'LibEntry.run')]:
   if cls is None:continue
   original=cls.run
   def make_wrapper(original,label):
    def observed(obj,*args,**kwargs):
     value=original(obj,*args,**kwargs);self._observe(value,label);return value
    return observed
   wrapper=make_wrapper(original,label);self.patches.append((cls,original,wrapper,label));cls.run=wrapper
  self.active=True
 def restore(self):
  if not self.active:return
  errors=[]
  for cls,original,wrapper,label in reversed(self.patches):
   if cls.run is wrapper:cls.run=original
   elif cls.run is not original:errors.append(label+' changed by another instrumenter')
  self.active=False
  if errors:raise RuntimeError('; '.join(errors))
  self.assert_restored()
 def assert_restored(self):
  if self.active or any(cls.run is not original for cls,original,_,_ in self.patches):raise RuntimeError('Instrumentation remains installed; timing forbidden')
 @staticmethod
 def _metadata(kernel):
  m=getattr(kernel,'metadata',None)
  if hasattr(m,'_asdict'):m=m._asdict()
  elif hasattr(m,'__dict__'):m=vars(m)
  return json.loads(json.dumps(m,default=str))
 def archive(self,folder):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);records=[];seen=set()
  for order,kernel in self.kernels.values():
   meta=self._metadata(kernel);asm=getattr(kernel,'asm',{})
   ident=hashlib.sha256((json.dumps(meta,sort_keys=True)+str(asm.get('ptx',''))).encode()).hexdigest()
   if ident in seen:continue
   seen.add(ident);target=folder/ident;target.mkdir(parents=True,exist_ok=True);files={}
   for ext,content in asm.items():
    raw=content.encode() if isinstance(content,str) else content if isinstance(content,bytes) else None
    if raw is None:continue
    p=target/('kernel.'+ext);p.write_bytes(raw);files[ext]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
   records.append({'first_hook_observation':order,'id':ident,'metadata':meta,'files':files})
  result={'hook_observation_count':self.observations,'hook_counts':self.by_hook,'unique_compiled_kernels':records,
   'last_observed_metadata':self._metadata(self.last) if self.last is not None else None,
   'scope':'Correctness stage only, including native tuner trials. Hook observations can overlap JIT/LibEntry and are not GPU-launch counts. Last returned kernel is recorded, but TF32/IEEE/architecture qualification requires actual IR/PTX inspection.'}
  p=folder/'compilation.json';p.write_text(json.dumps(result,indent=2))
  return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'hook_observation_count':self.observations,
   'unique_kernels':len(records),'last_observed_metadata':result['last_observed_metadata']}
