"""CPU-only immutable upstream archive; never imports torch/triton."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'exp3-source-audit'}),timeout=60).read()
def resolve(repo,ref):
 # Freeze the audited main snapshot rather than following future main changes.
 ref={('triton-lang/triton','main'):'93035c4472eb820f78074de0e28341fa321a759d',('triton-lang/triton','v3.6.0'):'7c56a5e40f7fd928dfd5c72902d5def0097db73a'}.get((repo,ref),ref)
 return json.loads(get(f'https://api.github.com/repos/{repo}/commits/{ref}'))['sha']
records=[]
for repo,ref,files in [
 ('triton-lang/triton','v3.6.0',['python/tutorials/09-persistent-matmul.py','python/tutorials/06-fused-attention.py','python/tutorials/08-grouped-gemm.py','LICENSE','python/triton/tools/tensor_descriptor.py']),
 ('FlagOpen/FlagGems','30bfcc0735b8ebd48bd7339ef752ec64df69663f',['src/flag_gems/runtime/backend/_nvidia/__init__.py','src/flag_gems/ops/mm.py','src/flag_gems/ops/bmm.py','src/flag_gems/ops/group_gemm.py','LICENSE']),
 ('FlagOpen/FlagGems','32e4a4cf9b8618185a13eb82d72687fcdbbfb982',['src/flag_gems/runtime/backend/_nvidia/__init__.py','src/flag_gems/runtime/backend/__init__.py','src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py','src/flag_gems/ops/mm.py','LICENSE']),
 ('triton-lang/triton','main',['python/tutorials/09-persistent-matmul.py','python/tutorials/06-fused-attention.py','LICENSE']),
]:
 commit=resolve(repo,ref);directory=ROOT/(repo.split('/')[-1]+'_'+ref[:12]);directory.mkdir(exist_ok=True)
 for path in files:
  url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
  try:data=get(url)
  except Exception as e:
   records.append({'repo':repo,'ref':ref,'commit':commit,'path':path,'error':str(e)});continue
  target=directory/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  records.append({'repo':repo,'ref':ref,'commit':commit,'path':path,'archive':str(target.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'url':f'https://github.com/{repo}/blob/{commit}/{path}'})
(ROOT/'source_manifest.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
print(json.dumps(records,indent=2))
