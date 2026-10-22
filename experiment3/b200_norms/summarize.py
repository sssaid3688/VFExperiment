import collections,hashlib,json,statistics
from pathlib import Path
root=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_norms')
src=root/'full/cases.jsonl';dest=root/'cases_summary.jsonl'
counts=collections.Counter();seen=set();events=groups=0;cv_high=[]
with src.open() as fi,dest.open('x') as fo:
 for line in fi:
  r=json.loads(line);key=(r['operator'],r['dtype'],tuple(r['shape']))
  assert key not in seen;seen.add(key);counts[(r['operator'],r['dtype'],r['status'])]+=1
  for g in r.get('groups',[]):
   vals=g.pop('samples_us');assert len(vals)==g['samples'] and len(vals)>0
   assert abs(statistics.median(vals)-g['median_us'])<1e-9
   events+=len(vals);groups+=1
  if r.get('group_median_cv',0)>.1:cv_high.append({'case_id':r['case_id'],'cv':r['group_median_cv']})
  fo.write(json.dumps(r,allow_nan=False)+'\n')
guards=[json.loads(x) for x in (root/'full/gpu_guards.jsonl').read_text().splitlines()]
report={'source_sha256':hashlib.file_digest(src.open('rb'),'sha256').hexdigest(),'cases':len(seen),'counts':{'/'.join(k):v for k,v in counts.items()},'raw_event_samples':events,'timing_groups':groups,'cv_above_10pct':cv_high,'gpu_guards':len(guards),'guards_with_foreign_pids':sum(bool(x['foreign_pids']) for x in guards),'timing_eligible':'B200 only; A5 matched timing still deferred; no completed cross-platform claim'}
(root/'analysis_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cv_above_10pct'},indent=2));print('CV>10%:',len(cv_high))
