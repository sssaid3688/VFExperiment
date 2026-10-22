import collections,hashlib,json,statistics
from pathlib import Path
base=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/softmax_b200')
src=base/'full/all.jsonl';rows=[];counts=collections.Counter();n=0;cv=[]
for line in src.open():
 r=json.loads(line);counts[(r['dtype'],r['status'])]+=1
 for g in r['groups']:
  vals=g.pop('samples_us');assert len(vals)==g['samples']
  assert statistics.median(vals)==g['median_us'];n+=len(vals)
 if r.get('group_median_cv',0)>.1:cv.append({'case_id':r['case_id'],'cv':r['group_median_cv']})
 rows.append(r)
assert len(rows)==200 and len({(r['dtype'],tuple(r['shape'])) for r in rows})==200
out=base/'cases_summary.jsonl'
with out.open('x') as f:
 for r in rows:f.write(json.dumps(r)+'\n')
guards=[json.loads(s) for s in (base/'full/gpu_guards.jsonl').read_text().splitlines()]
report={'raw_sha256':hashlib.file_digest(src.open('rb'),'sha256').hexdigest(),'cases':len(rows),'counts':{'/'.join(k):v for k,v in counts.items()},'raw_event_samples':n,'timing_groups':sum(len(r['groups']) for r in rows),'cv_above_10pct':cv,'gpu_guards':len(guards),'foreign_pid_snapshots':sum(bool(g['foreign_pids']) for g in guards)}
(base/'analysis_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
