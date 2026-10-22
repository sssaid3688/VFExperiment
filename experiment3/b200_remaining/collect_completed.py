"""Run only after finite GPU queues finish. Preserve and hash all raw events."""
import argparse,collections,gzip,hashlib,json,math,statistics,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--require-batches',nargs='+',required=True);a=ap.parse_args()
assert {p.name for p in (ROOT/'runs').iterdir() if p.is_dir()}==set(a.require_batches),'Unaccounted run batch'
for batch in a.require_batches:
 p=ROOT/'runs'/batch/'supervisor.jsonl';tail=json.loads(p.read_text().splitlines()[-1]);assert tail['event']=='QUEUE_COMPLETE',(batch,tail)
dest=ROOT/'collected'/a.tag;dest.mkdir(parents=True,exist_ok=False)
files=[];counts=collections.Counter();events=groups=0
with gzip.open(dest/'compact_records.jsonl.gz','wt',encoding='utf-8') as fo:
 for path in sorted((ROOT/'runs').glob('*/*/*/*/cases.jsonl')):
  summary=json.loads((path.parent/'summary.json').read_text());assert summary['status']=='COMPLETE',(path,summary)
  run_manifest=json.loads((path.parent/'manifest.json').read_text());plan=run_manifest['plan']
  assert len(plan)==100 and len({json.dumps(x,sort_keys=True) for x in plan})==100
  rel=path.relative_to(ROOT);_,batch,op,dt,mode,_=rel.parts
  guards=[json.loads(s) for s in (path.parent/'gpu_guards.jsonl').read_text().splitlines()]
  contaminated=any(g.get('foreign_pids') for g in guards)
  sha=hashlib.sha256();rows=0;fileevents=0;filegroups=0
  with path.open('rb') as fi:
   for line in fi:
    sha.update(line);r=json.loads(line);rows+=1
    assert r['shape']==plan[r['shape_index']],(path,r['case_id'],'shape/plan mismatch')
    raw_groups=r.get('groups',[])
    if mode=='full' and r['status']=='PASS':assert len(raw_groups)==5,(path,r['case_id'],'group count')
    if mode=='smoke':assert not raw_groups
    assert [g['group'] for g in raw_groups]==list(range(len(raw_groups)))
    for g in raw_groups:
     samples=g.pop('samples_us');assert len(samples)==g['samples'] and samples
     assert all(math.isfinite(x) and x>0 for x in samples),(path,r['case_id'],'invalid event time')
     assert statistics.median(samples)==g['median_us'],(path,r['case_id'])
     filegroups+=1;fileevents+=len(samples)
    r.update(batch=batch,mode=mode,raw_file=str(path),raw_relative_path=str(rel),timing_session_contaminated=contaminated)
    fo.write(json.dumps(r,allow_nan=False)+'\n');counts[(batch,op,dt,mode,r['status'])]+=1
  assert rows==sum(summary.get('counts',{}).values()),(path,rows,summary)
  if not summary.get('scope_wide_unsupported'):assert rows==summary['expected'],(path,'incomplete cases')
  files.append({'relative_path':str(rel),'path':str(path),'sha256':sha.hexdigest(),'bytes':path.stat().st_size,'rows':rows,'event_samples':fileevents,'timing_groups':filegroups,'gpu_guards':len(guards),'foreign_pid_snapshots':sum(bool(g.get('foreign_pids')) for g in guards)})
  events+=fileevents;groups+=filegroups
for source in (ROOT/'source_archive').glob('*/*'):
 assert hashlib.sha256(source.read_bytes()).hexdigest()==source.parent.name,source
report={'collected_unix':time.time(),'require_batches':a.require_batches,'files':files,'counts':{'/'.join(k):v for k,v in counts.items()},'event_samples':events,'timing_groups':groups,'all_raw_sha256_verified_during_streaming':True,'full_pass_five_groups_and_positive_finite_samples_verified':True,'case_shapes_match_unique_frozen_100_shape_plans':True,'source_archive_content_address_hashes_verified':True}
(dest/'collection_manifest.json').write_text(json.dumps(report,indent=2))
# Raw files remain in place. Tar includes exact sources, licenses and run logs;
# caches are excluded. Earlier six-operator raw evidence is included as well.
archive=BASE/'outputs/forward/B200_forward_raw_evidence.tar.gz';assert not archive.exists(),archive
names=['b200_remaining/runs','b200_remaining/source_archive','b200_remaining/matrix_upstream','b200_remaining/basic_upstream','b200_remaining/upstream','b200_remaining/cpu_audit','b200_remaining/collected/'+a.tag,'b200_pilot/full','b200_pilot/upstream','b200_norms/full','b200_norms/upstream','softmax_b200/full','softmax_b200/smoke','b200_pilot/benchmark.py','b200_pilot/run.sh','b200_norms/benchmark.py','b200_norms/run.sh','softmax_forward.py']
names += [str(p.relative_to(BASE)) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.json','.md','.diff']]
names=list(dict.fromkeys(names))
with tarfile.open(archive,'w:gz',compresslevel=6) as t:
 for name in names:
  p=BASE/name
  if p.exists():t.add(p,arcname=name)
digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
brief={'compact_sha256':hashlib.file_digest((dest/'compact_records.jsonl.gz').open('rb'),'sha256').hexdigest(),'raw_archive_path':str(archive),'raw_archive_sha256':digest,'raw_archive_bytes':archive.stat().st_size,'compact_path':str(dest/'compact_records.jsonl.gz'),'manifest_path':str(dest/'collection_manifest.json'),'event_samples_new_runs':events,'timing_groups_new_runs':groups,'case_files':len(files)}
(dest/'archive_manifest.json').write_text(json.dumps(brief,indent=2));print(json.dumps(brief,indent=2),flush=True)
