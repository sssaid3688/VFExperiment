"""Recover only source bytes that match already-recorded manifest SHA256."""
import datetime,difflib,hashlib,json
from pathlib import Path
root=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_remaining')
dest=root/'cpu_audit/harness_recovery';dest.mkdir(parents=True,exist_ok=True)
digest=lambda b:hashlib.sha256(b).hexdigest()
manifests=[]
for p in sorted((root/'runs/base_fp8_v1/silu').glob('*/*/manifest.json')):
 j=json.loads(p.read_text());manifests.append({'path':str(p),'harness_sha256':j['harness_sha256'],'adapter_sha256':j['module_sha256'],'dtype':j['dtype'],'mode':j['mode']})
assert manifests,'No target manifests'
targets={m['harness_sha256'] for m in manifests};adapter_targets={m['adapter_sha256'] for m in manifests}
source_paths=[root/'runner.py',*(root/'runs').glob('*/*/*/*/harness.py.snapshot')]
unique={}
for p in source_paths:
 raw=p.read_bytes();unique.setdefault(digest(raw),(p,raw))
attempts=[];matches={}
for original_sha,(p,raw) in unique.items():
 variants={'unchanged':raw}
 filtered=b''.join(line for line in raw.splitlines(keepends=True) if b"(folder/'harness.py.snapshot').write_bytes" not in line and b"(folder/'adapter.py.snapshot').write_bytes" not in line)
 variants['remove_only_two_snapshot_writer_lines']=filtered
 for name,b in list(variants.items()):
  lf=b.replace(b'\r\n',b'\n');variants[name+'_LF']=lf;variants[name+'_CRLF']=lf.replace(b'\n',b'\r\n')
 for method,b in variants.items():
  h=digest(b);attempts.append({'source':str(p),'source_sha256':original_sha,'method':method,'candidate_sha256':h,'bytes':len(b),'matches_target':h in targets})
  if h in targets and h not in matches:
   path=dest/f'recovered_harness_manifest_sha_{h}.py';path.write_bytes(b);assert digest(path.read_bytes())==h
   diff=''.join(difflib.unified_diff(raw.decode().splitlines(keepends=True),b.decode().splitlines(keepends=True),fromfile=str(p),tofile=path.name))
   (dest/f'harness_recovery_{h}.diff').write_text(diff)
   matches[h]={'path':str(path),'sha256':h,'bytes':len(b),'reconstructed_from':str(p),'source_sha256':original_sha,'transformation':method,'label':'Recovered exact manifest-matching bytes; NOT an original run-time snapshot'}
adapters={}
paths=[root/'base_ops.py',*(root/'runs').glob('*/*/*/*/adapter.py.snapshot')]
for p in paths:
 raw=p.read_bytes();h=digest(raw)
 if h in adapter_targets and h not in adapters:
  q=dest/f'recovered_adapter_manifest_sha_{h}.py';q.write_bytes(raw);assert digest(q.read_bytes())==h
  adapters[h]={'path':str(q),'sha256':h,'bytes':len(raw),'copied_from':str(p),'transformation':'none; existing file bytes exactly match early manifest','label':'Recovered from byte-identical retained file; NOT an original run-time snapshot'}
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Earliest base_fp8_v1 SiLU smoke/full missing source snapshots','target_manifests':manifests,'harness_recovered':matches,'adapters_recovered':adapters,'missing_harness_hashes':sorted(targets-set(matches)),'missing_adapter_hashes':sorted(adapter_targets-set(adapters)),'recovery_candidates_checked':attempts,'runtime_snapshots_created_or_modified':False,'run_manifests_modified':False,'running_scripts_or_sources_modified':False,'GPU_work':'NONE; standard library CPU/file operations only','limitation':'This recovers exact bytes attested by prior manifest hashes, not original file metadata or proof beyond the original manifest hash. The recovered files are explicitly labelled and kept solely under cpu_audit/harness_recovery.'}
(dest/'recovery_report.json').write_text(json.dumps(report,indent=2))
md=['# Early SiLU harness recovery','',f"UTC: {report['created_utc']}",'','Recovered source files are explicitly labelled reconstructions or copies matching pre-existing manifest hashes. They are not presented as original run-time snapshots. Original runs/manifests/source files were not edited.','',f'Observed target manifests: {len(manifests)}',f'Harness targets recovered: {len(matches)}/{len(targets)}',f'Adapter targets recovered: {len(adapters)}/{len(adapter_targets)}','']
for h,r in matches.items():md += [f"- Harness `{h}`: {r['bytes']} bytes, {r['transformation']}; source `{r['reconstructed_from']}`."]
for h,r in adapters.items():md += [f"- Adapter `{h}`: {r['bytes']} bytes, unchanged copy from `{r['copied_from']}`."]
md += ['',report['limitation'],'','Missing harness hashes: '+json.dumps(report['missing_harness_hashes']),'Missing adapter hashes: '+json.dumps(report['missing_adapter_hashes']),'','No GPU library was imported and no GPU work was launched.']
(dest/'recovery_report.md').write_text('\n'.join(md)+'\n')
print(json.dumps({k:report[k] for k in ['scope','harness_recovered','adapters_recovered','missing_harness_hashes','missing_adapter_hashes','limitation']},indent=2))
