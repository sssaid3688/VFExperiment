"""Use actual saved source-gate summaries only. Never approve any source/IR."""
import collections,hashlib,json,importlib.util
from pathlib import Path
from qualification_contract import dependency_records,dependency_hashes
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
EVIDENCE=WORK/'b200_optimized_run/evidence/runs/sourcegate_20261022_v2'
queue=[];pairs=[];excluded=[];counts=collections.Counter()
for summary_path in sorted(EVIDENCE.rglob('summary.json')):
 summary=json.loads(summary_path.read_text());rel=summary_path.relative_to(EVIDENCE).parts
 if len(rel)!=5:continue
 provider,op,dtype,mode,_=rel
 if mode!='smoke':continue
 passed=summary.get('status')=='COMPLETE' and summary.get('counts')=={'PASS':2}
 category='PASS_TWO_SMOKE_SHAPES' if passed else 'UNSUPPORTED' if summary.get('counts',{}).get('UNSUPPORTED') else 'FAIL_OR_INCOMPLETE'
 counts[category]+=1
 if not passed or (provider.startswith('sm100') and op=='dense_matmul' and dtype=='fp32'):
  excluded.append({'provider':provider,'op':op,'dtype':dtype,'summary':summary,'reason':'SM100 DenseFP32 IEEE/precision/shared-memory qualification failed' if op=='dense_matmul' and dtype=='fp32' and provider.startswith('sm100') else category});continue
 manifest=json.loads((summary_path.parent/'manifest.json').read_text());plan=manifest['plan']
 adapter_path=WORK/'b200_optimized_audit/native_dispatch_candidates.py' if provider=='native' else WORK/'b200_sm100_sources/sm100_ops.py' if provider=='sm100' else WORK/'b200_optimized_audit/liger_blackwell_swiglu_candidate.py'
 spec=importlib.util.spec_from_file_location('review_dependency_only_'+provider,adapter_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 dependencies=dependency_records(module,provider,op,WORK,package_root=WORK/'b200_optimized_audit/installed/flag_gems')
 sourcegate_ir=[]
 for index in [0,99]:
  path=summary_path.parent/('compiled_'+str(index))/'compilation.json'
  if path.exists():
   relative=path.relative_to(EVIDENCE.parent.parent)
   remote='/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_optimized_run/'+relative.as_posix()
   sourcegate_ir.append({'path':remote,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'case_index':index})
 queue.append([provider,op,dtype])
 pairs.append({'provider':provider,'op':op,'dtype':dtype,'source_review_passed':False,'ir_review_passed':False,'precision_contract_review_passed':False,
  'adapter_sha256':manifest['module_sha256'],'adapter_path_as_smoked':manifest['module_path'],
  'plan_sha256':hashlib.sha256(json.dumps(plan,separators=(',',':'),sort_keys=True).encode()).hexdigest(),
  'runner_sha256':hashlib.sha256((ROOT/'runner_qualified.py').read_bytes()).hexdigest(),
  'capture_helper_sha256':hashlib.sha256((ROOT/'capture_scope.py').read_bytes()).hexdigest(),
  'contract_helper_sha256':hashlib.sha256((ROOT/'qualification_contract.py').read_bytes()).hexdigest(),
  'dependency_sha256':dependency_hashes(dependencies),'sourcegate_ir_evidence':sourcegate_ir,
  'reviewed_ir_evidence':[], 'source_smoke_summary_path':str(summary_path),'source_smoke_summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
  'root_review_note':'Pending actual IR/PTX and source/precision qualification; a PASS smoke or IEEE backend metadata alone is not approval.'})
doc={'status':'PENDING_ROOT_SOURCE_IR_REVIEW','approved_queue_sha256':None,'source_gate_counts':dict(counts),'pairs':pairs,'excluded':excluded,
 'fla_smoke_state':'New8pair queue prepared separately, not run; not included in this pending FULL queue until actual source gate and root review.',
 'execution_state':'NOT_STARTED','instructions':'Root reviews each pair actual TTIR/PTX/math contract, fills evidence, verifies all hashes, selects approved pairs, then explicitly changes status to APPROVED. No script auto-approves.'}
(ROOT/'queue_full_pending_review.json').write_text(json.dumps(queue,indent=2))
doc['queue_sha256_pending_root_review']=hashlib.sha256((ROOT/'queue_full_pending_review.json').read_bytes()).hexdigest()
(ROOT/'full_source_review_template.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'actual_source_gate_counts':dict(counts),'pending_full_pairs':len(queue),'excluded_pairs':len(excluded),'new_fla_smoke_pairs':8,'approval':'PENDING','GPU_or_queue_execution':False},indent=2))
