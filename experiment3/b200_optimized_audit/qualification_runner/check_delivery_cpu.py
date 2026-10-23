"""Static delivery audit; no GPU/framework imports, no queue launch."""
import hashlib,json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for path in ROOT.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
sys.path.insert(0,str(ROOT))
import runner_qualified
assert 'torch' not in sys.modules and 'triton' not in sys.modules
doc=json.loads((ROOT/'full_source_review_template.json').read_text())
queue=json.loads((ROOT/'queue_full_pending_review.json').read_text())
smoke=json.loads((ROOT/'queue_fla_source_smoke.json').read_text())
assert len(queue)==22 and len(smoke)==8
assert doc['status']=='PENDING_ROOT_SOURCE_IR_REVIEW' and doc['approved_queue_sha256'] is None
assert all(len(p['sourcegate_ir_evidence'])==2 and not p['reviewed_ir_evidence'] for p in doc['pairs'])
assert all(not p['source_review_passed'] and not p['ir_review_passed'] and not p['precision_contract_review_passed'] for p in doc['pairs'])
assert not any(p.startswith('sm100') and o=='dense_matmul' and d=='fp32' for p,o,d in queue)
assert all(p['dependency_sha256'] or p['provider']=='liger_wide' for p in doc['pairs'])
candidate=ROOT.parent/'fla_native_candidates.py';spec=importlib.util.spec_from_file_location('fla_source_only',candidate)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for op in ['group_rmsnorm','causal_conv1d']:
 plan=module.shape_plan(op)
 assert len(plan)==100 and len({json.dumps(s) for s in plan})==100
assert 'torch' not in sys.modules and 'triton' not in sys.modules
print(json.dumps({'python_syntax':'PASS','safe_import_without_framework':'PASS','actual_saved_sourcegate_pairs':22,'actual_ir_manifests':44,
 'new_fla_source_smoke_pairs':8,'fla_unique_shapes_per_op':100,'full_qualification':'PENDING','gpu_execution':False},indent=2))
