"""Repair transport line endings only, retaining the original declared hash."""
import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--audit',required=True);a=p.parse_args()
base=Path(a.base).resolve();audit=Path(a.audit).resolve();audit.mkdir(parents=True,exist_ok=True)
manifest_path=base/'provenance.json';manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
m=manifest['causal_conv1d'];target=base/m['extracted_file'];raw=target.read_bytes();normalized=raw.replace(b'\r\n',b'\n')
sha=lambda x:hashlib.sha256(x).hexdigest()
assert raw.count(b'\r')==raw.count(b'\r\n'),'Unexplained standalone CR: refuse repair'
assert sha(normalized)==m['extracted_file_sha256'],'LF normalization does not recover fixed expected hash: refuse repair'
original=(base/m['archive_file']).read_bytes();assert sha(original)==m['sha256']
def function(text):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==m['extracted_function'])
ot=original.decode('utf-8');nt=normalized.decode('utf-8');on=function(ot);nn=function(nt)
obody=ast.get_source_segment(ot,on);nbody=ast.get_source_segment(nt,nn)
assert obody==nbody,'Exact original function source segment changed'
assert sha(nbody.encode())==m['function_body_sha256'],'Original fixed function hash mismatch'
on.decorator_list=[];nn.decorator_list=[]
assert ast.dump(on,include_attributes=False)==ast.dump(nn,include_attributes=False),'Kernel AST changed'
before=sha(raw);backup=base/(target.name+'.crlf_original.'+before+'.snapshot')
if raw!=normalized:
 if backup.exists():assert backup.read_bytes()==raw
 else:backup.write_bytes(raw)
 target.write_bytes(normalized)
assert sha(target.read_bytes())==m['extracted_file_sha256']
assert manifest_path.read_bytes()==manifest_raw,'Manifest must remain unchanged'
report={'target':str(target),'before_sha256':before,'expected_fixed_sha256':m['extracted_file_sha256'],'after_sha256':sha(target.read_bytes()),'before_bytes':len(raw),'after_bytes':len(normalized),'crlf_pairs_removed':raw.count(b'\r\n'),'only_change':'CRLF byte pairs converted to LF; no source token or operation changes','fixed_original_function_body_sha256':m['function_body_sha256'],'exact_original_function_text_equal':True,'function_AST_equal_ignoring_only_decorator_wrappers':True,'original_source_sha256':sha(original),'manifest_sha256_unchanged':sha(manifest_raw),'original_backup':str(backup) if raw!=normalized else None,'device_execution':'NONE; stdlib-only CPU repair'}
(audit/'conv_line_ending_repair.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
