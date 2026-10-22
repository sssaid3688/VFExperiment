"""Read-only cache audit; stdlib only, no torch/Triton/GPU imports."""
import collections,datetime,hashlib,json,re,tarfile
from pathlib import Path
root=Path('/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_remaining')
cache=root/'cache/triton';out=root/'cpu_audit/compiled_precision';out.mkdir(parents=True,exist_ok=True)
names=['mm_kernel_general','bmm_kernel','grouped_mm_kernel','_attn_fwd','reshape_and_cache_flash_kernel','causal_conv1d_fwd_kernel']
records=[];picked={};issues=[]
for name in names:
 for ir in sorted(cache.glob('*/'+name+'.ttir')):
  ptx=ir.with_suffix('.ptx');meta=ir.with_suffix('.json')
  if not (ptx.is_file() and meta.is_file()):continue
  text=ir.read_text();pt=ptx.read_text();j=json.loads(meta.read_text())
  header=next((l for l in text.splitlines() if 'tt.func public @' in l),'')
  dtype=re.search(r'!tt.ptr<([^,>]+)',header);dtype=dtype[1] if dtype else 'unknown'
  dots=[l.strip() for l in text.splitlines() if re.search(r'\btt\.dot\b',l)]
  precisions=[re.search(r'inputPrecision\s*=\s*([\w]+)',l) for l in dots]
  # Triton v3.6.0 TT_DotOp defaults to IEEE, and its optional assembly format
  # elides that default. Backend JSON's tf32 default is a different layer.
  precision_names=[m[1] if m else 'ieee' for m in precisions]
  mma=[l.strip() for l in pt.splitlines() if re.search(r'\b(?:mma|wgmma|tcgen05\.mma)\.',l) and not l.lstrip().startswith('//')]
  tf32=[l.strip() for l in pt.splitlines() if 'tf32' in l.lower() and not l.lstrip().startswith('//')]
  cvt_half=[l.strip() for l in text.splitlines() if ('arith.' in l or 'tt.fp_to_fp' in l) and re.search(r'\bf16\b',l)]
  casts=[l.strip() for l in text.splitlines() if 'tt.fp_to_fp' in l or 'arith.extf' in l or 'arith.truncf' in l]
  scalar_f32=[l.strip() for l in pt.splitlines() if re.search(r'\bfma\.[\w.]*f32',l) and not l.lstrip().startswith('//')]
  transfers=[l.strip() for l in text.splitlines() if 'tt.load' in l or 'tt.store' in l]
  target=next((l.strip() for l in pt.splitlines() if l.strip().startswith('.target')),'')
  r={'kernel':name,'cache_key':ir.parent.name,'first_pointer_dtype':dtype,'target':target,'metadata_target':j.get('target'),'metadata_default_dot_input_precision':j.get('default_dot_input_precision'),'metadata_max_num_imprecise_acc_default':j.get('max_num_imprecise_acc_default'),'dot_precisions':precision_names,'elided_ieee_default_count':sum(m is None for m in precisions),'dot_count':len(dots),'ttir_dot_lines':dots,'ptx_tensor_mma_lines':mma[:4],'ptx_f32_fma_lines':scalar_f32[:4],'ptx_tf32_instruction_lines':tf32[:4],'cache_half_arithmetic_lines':cvt_half[:10] if name=='reshape_and_cache_flash_kernel' else [],'cache_float_conversion_lines':casts if name=='reshape_and_cache_flash_kernel' else [],'cache_transfer_lines':transfers if name=='reshape_and_cache_flash_kernel' else [],'ttir_path':str(ir),'ttir_sha256':hashlib.sha256(ir.read_bytes()).hexdigest(),'ptx_path':str(ptx),'ptx_sha256':hashlib.sha256(ptx.read_bytes()).hexdigest(),'metadata_path':str(meta),'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest()}
  records.append(r)
  if target!='.target sm_100a':issues.append({'cache_key':ir.parent.name,'kind':'UNEXPECTED_TARGET','target':target})
  if name in ['mm_kernel_general','bmm_kernel','grouped_mm_kernel','_attn_fwd'] and ('f32' in dtype or dtype=='unknown'):
   if not dots or any(p!='ieee' for p in precision_names) or tf32:issues.append({'cache_key':ir.parent.name,'kind':'FP32_IEEE_NOT_PROVED','dtype':dtype,'precisions':precision_names,'tf32':tf32})
  if name=='reshape_and_cache_flash_kernel' and dtype.startswith('f8') and cvt_half:issues.append({'cache_key':ir.parent.name,'kind':'CACHE_FP8_HAS_HALF_ARITHMETIC','lines':cvt_half})
  bucket=(name,dtype)
  if bucket not in picked:picked[bucket]=ir
counts=collections.Counter((r['kernel'],r['first_pointer_dtype']) for r in records)
fp8_cache=[r for r in records if r['kernel']=='reshape_and_cache_flash_kernel' and r['first_pointer_dtype'].startswith('f8')]
fp8_dot=[r for r in records if r['kernel'] in names[:4] and r['first_pointer_dtype'].startswith('f8')]
fp8_findings={'cache_completed_variants':len(fp8_cache),'cache_all_have_no_fp16_arithmetic_or_float_conversion':bool(fp8_cache) and all(not r['cache_half_arithmetic_lines'] and not r['cache_float_conversion_lines'] for r in fp8_cache),'cache_direct_transfer_dtype_groups':dict(collections.Counter(r['first_pointer_dtype'] for r in fp8_cache)),'dot_completed_variants':len(fp8_dot),'dot_all_have_fp8_TTIR_operands':bool(fp8_dot) and all(r['ttir_dot_lines'] and all('xf8' in line for line in r['ttir_dot_lines']) for r in fp8_dot),'dot_all_have_tensor_mma_instructions':bool(fp8_dot) and all(r['ptx_tensor_mma_lines'] for r in fp8_dot),'interpretation':'Completed FP8 dot artifacts require actual f8 TTIR operands and tensor-MMA PTX; completed FP8 cache artifacts require direct load/store and no floating conversion. These are compiler-path findings, not timing or numerical result claims.'}
brief={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cache_root':str(cache),'audit_scope':'Current completed compiler-cache artifacts only, no GPU work, no inference from uncompiled dtypes/operators','records':len(records),'kernel_dtype_counts':{'/'.join(k):v for k,v in counts.items()},'missing_kernels':[n for n in names if not any(r['kernel']==n for r in records)],'issues':issues,'metadata_note':'default_dot_input_precision=tf32 in JSON is a backend default. Per-dot TTIR IEEE mode, including an elided default attribute, overrides it. A backend default alone does not prove TF32 execution.','dot_schema_source':'https://github.com/triton-lang/triton/blob/v3.6.0/include/triton/Dialect/Triton/IR/TritonOps.td#L626','dot_schema_default':'TT_DotOp inputPrecision is DefaultValuedAttr<TT_InputPrecisionAttr, InputPrecision::IEEE>; optional custom assembly omits the default. Missing printed inputPrecision therefore means ieee, not unspecified.','fp32_statement':'Only listed compiled f32 variants with all TTIR dots ieee and no PTX TF32 instructions qualify as IEEE input dot mode; missing operators remain pending.','cache_fp8_statement':'Only actual f8 compiled cache variants with direct f8 load/store and no f16 arithmetic are eligible for storage-copy claim; absence of compiled FP8 artifacts is not proof of support.'}
brief['fp8_compiler_findings']=fp8_findings
report={'summary':brief,'artifacts':records}
prior=out/'compiled_precision_audit.json'
if prior.exists() and not (out/'initial_parser_report.json').exists():(out/'initial_parser_report.json').write_bytes(prior.read_bytes())
prior.write_text(json.dumps(report,indent=2))
with tarfile.open(out/'representative_ir_evidence.tar.gz','w:gz') as archive:
 for (name,dtype),ir in picked.items():
  for ext in ['.ttir','.ptx','.json','.ttgir','.llir']:
   p=ir.with_suffix(ext)
   if p.is_file():archive.add(p,arcname=f'{name}/{dtype}/{ir.parent.name}/{p.name}')
lines=['# Compiled precision audit (CPU/read-only)','',f"Snapshot UTC: {brief['created_utc']}",'','This snapshot reads completed compiler-cache files only; it launches no device work and imports no GPU library.','', '| Kernel / input dtype | Compiled variants | Result |','|---|---:|---|']
for (name,dtype),count in sorted(counts.items()):
 subset=[r for r in records if r['kernel']==name and r['first_pointer_dtype']==dtype]
 ieee=all(all(p=='ieee' for p in r['dot_precisions']) and not r['ptx_tf32_instruction_lines'] for r in subset)
 result='all targets sm_100a; '+('per-dot IEEE, no PTX TF32' if name in names[:4] else 'storage/vector kernel; no tensor precision claim')
 if not ieee:result+='; REVIEW PRECISION'
 lines.append(f'| {name} / {dtype} | {count} | {result} |')
lines+=['','Pending/uncompiled kernels: '+', '.join(brief['missing_kernels']), '',brief['metadata_note'],'',brief['dot_schema_default'],'','Schema source: '+brief['dot_schema_source'],'',brief['fp32_statement'],'',brief['cache_fp8_statement'],'','FP8 compiler-path findings: '+json.dumps(fp8_findings),'','Issues: '+json.dumps(issues), '','Representative .ttir/.ttgir/.llir/.ptx and metadata are archived beside this report; all scanned artifacts have SHA256 in JSON.','Initial parser snapshot is retained separately; it incorrectly treated elided IEEE defaults as unspecified. That parser issue was corrected against the exact v3.6.0 upstream dialect schema and is not an experiment failure.']
(out/'compiled_precision_audit.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(brief,indent=2))
