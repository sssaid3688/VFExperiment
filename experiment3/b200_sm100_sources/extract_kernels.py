"""Extract exact upstream definitions, replacing decorators only. CPU stdlib."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
specs=[('mm_ws','triton_v3.6.0/python/tutorials/09-persistent-matmul.py',['_compute_pid','matmul_kernel_tma_persistent','matmul_kernel_descriptor_persistent']),('attention_ws_v360','triton_v3.6.0/python/tutorials/06-fused-attention.py',['_attn_fwd_inner','_maybe_make_tensor_desc','_attn_fwd']),('attention_ws_corrected','triton_main/python/tutorials/06-fused-attention.py',['_attn_fwd_inner','_maybe_make_tensor_desc','_attn_fwd']),('group_tma','triton_v3.6.0/python/tutorials/08-grouped-gemm.py',['grouped_matmul_tma_kernel'])]
manifest=json.loads((ROOT/'source_manifest.json').read_text());records=[]
for name,path,names in specs:
 source=ROOT/path;raw=source.read_bytes();text=raw.decode();tree=ast.parse(text);parts=[];defs=[]
 for fname in names:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fname)
  body=ast.get_source_segment(text,node);parts.append('@triton.jit\n'+body+'\n')
  defs.append({'name':fname,'start_line':node.lineno,'end_line':node.end_lineno,'definition_sha256':hashlib.sha256(body.encode()).hexdigest()})
 # Full original source and MIT license remain in the archive. No tutorial
 # imports, tests, global device queries, cuBLAS handles or profiling execute.
 license_text=(ROOT/'triton_v3.6.0/LICENSE').read_text()
 notice=''.join('# '+line+'\n' for line in license_text.splitlines())
 if name=='group_tma':notice+='# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.\n'
 preamble='# Exact upstream function definitions; only decorators replaced by @triton.jit.\n'+notice+'import triton\nimport triton.language as tl\n\n'
 target=ROOT/'kernels'/(name+'.py');target.parent.mkdir(exist_ok=True);target.write_bytes((preamble+'\n'.join(parts)).encode())
 extracted=ast.parse(target.read_text())
 for before in [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]:
  after=next(n for n in extracted.body if isinstance(n,ast.FunctionDef) and n.name==before.name)
  before.decorator_list=[];after.decorator_list=[]
  assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
 record=next(m for m in manifest if m.get('archive')==path)
 assert hashlib.sha256(raw).hexdigest()==record['sha256']
 records.append({'name':name,'original':record,'extracted_file':str(target.relative_to(ROOT)).replace('\\','/'),'extracted_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'definitions':defs,'ast_equal_excluding_decorators':True,'transform':'Exact original source function definitions; wrapper autotune/profiler decorators replaced by @triton.jit. Fixed launch configs supplied by adapter. No statement/expression/math changes.'})
(ROOT/'extraction_manifest.json').write_bytes((json.dumps(records,indent=2)+'\n').encode())
print('Verified exact AST for',sum(len(r['definitions']) for r in records),'functions in',len(records),'modules; no torch/triton imported.')
