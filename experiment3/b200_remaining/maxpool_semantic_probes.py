"""Untimed edge diagnostics; serial device lock, separate from finite cohort."""
import fcntl,hashlib,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
with (ROOT/'device0.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    import torch
    import basic_ops,runner
    torch.set_num_threads(4)
    out=ROOT/'cpu_audit/maxpool_semantics';out.mkdir(parents=True,exist_ok=True)
    runner.guard(out)
    rows=[]
    for dtype in ['fp32','fp16']:
        for probe in ['nan','all_negative_infinity']:
            r={'operator':'max_pool2d','dtype':dtype,'probe':probe,'timed':False}
            try:
                c=basic_ops.build_max_pool2d_probe(dtype,probe)
                c['fn']();torch.cuda.synchronize()
                got=c['outputs']['output'].detach().cpu().double();ref=c['refs']['output'].double()
                gi=c['outputs']['indices'].detach().cpu();ri=c['refs']['indices']
                r.update(output_equal_including_nan=bool(((got==ref)|(got.isnan()&ref.isnan())).all()),indices_exact=bool(torch.equal(gi,ri)),first_output=float(got.flatten()[0]) if torch.isfinite(got.flatten()[0]) else str(float(got.flatten()[0])),first_reference=str(float(ref.flatten()[0])),first_index=int(gi.flatten()[0]),first_reference_index=int(ri.flatten()[0]),sources=c['sources'],contract=c['contract'])
            except Exception as e:r.update(error=repr(e),traceback=traceback.format_exc())
            rows.append(r)
    runner.guard(out)
    report={'scope':'Untimed diagnostic outside the 100-shape finite-input performance cohort','adapter_sha256':hashlib.sha256(Path(basic_ops.__file__).read_bytes()).hexdigest(),'records':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2,allow_nan=False))
