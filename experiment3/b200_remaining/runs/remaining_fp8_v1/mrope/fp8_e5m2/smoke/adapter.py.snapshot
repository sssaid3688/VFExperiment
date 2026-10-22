"""Five forward cases: direct, unmodified upstream Triton kernels only.

Importing this module does not import torch/FlagGems/FLA or create CUDA state.
The authoritative original files, exact function extractions, hashes and licenses
are archived under upstream/norm_rope. CPU reference construction is outside fn.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

OPS = ('residual_rmsnorm', 'residual_layernorm', 'rope', 'group_rmsnorm', 'mrope')
ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream' / 'norm_rope'
EPS = 1e-6
_MODULES = {}


def shape_plan(op):
    """Prespecified plans; all100 indices identify distinct tensor contracts."""
    if op not in OPS:
        raise ValueError(op)
    if op.startswith('residual_'):
        return [[m, d] for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                for d in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]]
    if op == 'group_rmsnorm':
        # One head per group is intentional: FLA's row id % G selects its own
        # group weight, exactly matching the Mojo grouped-head contract here.
        return [[t, 2 + (i % 3), d]
                for i, t in enumerate([1, 2, 4, 8, 16, 32, 64, 96, 128, 256])
                for d in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]]
    return [[1 if i < 5 else 2, s, 4 if i % 2 == 0 else 8, 2, d]
            for i, s in enumerate([16, 32, 64, 96, 128, 192, 256, 384, 512, 768])
            for d in [16, 32, 48, 64, 80, 96, 112, 128, 160, 256]]


def _kernel(key):
    """Only import the isolated exact kernel module, never upstream packages."""
    if key not in _MODULES:
        manifest = json.loads((UPSTREAM / 'manifest.json').read_text())
        meta = manifest[key]
        path = UPSTREAM / meta['extracted_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['extracted_sha256']
        module_name = '_b200_remaining_exact_norm_rope_' + key
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        _MODULES[key] = getattr(mod, meta['symbol'])
    return _MODULES[key]


def _sources(keys):
    manifest = json.loads((UPSTREAM / 'manifest.json').read_text())
    result = []
    for key in keys:
        m = manifest[key]
        for filename, hashkey, role in [(m['archive_path'], 'sha256', 'complete_unmodified_upstream_file'),
                                       (m['extracted_path'], 'extracted_sha256', 'exact_function_extraction')]:
            result.append(dict(path=str(UPSTREAM / filename), sha256=m[hashkey],
                               commit=m['commit'], upstream_commit=m['commit'], url=m['url'],
                               symbol=m['symbol'], role=role,
                               function_ast_unchanged=True))
    return result


def build_case(op, dtype, index, device='cuda:0'):
    import torch
    import triton
    if op not in OPS or not 0 <= index < 100:
        raise ValueError((op, index))
    names = {'fp32': 'float32', 'fp16': 'float16',
             'fp8_e4m3fn': 'float8_e4m3fn', 'fp8_e5m2': 'float8_e5m2'}
    if dtype not in names or not hasattr(torch, names[dtype]):
        raise NotImplementedError('Requested genuine storage dtype unavailable: ' + str(dtype))
    dt = getattr(torch, names[dtype])
    shape = shape_plan(op)[index]
    gen = torch.Generator(device='cpu').manual_seed(20261020 + index)
    cpu = {}

    def rand(name, dims):
        # Bounded amplitudes prevent unrelated FP8 overflow, while preserving
        # signed nonzero/nearzero values and the same seeded generator per case.
        tensor = (torch.randn(dims, generator=gen, dtype=torch.float32) * .5).to(dt)
        cpu[name] = tensor
        return tensor

    rt, at = {'fp32': (2e-5, 2e-6), 'fp16': (2e-3, 2e-3),
              'fp8_e4m3fn': (.063, 2**-10),
              'fp8_e5m2': (.126, 2**-17)}[dtype]
    out = {}
    refs = {}
    output_tol = {}
    contract = dict(operator=op, seed=20261020 + index,
                    reference='Independent CPU float64 formulas evaluated on actual CPU-quantized input values.',
                    input_generation='CPU seeded FP32 normal *0.5, then genuine requested storage dtype; no device-side harness casts.',
                    fp8_policy='Genuine requested FP8 input/output storage; arithmetic is individually disclosed below. Norm and RoPE use FP32; mRoPE retains source low-precision/cache policy. None is FP8 TensorCore compute.',
                    tolerance_policy='FP8 reference is unquantized FP64 mathematics; acceptance includes nearest-rounding half-relative-ULP bound plus half-subnormal step (E4M3:6.25%,2^-10; E5M2:12.5%,2^-17), with small FP32 margin. Not a relaxed FP16 substitute.',
                    source_policy='Exact original kernel function AST, isolated imports; fixed num_warps launch replaces only FLA autotuner selection. No mathematical edits.',
                    adapter_revision='mrope_cache_storage_match_v2_before_first_device_measurement',
                    targeted_B200_optimization=False)

    if op.startswith('residual_'):
        m, d = shape
        x = rand('x', (m, d)); r = rand('residual', (m, d)); w = rand('weight', (d,))
        if op == 'residual_layernorm':
            bias = rand('bias', (d,))
        inp = {name: value.to(device) for name, value in cpu.items()}
        residual_out = torch.empty((m, d), device=device, dtype=dt)
        y = torch.empty_like(residual_out)
        rstd = torch.empty(m, device=device, dtype=torch.float32)
        out.update(normalized=y, rounded_residual=residual_out, rstd=rstd)
        # Required residual rounding is a mathematical stage of this contract.
        # Quantize only the CPU reference here, never the device compute input.
        summed = (x.double() + r.double()).to(dt).double()
        refs['rounded_residual'] = summed
        output_tol['rounded_residual'] = {'rtol': 0., 'atol': 0.}
        block = triton.next_power_of_2(d)
        warps = 4 if block < 2048 else 8
        add = _kernel('add')
        add_grid = (triton.cdiv(m*d, 1024),)
        xd, rd, wd = inp['x'], inp['residual'], inp['weight']
        if op == 'residual_rmsnorm':
            norm = _kernel('rms_norm')
            refs['rstd'] = torch.rsqrt(summed.square().mean(-1) + EPS)
            refs['normalized'] = summed * refs['rstd'][:, None] * w.double()

            def fn():
                add[add_grid](xd, rd, residual_out, m*d, BLOCK_SIZE=1024)
                norm[(m,)](y, d, residual_out, d, wd, 1, rstd, 1, d, EPS, 0., 1,
                           elementwise_affine=True, BLOCK_SIZE=block, num_warps=warps)

            source_keys = ['add', 'rms_norm']
            useful = (4*m*d + d) * x.element_size()
            additional = m*d*x.element_size() + 4*m
            contract.update(formula='s=round_input_dtype(x+residual); y=s*rsqrt(mean(s*s,lastdim)+eps)*weight',
                            normalization_mode='GEMMA casting_mode=1, offset=0; FP32 normalization and weight multiply.')
        else:
            norm = _kernel('layer_norm')
            mean = torch.empty_like(rstd)
            out['mean'] = mean
            refs['mean'] = summed.mean(-1)
            centered = summed - refs['mean'][:, None]
            refs['rstd'] = torch.rsqrt(centered.square().mean(-1) + EPS)
            refs['normalized'] = centered * refs['rstd'][:, None] * w.double() + bias.double()
            bd = inp['bias']

            def fn():
                add[add_grid](xd, rd, residual_out, m*d, BLOCK_SIZE=1024)
                norm[(m,)](y, d, residual_out, d, wd, 1, bd, 1, mean, 1, rstd, 1,
                           d, EPS, BLOCK_SIZE=block, num_warps=warps)

            source_keys = ['add', 'layer_norm']
            useful = (4*m*d + 2*d) * x.element_size()
            additional = m*d*x.element_size() + 8*m
            contract.update(formula='s=round_input_dtype(x+residual); y=(s-mean(s))*rsqrt(mean((s-mean(s))^2)+eps)*weight+bias',
                            variance_policy='Centered-square FP32 variance; A5 source E[x^2]-E[x]^2 may round differently. No artificial matching rewrite.')
            output_tol['mean'] = {'rtol': 2e-5, 'atol': 2e-6}
        output_tol['rstd'] = {'rtol': 2e-5, 'atol': 2e-6}
        contract.update(eps=EPS, timed_scope='Two direct Triton kernels, add then norm; normalized and rounded residual outputs plus stats.',
                        additional_bytes_policy='Rounded residual is a required output; only its extra normalization read and stat writes are additional.',
                        launch=dict(add_BLOCK_SIZE=1024, norm_BLOCK_SIZE=block, norm_num_warps=warps),
                        baseline='Unfused composition of authoritative pure Triton kernels, not a vendor-native fused baseline.')

    elif op == 'group_rmsnorm':
        t, groups, d = shape
        x = rand('x', (t, groups, d)); w = rand('weight', (groups, d))
        inp = {name: value.to(device) for name, value in cpu.items()}
        y = torch.empty((t, groups, d), device=device, dtype=dt)
        rstd = torch.empty(t*groups, device=device, dtype=torch.float32)
        # The actual group-specific weights stay inside the timed kernel.
        out['normalized'] = y; out['rstd'] = rstd
        xd = x.double()
        refs['rstd'] = torch.rsqrt(xd.square().mean(-1) + EPS).reshape(-1)
        refs['normalized'] = xd * refs['rstd'].view(t, groups, 1) * w.double()[None, :, :]
        output_tol['rstd'] = {'rtol': 2e-5, 'atol': 2e-6}
        kernel = _kernel('group_rms')
        xdev, wdev = inp['x'], inp['weight']
        block = triton.next_power_of_2(d)
        warps = 4

        def fn():
            kernel[(t*groups,)](xdev, y, wdev, None, None, None, None, rstd, EPS,
                               G=groups, D=d, BD=block, IS_RMS_NORM=True,
                               HAS_RESIDUAL=False, STORE_RESIDUAL_OUT=False,
                               HAS_WEIGHT=True, HAS_BIAS=False, num_warps=warps)

        useful = (2*t*groups*d + groups*d) * x.element_size()
        additional = 4*t*groups
        source_keys = ['group_rms']
        contract.update(formula='y[t,g,d] = x[t,g,d]*rsqrt(mean_d(x[t,g,:]^2)+eps)*weight[g,d]',
                        group_layout='G=2..4, one head per group; FLA row_id%G selects group weight inside kernel. No pre-normalized or pre-weighted input.',
                        normalized_output_layout='Single [T,G,D] storage; slice[:,g:g+1,:] corresponds to A5 group g output.',
                        eps=EPS, timed_scope='One direct FLA row Triton RMS kernel; stats included.',
                        launch=dict(BD=block, num_warps=warps), additional_bytes_policy='FP32 rstd writes.')

    else:
        b, s, hq, hk, d = shape
        q = rand('q', (b, s, hq, d)); k = rand('k', (b, s, hk, d))
        inv = 1. / (10000. ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        positions = torch.arange(b*s, dtype=torch.float64).view(b, s)
        if op == 'rope':
            angle = positions[..., None] * inv[None, None, :]
        else:
            # Distinct multi-axis positional streams ensure the kernel's actual
            # T/H/W selection is tested, not equivalent preselected plain RoPE.
            axes = torch.stack([positions, positions*1.5 + .25, positions*.75 + .5])
            angle = axes[..., None] * inv[None, None, None, :]
        cpu['cos'] = torch.cat([angle.cos(), angle.cos()], -1).float()
        cpu['sin'] = torch.cat([angle.sin(), angle.sin()], -1).float()
        if op == 'mrope':
            # A5 casts selected FP32 cache coordinates to Q dtype inside its
            # public kernel. Give Liger cache values in that exact storage dtype
            # so its Q-to-cache cast does not silently force FP32 rotation.
            # This casts ancillary CPU cache INPUTS only, never Q/K compute data.
            cpu['cos'] = cpu['cos'].to(dt)
            cpu['sin'] = cpu['sin'].to(dt)
        inp = {name: value.to(device) for name, value in cpu.items()}
        qout = torch.empty(q.shape, device=device, dtype=dt)
        kout = torch.empty(k.shape, device=device, dtype=dt)
        out.update(q_rotated=qout, k_rotated=kout)
        copy = _kernel('copy')
        qdev, kdev, cdev, sdev = inp['q'], inp['k'], inp['cos'], inp['sin']
        nq, nk = q.numel(), k.numel()
        qgrid, kgrid = (triton.cdiv(nq, 1024),), (triton.cdiv(nk, 1024),)
        phq, phk, pd = map(triton.next_power_of_2, (hq, hk, d))
        rotate_block = max(phq, phk)
        half = d//2
        if op == 'rope':
            kernel = _kernel('rope')
            cos = cpu['cos'].double()[..., :half]
            sin = cpu['sin'].double()[..., :half]

            def fn():
                copy[qgrid](qdev, qout, nq, BLOCK_SIZE=1024)
                copy[kgrid](kdev, kout, nk, BLOCK_SIZE=1024)
                kernel[(b*s,)](qout, hq*d, kout, hk*d, cdev, d, sdev, d, s,
                               b, b, hq, hk, d, phq, phk, pd,
                               BLOCK_SIZE=rotate_block, BACKWARD_PASS=False)

            source_keys = ['copy', 'rope']
            contract.update(cache_contract='[B,S,D] FP32 cos/sin with duplicated NeoX halves; kernel reads first half. Aux cache stays FP32 for all Q/K storage precisions.')
        else:
            kernel = _kernel('mrope')
            nt, nh = d//6, d//6
            sections = [nt, nh, half-nt-nh]
            c, sn = cpu['cos'].double(), cpu['sin'].double()
            cos = torch.cat([c[0, ..., :nt], c[1, ..., nt:nt+nh], c[2, ..., nt+nh:half]], -1)
            sin = torch.cat([sn[0, ..., :nt], sn[1, ..., nt:nt+nh], sn[2, ..., nt+nh:half]], -1)

            def fn():
                copy[qgrid](qdev, qout, nq, BLOCK_SIZE=1024)
                copy[kgrid](kdev, kout, nk, BLOCK_SIZE=1024)
                kernel[(b*s,)](qout, kout, cdev, sdev, s, b, hq, hk, d, phq, phk, pd,
                               nt, nh, BLOCK_SIZE=rotate_block, BACKWARD_PASS=False)

            source_keys = ['copy', 'mrope']
            contract.update(cache_contract='[3,B,S,D] raw T/H/W cache values rounded to Q/K storage dtype, distinct axis positions, duplicated halves. Matches the A5 public kernel cast to Q dtype; all multi-axis selection remains inside the timed kernel.',
                            mrope_sections=sections, forbidden_shortcut='No preselected GPU caches or ordinary RoPE substitution.')
        for name, x in [('q_rotated', q), ('k_rotated', k)]:
            xd = x.double(); left, right = xd[..., :half], xd[..., half:]
            cc, ss = cos[:, :, None, :], sin[:, :, None, :]
            refs[name] = torch.cat([left*cc-right*ss, right*cc+left*ss], -1)
        # Logical input/output Q/K traffic plus the selected half-cache loads.
        # An in-place upstream kernel requires refresh copies inside each call:
        # copy read+write is the honest extra, never an out-of-timer reset.
        useful = 2*(nq+nk)*q.element_size() + b*s*d*cpu['cos'].element_size()
        additional = 2*(nq+nk)*q.element_size()
        contract.update(formula='NeoX half split: q/k_out = concat(x_left*cos-x_right*sin, x_right*cos+x_left*sin)',
                        arithmetic=('Q/K converted to FP32 cache dtype in source; FP32 rotation, final requested-dtype store.' if op == 'rope' else 'Q/K and selected caches share requested dtype. Arithmetic follows original Triton low-precision operators, final requested-dtype store. FP8 compile support is unproven and must not be replaced by FP32 caches.'),
                        timed_scope='Two exact Triton copy kernels + one exact in-place rotation kernel; refresh is inside every timed call.',
                        byte_policy='Useful Q/K read+write plus one selected cos/sin half per token; additional Q/K copy read+write. Unselected/duplicated cache coordinates are not counted as loaded bytes.',
                        input_immutability='Q/K source tensors are never mutated; outputs reset by included copies each invocation.',
                        launch=dict(grid=[b*s], pad_n_qh=phq, pad_n_kh=phk, pad_hd=pd))

    return dict(fn=fn, outputs=out, refs=refs, cpu_inputs=cpu, device_inputs=inp,
                useful_bytes=useful, additional_bytes=additional, shape=shape,
                contract=contract, sources=_sources(source_keys), rtol=rt, atol=at,
                output_tolerances=output_tol)
