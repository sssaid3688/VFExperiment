# Early SiLU harness recovery

UTC: 2026-10-21T07:06:14.370432+00:00

Recovered source files are explicitly labelled reconstructions or copies matching pre-existing manifest hashes. They are not presented as original run-time snapshots. Original runs/manifests/source files were not edited.

Observed target manifests: 4
Harness targets recovered: 1/1
Adapter targets recovered: 1/1

- Harness `cf7521d25720f1eab7316d2633af42511276ffadad937afe1b99388fb457bc2c`: 8383 bytes, remove_only_two_snapshot_writer_lines; source `/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_remaining/runs/base_fp8_v1/swiglu/fp8_e5m2/smoke/harness.py.snapshot`.
- Adapter `51288a27d522b3996bc81a2093fefd0a6a182872462b94cd8206822440f6bf10`: 4271 bytes, unchanged copy from `/home/ubuntu/workspace/oyhj/papers2/experiment3/b200_remaining/base_ops.py`.

This recovers exact bytes attested by prior manifest hashes, not original file metadata or proof beyond the original manifest hash. The recovered files are explicitly labelled and kept solely under cpu_audit/harness_recovery.

Missing harness hashes: []
Missing adapter hashes: []

No GPU library was imported and no GPU work was launched.
