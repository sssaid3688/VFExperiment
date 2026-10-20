"""Minimal stub of vllm.envs — only what triton_unified_attention.py reads.

vLLM 代码用 `import vllm.envs as envs` 后取 `envs.VLLM_BATCH_INVARIANT`,
因此必须是模块级属性。
"""

import os

VLLM_BATCH_INVARIANT = os.getenv("VLLM_BATCH_INVARIANT", "0") == "1"


class _Envs:
    VLLM_BATCH_INVARIANT = VLLM_BATCH_INVARIANT


envs = _Envs()
