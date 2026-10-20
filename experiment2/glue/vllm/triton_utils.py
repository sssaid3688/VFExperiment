"""Minimal stub of vllm.triton_utils — plain triton re-export.

vLLM's real triton_utils layers kernel caching / patching on top of triton;
for running the kernel on triton-ascend a plain re-export is sufficient.
"""

import triton
from triton import language as tl  # noqa: F401

__all__ = ["triton", "tl"]
