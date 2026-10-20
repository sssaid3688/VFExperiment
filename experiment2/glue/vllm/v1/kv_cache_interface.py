"""Minimal stub of vllm.v1.kv_cache_interface.KVQuantMode.

Values must match the kernel's constexpr expectations (see _cast_kv_tile
docstring in triton_unified_attention.py):
  0 = NONE, 1 = FP8_PER_TENSOR, 2 = INT8_PER_TOKEN_HEAD, 3 = FP8_PER_TOKEN_HEAD
"""

from enum import IntEnum


class KVQuantMode(IntEnum):
    NONE = 0
    CHARGED = 0  # vLLM alias of NONE
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
