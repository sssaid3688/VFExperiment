"""Minimal stub of vllm.platforms.current_platform for NPU.

triton_unified_attention.py uses:
  - current_platform.fp8_dtype()                      (float8 finfo constants)
  - current_platform.is_device_capability_family(100) (Blackwell tuning gate)

On Ascend NPU both take the non-CUDA path, which is exactly what we want:
fp8 constants remain numeric constants only, and the B200-specific launch
tuning (num_warps=8/stages=2) is skipped, leaving triton-ascend defaults.
"""

import torch


class _AscendPlatform:
    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn

    @staticmethod
    def is_cuda() -> bool:
        return False

    @staticmethod
    def is_device_capability_family(_major) -> bool:
        return False

    @staticmethod
    def get_device_capability(_idx=0):
        return (0, 0)


current_platform = _AscendPlatform()
