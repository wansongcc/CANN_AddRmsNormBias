#!/usr/bin/env python3
"""NumPy reference for AddRmsNormBias."""

import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:  # The fp16/fp32 reference remains usable without ml_dtypes.
    bfloat16 = None


def _is_bf16(dtype):
    return "bfloat16" in str(dtype) or "bf16" in str(dtype)


def impl(x, residual, gamma, bias, epsilon=1e-5):
    """Return RMSNorm(x + residual, gamma, epsilon) + bias."""
    if x.shape != residual.shape:
        raise ValueError("x and residual must have identical shapes")
    if x.ndim < 2 or gamma.shape != (x.shape[-1],) or bias.shape != gamma.shape:
        raise ValueError("gamma and bias must match the final input dimension")

    original_dtype = x.dtype
    x_f = x.astype(np.float32)
    residual_f = residual.astype(np.float32)
    gamma_f = gamma.astype(np.float32)
    bias_f = bias.astype(np.float32)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = x_f + residual_f
        rms = np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + epsilon)
        out = y / rms * gamma_f + bias_f

    if _is_bf16(original_dtype):
        if bfloat16 is None:
            raise RuntimeError("bf16 requires the Python package ml_dtypes")
        return out.astype(bfloat16)
    if original_dtype == np.float16:
        return out.astype(np.float16)
    return out.astype(np.float32)
