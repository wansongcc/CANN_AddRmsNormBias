#!/usr/bin/env python3
"""Compare an AddRmsNormBias device result with its golden output."""

import argparse
import pathlib

import numpy as np

from gen_data import resolve_dtype


def compare_arrays(actual, golden, dtype_name):
    if actual.shape != golden.shape:
        return False, {
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "mismatch_count": abs(actual.size - golden.size),
        }

    threshold = 1e-4 if dtype_name == "fp32" else 1e-3
    actual_f = actual.astype(np.float32)
    golden_f = golden.astype(np.float32)
    close = np.isclose(actual_f, golden_f, rtol=threshold, atol=threshold,
                       equal_nan=True)

    finite_pairs = np.isfinite(actual_f) & np.isfinite(golden_f)
    if np.any(finite_pairs):
        abs_diff = np.abs(actual_f[finite_pairs] - golden_f[finite_pairs])
        denom = np.maximum(np.abs(golden_f[finite_pairs]), np.finfo(np.float32).tiny)
        max_abs = float(np.max(abs_diff))
        max_rel = float(np.max(abs_diff / denom))
    else:
        max_abs = 0.0
        max_rel = 0.0

    if np.any(~close & ~finite_pairs):
        max_abs = float("inf")
        max_rel = float("inf")
    metrics = {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mismatch_count": int(np.count_nonzero(~close)),
    }
    return bool(np.all(close)), metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=pathlib.Path, required=True)
    parser.add_argument("--golden", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    actual = np.fromfile(args.actual, dtype=dtype)
    golden = np.fromfile(args.golden, dtype=dtype)
    passed, metrics = compare_arrays(actual, golden, args.dtype)
    result = "PASS" if passed else "FAIL"
    print(f"{result}: mismatches={metrics['mismatch_count']} "
          f"max_abs={metrics['max_abs']:.9g} max_rel={metrics['max_rel']:.9g}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
