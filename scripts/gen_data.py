#!/usr/bin/env python3
"""Generate deterministic AddRmsNormBias inputs and golden output."""

import argparse
import json
import pathlib

import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

from AddRmsNormBias import impl


def resolve_dtype(name):
    if name == "fp16":
        return np.float16
    if name == "fp32":
        return np.float32
    if name == "bf16":
        if bfloat16 is None:
            raise RuntimeError("bf16 requires the Python package ml_dtypes")
        return bfloat16
    raise ValueError("dtype must be fp16, bf16, or fp32")


def generate_case(rows, hidden, dtype_name, epsilon, seed, directory,
                  pattern="random"):
    if rows <= 0:
        raise ValueError("rows must be positive")
    if not 64 <= hidden <= 32768:
        raise ValueError("hidden must be in [64, 32768]")
    if pattern not in {"random", "zero", "nan", "inf"}:
        raise ValueError("pattern must be random, zero, nan, or inf")

    dtype = resolve_dtype(dtype_name)
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = (rows, hidden)

    if pattern == "random":
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.0, 2.0, size=shape).astype(dtype)
        residual = rng.uniform(-2.0, 2.0, size=shape).astype(dtype)
        gamma = rng.uniform(0.8, 1.2, size=hidden).astype(dtype)
        bias = rng.uniform(-0.3, 0.3, size=hidden).astype(dtype)
    else:
        x = np.zeros(shape, dtype=dtype)
        residual = np.zeros(shape, dtype=dtype)
        gamma = np.ones(hidden, dtype=dtype)
        bias = np.zeros(hidden, dtype=dtype)
        if pattern == "nan":
            x[...] = np.nan
        elif pattern == "inf":
            x[...] = np.inf

    golden = impl(x, residual, gamma, bias, epsilon)
    for name, value in (("x", x), ("residual", residual), ("gamma", gamma),
                        ("bias", bias), ("golden_output", golden)):
        value.tofile(directory / f"{name}.bin")

    metadata = {
        "rows": rows,
        "hidden": hidden,
        "shape": list(shape),
        "dtype": dtype_name,
        "epsilon": epsilon,
        "seed": seed,
        "pattern": pattern,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("input/case0"))
    parser.add_argument("--pattern", choices=("random", "zero", "nan", "inf"),
                        default="random")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = generate_case(args.rows, args.hidden, args.dtype, args.epsilon,
                             args.seed, args.output_dir, args.pattern)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
