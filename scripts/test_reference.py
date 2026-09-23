import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from AddRmsNormBias import impl
from gen_data import generate_case, resolve_dtype
from verify_result import compare_arrays


class ReferenceTest(unittest.TestCase):
    def test_fp32_matches_direct_formula(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
        residual = np.array([[0.1, 0.2, 0.3, 0.4]], np.float32)
        gamma = np.ones(4, np.float32)
        bias = np.full(4, 0.5, np.float32)
        y = x + residual
        expected = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + 1e-5) + bias
        np.testing.assert_allclose(
            impl(x, residual, gamma, bias), expected, rtol=1e-6, atol=1e-6
        )

    def test_zero_nan_and_inf_do_not_raise(self):
        for row in (np.zeros((1, 64), np.float32),
                    np.full((1, 64), np.nan, np.float32),
                    np.full((1, 64), np.inf, np.float32)):
            out = impl(row, np.zeros_like(row), np.ones(64, np.float32),
                       np.zeros(64, np.float32))
            self.assertEqual(row.shape, out.shape)

    def test_generator_writes_boundary_case_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            meta = generate_case(2, 4097, "fp16", 1e-5, 42, directory)
            self.assertEqual(8194, np.fromfile(directory / "x.bin",
                                              dtype=np.float16).size)
            self.assertEqual(4097, meta["hidden"])
            self.assertTrue((directory / "golden_output.bin").is_file())

    def test_compare_arrays_uses_competition_thresholds(self):
        golden = np.array([1.0, 2.0], np.float32)
        self.assertTrue(compare_arrays(golden + 5e-5, golden, "fp32")[0])
        self.assertFalse(compare_arrays(golden + 2e-3, golden, "fp32")[0])

    def test_unknown_dtype_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "fp16, bf16, or fp32"):
            resolve_dtype("int8")


if __name__ == "__main__":
    unittest.main()
