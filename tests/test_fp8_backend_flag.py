#!/usr/bin/env python3
"""boot.py must keep --fp8-gemm-backend when EXTRA_SGLANG_ARGS is set to something unrelated.

boot.py has import-time side effects, so the helper is extracted by AST and exec'd alone.
"""
import ast
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    tree = ast.parse((ROOT / "boot.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "with_fp8_gemm_backend")
    ns = {"os": os}
    exec(compile(ast.Module([fn], []), "boot.py", "exec"), ns)
    return ns["with_fp8_gemm_backend"]


class Fp8Backend(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.pop("DSV41_MXFP8_BACKEND", None)
        self.f = load()

    def tearDown(self):
        os.environ.pop("DSV41_MXFP8_BACKEND", None)
        if self.old is not None:
            os.environ["DSV41_MXFP8_BACKEND"] = self.old

    def test_unrelated_args_get_the_flag(self):
        self.assertEqual(self.f(["--enable-metrics"]),
                         ["--fp8-gemm-backend", "flashinfer_cutlass", "--enable-metrics"])
        self.assertEqual(self.f([]), ["--fp8-gemm-backend", "flashinfer_cutlass"])

    def test_user_choice_is_kept(self):
        for toks in (["--fp8-gemm-backend", "triton"], ["--fp8-gemm-backend=triton", "-x"]):
            self.assertEqual(self.f(list(toks)), toks)

    def test_off_adds_nothing(self):
        os.environ["DSV41_MXFP8_BACKEND"] = "off"
        self.assertEqual(self.f(["--enable-metrics"]), ["--enable-metrics"])


if __name__ == "__main__":
    unittest.main()
