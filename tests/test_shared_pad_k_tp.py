#!/usr/bin/env python3
"""CPU-only checks of the TP-size-derived shape constants in adapter/shared_pad_k.py.

torch is stubbed: only the module-level constants and _enabled() are exercised.
"""
import importlib.util
import os
import sys
import unittest.mock
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tp, flag="1", src_k=None):
    env = {"TP_SIZE": str(tp), "DSV41_SHARED_PAD_K": flag}
    if src_k:
        env["DSV41_SHARED_PAD_SRC_K"] = src_k
    old = {k: os.environ.get(k) for k in (*env, "DSV41_SHARED_PAD_SRC_K")}
    os.environ.pop("DSV41_SHARED_PAD_SRC_K", None)
    os.environ.update(env)
    sys.modules.setdefault("torch", unittest.mock.MagicMock())
    try:
        spec = importlib.util.spec_from_file_location("shared_pad_k_t", ROOT / "adapter" / "shared_pad_k.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.ENABLED = mod._enabled()
        return mod
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SharedPadK(unittest.TestCase):
    def test_tp4_unchanged(self):
        m = load(4)
        self.assertEqual((m.SRC_K, m.TGT_K), (576, 640))
        self.assertTrue(m.ENABLED)

    def test_tp8(self):
        m = load(8)
        self.assertEqual((m.SRC_K, m.TGT_K), (288, 384))
        self.assertTrue(m.ENABLED)

    def test_tp3_is_noop(self):
        m = load(3)
        self.assertEqual((m.SRC_K, m.TGT_K), (768, 768))
        self.assertFalse(m.ENABLED)

    def test_flag_off(self):
        self.assertFalse(load(4, flag="0").ENABLED)


if __name__ == "__main__":
    unittest.main()
