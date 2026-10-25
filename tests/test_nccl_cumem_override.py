#!/usr/bin/env python3
"""NCCL_CUMEM_ENABLE reaches every rank from .env, defaulting to 0.

With cuMem on, stock NCCL connects only the ring on a switchless ring (docs/switchless-ring.md,
"Stock NCCL with cuMem on"); start.sh used to hard-code 0 on the head and the workers.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CumemOverride(unittest.TestCase):
    def test_head_and_workers_take_the_setting(self):
        src = (ROOT / "start.sh").read_text()
        lines = re.findall(r'-e "NCCL_CUMEM_ENABLE=[^"]*"', src)
        self.assertEqual(lines, ['-e "NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}"'] * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
