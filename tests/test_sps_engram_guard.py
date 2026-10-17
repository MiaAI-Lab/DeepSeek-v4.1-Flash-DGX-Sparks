#!/usr/bin/env python3
"""A profiled SPS table must not reach the engine on the Engram model by default."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import boot  # noqa: E402  (module level only reads environment variables)


class SpsEngramGuardTests(unittest.TestCase):
    def setUp(self):
        fd, self.table = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        self.addCleanup(os.unlink, self.table)

    def call(self, table, env):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return boot.sps_table_args(table, env), out.getvalue()

    def test_existing_table_is_ignored_by_default(self):
        args, msg = self.call(self.table, {})
        self.assertEqual(args, [])
        self.assertIn('Engram', msg)

    def test_override_restores_table_flags(self):
        args, _ = self.call(self.table, {'DSPARK_SPS_ALLOW_ENGRAM': '1'})
        self.assertEqual(args[:2], ['--speculative-dspark-sps-table-path', self.table])
        self.assertIn('--speculative-dspark-align-verify-tokens-to-graph-tier', args)

    def test_missing_or_unset_table(self):
        self.assertEqual(self.call('', {})[0], [])
        args, msg = self.call('/nonexistent/sps.json', {'DSPARK_SPS_ALLOW_ENGRAM': '1'})
        self.assertEqual(args, [])
        self.assertIn('not found', msg)


if __name__ == '__main__':
    unittest.main()
