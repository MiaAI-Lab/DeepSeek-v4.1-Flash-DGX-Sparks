#!/usr/bin/env python3
"""scripts/remote.py must honour --env-file for its own argument defaults.

No SSH or network: pexpect is stubbed and ssh_cmd is replaced, so only the argument
defaults that main() builds from the environment are checked.
"""
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("SSH_IDENTITY", "WORKER_HOST", "WORKER_USER", "REMOTE_TIMEOUT", "WORKER_PASS")


def load_remote():
    sys.modules.setdefault("pexpect", types.ModuleType("pexpect"))
    spec = importlib.util.spec_from_file_location("remote_under_test", ROOT / "scripts" / "remote.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RemoteEnvFile(unittest.TestCase):
    def run_main(self, argv_style):
        saved = {k: os.environ.pop(k, None) for k in KEYS}
        try:
            with tempfile.TemporaryDirectory() as d:
                env = Path(d) / ".env"
                env.write_text('SSH_IDENTITY=$HOME/.ssh/id_test\nWORKER_HOST=10.9.9.9\n'
                               'WORKER_USER="alice"\nREMOTE_TIMEOUT=42\n')
                mod = load_remote()
                seen = {}
                mod.ssh_cmd = lambda host, user, remote, pw, ident, to: seen.update(
                    host=host, user=user, ident=ident, timeout=to) or 0
                argv = ["--env-file", str(env)] if argv_style == "sep" else [f"--env-file={env}"]
                old = sys.argv
                sys.argv = ["remote.py", *argv, "true"]
                try:
                    self.assertEqual(mod.main(), 0)
                finally:
                    sys.argv = old
                return seen
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def check(self, style):
        seen = self.run_main(style)
        self.assertEqual(seen["host"], "10.9.9.9")
        self.assertEqual(seen["user"], "alice")
        self.assertEqual(seen["timeout"], 42)
        self.assertNotIn("$", seen["ident"])
        self.assertTrue(seen["ident"].endswith("/.ssh/id_test"))

    def test_separate_value(self):
        self.check("sep")

    def test_equals_form(self):
        self.check("eq")


if __name__ == "__main__":
    unittest.main()
