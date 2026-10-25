#!/usr/bin/env python3
"""build and doctor check that scripts/remote.py can import pexpect.

Every worker step goes through scripts/remote.py, which imports pexpect. Without it, build used
to fail on the first worker only after the head image was built. CPU only: python3 is replaced
by a stub on PATH; no Docker, SSH or network.
"""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "start.sh").read_text()


def function(name):
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", SOURCE, re.S | re.M)
    assert m, f"{name} not found in start.sh"
    return m.group(0)


def hint():
    m = re.search(r'^REMOTE_PY_HINT=.*$', SOURCE, re.M)
    assert m, "REMOTE_PY_HINT not found in start.sh"
    return m.group(0)


def run_build(pexpect_ok):
    with tempfile.TemporaryDirectory() as d:
        stub = Path(d) / "python3"
        stub.write_text(f"#!/bin/sh\nexit {0 if pexpect_ok else 1}\n")
        stub.chmod(0o755)
        log = Path(d) / "log"
        script = f'''
set -u
info(){{ :; }}; warn(){{ :; }}; die(){{ echo "DIE $*"; exit 9; }}
docker(){{ echo docker "$@" >> "$LOG"; }}
IMAGE=i BASE_IMAGE=b ROOT=/r BUILD_DOCKERFILE=Dockerfile
{hint()}
{function("remote_py_ready")}
{function("cmd_build")}
cmd_build
'''
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           env={**os.environ, "PATH": f"{d}:{os.environ['PATH']}", "LOG": str(log)})
        return r, log.read_text() if log.exists() else ""


class RemotePyPreflight(unittest.TestCase):
    def test_build_stops_before_docker_without_pexpect(self):
        r, log = run_build(pexpect_ok=False)
        self.assertEqual(r.returncode, 9, r.stdout + r.stderr)
        self.assertIn("DIE python3 cannot import pexpect", r.stdout)
        self.assertEqual(log, "")

    def test_build_proceeds_with_pexpect(self):
        r, log = run_build(pexpect_ok=True)
        self.assertNotIn("pexpect", r.stdout)
        self.assertIn("docker", log)

    def test_doctor_checks_pexpect(self):
        self.assertIn("remote_py_ready", function("cmd_doctor"))

    def test_remote_py_still_imports_pexpect(self):
        # if remote.py stops needing pexpect, drop the check
        self.assertIn("import pexpect", (ROOT / "scripts/remote.py").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
