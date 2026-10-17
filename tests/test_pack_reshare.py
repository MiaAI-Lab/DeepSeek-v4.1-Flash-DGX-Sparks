#!/usr/bin/env python3
"""cmd_pack must re-share the checkpoint when a worker cannot see it.

`stop` can take the NFS exporter down; pack then died on the first worker after the head had
already spent ~10 minutes. The function is extracted from start.sh and run with stubbed helpers:
no Docker, SSH or network.
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pack_source():
    text = (ROOT / "start.sh").read_text()
    m = re.search(r"^cmd_pack\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert m, "cmd_pack not found in start.sh"
    return m.group(0)


def run(blind_hosts):
    script = f'''
set -u
info(){{ :; }}; die(){{ echo "DIE $*"; exit 9; }}
model_src(){{ echo /tmp; }}
docker(){{ echo docker-head >> "$LOG"; }}
remote_on(){{ echo "remote $1" >> "$LOG"; }}
nfs_worker_has_model(){{ case " {blind_hosts} " in *" $1 "*) return 1;; esac; return 0; }}
cmd_share(){{ echo share >> "$LOG"; }}
WORKER_HOSTS=(w1 w2); WORKER_IPS=(w1 w2); ENGRAM_DIR=$TMPD/e; WORKER_ENGRAM_DIR=/x
NFS_VOLUME=v; IMAGE=i; TP_SIZE=3
mkdir -p "$ENGRAM_DIR"
{pack_source()}
cmd_pack
'''
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log"
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                           env={**os.environ, "LOG": str(log), "TMPD": d})
        lines = log.read_text().split("\n") if log.exists() else []
    return r, [l for l in lines if l]


class PackReshare(unittest.TestCase):
    def test_healthy_export_does_not_reshare(self):
        r, log = run("")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("share", log)

    def test_blind_worker_reshares_before_packing(self):
        r, log = run("w2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(log.count("share"), 1)
        self.assertLess(log.index("share"), log.index("docker-head"))

    def test_all_blind_reshares_once(self):
        r, log = run("w1 w2")
        self.assertEqual(log.count("share"), 1)


if __name__ == "__main__":
    unittest.main()
