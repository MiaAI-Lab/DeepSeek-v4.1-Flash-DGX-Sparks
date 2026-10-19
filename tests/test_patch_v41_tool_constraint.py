#!/usr/bin/env python3
"""runtime/patch_v41_tool_constraint.py without the image: idempotence and refusal.

The stock dev-dsv41 file is not vendored here; the Dockerfile build applies the patch to the
real file and then runs tests/test_v41_tool_constraint.py against SGLang and xgrammar.
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_v41_tool_constraint.py'
BACKPORT = ROOT / 'runtime' / 'sglang' / 'deepseekv41_detector.py'


def run(path):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(path)], capture_output=True, text=True)


def main():
    src = PATCHER.read_text()
    digest = hashlib.sha256(BACKPORT.read_bytes()).hexdigest()
    assert f'BACKPORT_SHA256 = "{digest}"' in src, 'vendored backport and pinned hash disagree'
    with tempfile.TemporaryDirectory() as d:
        done = Path(d) / 'applied.py'
        shutil.copyfile(BACKPORT, done)
        r = run(done)
        assert r.returncode == 0 and 'already applied' in r.stdout, r
        other = Path(d) / 'other.py'
        other.write_text('class DeepSeekV41Detector: pass\n')
        r = run(other)
        assert r.returncode == 1 and 'neither the stock' in r.stderr, r
        assert other.read_text() == 'class DeepSeekV41Detector: pass\n', 'refused file was modified'
    print('PASS test_patch_v41_tool_constraint')


if __name__ == '__main__':
    main()
