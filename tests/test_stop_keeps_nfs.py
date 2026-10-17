#!/usr/bin/env python3
"""stop.sh's head sweep ("anything else this recipe named") must not remove the NFS exporter.

The sweep removes every container whose name contains "dsv41-", which also matched dsv41-nfs:
every stop force-killed the exporter while the workers still had the volume mounted, it took
minutes to die in the kernel, and a serve started meanwhile reused the dying exporter and hung
on the workers' hard NFS mounts. The test runs the sweep block from stop.sh against a stub
`docker` and checks which containers it would remove.
"""
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINERS = {'id-nfs': 'dsv41-nfs', 'id-probe': 'dsv41-share-probe', 'id-head': 'dsv41-head'}

STUB = r'''#!/bin/bash
# docker stub: names from $STUB_NAMES ("id name;id name"), rm -f args appended to $STUB_LOG
if [[ "$1" == "ps" ]]; then
  fmt=""; prev=""
  for a in "$@"; do [[ "$prev" == "--format" ]] && fmt="$a"; prev="$a"; done
  IFS=';' read -ra rows <<< "$STUB_NAMES"
  for r in "${rows[@]}"; do
    id=${r%% *}; name=${r#* }
    [[ "$name" == *dsv41-* ]] || continue
    if [[ -z "$fmt" ]]; then echo "$id"
    else out=${fmt//'{{.ID}}'/$id}; out=${out//'{{.Names}}'/$name}; echo -e "$out"; fi
  done
elif [[ "$1" == "rm" ]]; then
  shift; [[ "$1" == "-f" ]] && shift; echo "$@" >> "$STUB_LOG"
fi
'''


def sweep_block():
    text = (ROOT / 'stop.sh').read_text()
    m = re.search(r'# anything else this recipe named[^\n]*\n(.*?\nfi\n)', text, re.S)
    if not m:
        sys.exit('FAIL: head sweep block not found in stop.sh')
    return m.group(1)


def main():
    with tempfile.TemporaryDirectory() as d:
        stub = Path(d) / 'docker'
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        log = Path(d) / 'rm.log'
        env = dict(os.environ, PATH=f'{d}:{os.environ["PATH"]}', STUB_LOG=str(log), RM_TIMEOUT='10',
                   STUB_NAMES=';'.join(f'{i} {n}' for i, n in CONTAINERS.items()))
        r = subprocess.run(['bash', '-c', sweep_block()], env=env, capture_output=True, text=True)
        removed = set(log.read_text().split()) if log.exists() else set()
    checks = {
        'sweep runs': r.returncode == 0,
        'NFS exporter (dsv41-nfs) kept': 'id-nfs' not in removed,
        'other recipe containers still removed': {'id-probe', 'id-head'} <= removed,
    }
    for name, ok in checks.items():
        print(('ok   ' if ok else 'FAIL ') + name)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
