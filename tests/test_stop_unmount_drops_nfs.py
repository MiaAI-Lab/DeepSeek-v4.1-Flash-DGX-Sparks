#!/usr/bin/env python3
"""stop.sh --unmount must drop the shared NFSv4 exporter (plain ./stop.sh must keep it).

The exporter is kept by default because Qwen/GLM recipes reuse it, which is why it is excluded
from the head sweep. --unmount means "leave nothing of this recipe behind", so it removes the
exporter too — after the worker volumes, so no client holds the export when it goes — and the
end-of-stop report must neither claim it was kept nor hide it when it refuses to die. The test
runs the real stop.sh against stub docker/timeout/remote and inspects the removals and output.
"""
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINERS = ['id-nfs dsv41-nfs', 'id-probe dsv41-share-probe', 'id-head dsv41-head']

DOCKER = r'''#!/bin/bash
# docker stub, backed by $STUB_NAMES ("id name;id name"):
#   ps            print the names not yet in $STUB_STATE (ids, names and -a are honoured)
#   rm -f A B     record each A/B as a removal — a target in $STUB_FAIL (by id or name) exits 1
#                 instead, like a container that will not die.
targets() {   # resolve args to names
  IFS=';' read -ra rows <<< "$STUB_NAMES"
  for a in "$@"; do
    for r in "${rows[@]}"; do
      [[ "$a" == "${r%% *}" || "$a" == "${r#* }" ]] && { echo "${r#* }"; break; }
    done
  done
}
if [[ "$1" == "ps" ]]; then
  fmt=""; prev=""
  for a in "$@"; do [[ "$prev" == "--format" ]] && fmt="$a"; prev="$a"; done
  IFS=';' read -ra rows <<< "$STUB_NAMES"
  for r in "${rows[@]}"; do
    id=${r%% *}; name=${r#* }
    grep -qx "$name" "$STUB_STATE" 2>/dev/null && continue
    [[ "$name" == *dsv41-* ]] || continue
    if [[ -z "$fmt" ]]; then echo "$id"
    else out=${fmt//'{{.ID}}'/$id}; out=${out//'{{.Names}}'/$name}; echo -e "$out"; fi
  done
elif [[ "$1" == "rm" ]]; then
  shift; [[ "$1" == "-f" ]] && shift
  while read -r name; do
    [[ -n "$name" ]] || continue
    [[ "$name" == "$STUB_FAIL" ]] && exit 1
    echo "$name" >> "$STUB_RM_LOG"; echo "$name" >> "$STUB_STATE"
  done < <(targets "$@")
fi
'''

TIMEOUT = r'''#!/bin/bash
shift   # drop the duration; the stub work is instantaneous
exec "$@"
'''

REMOTE = r'''#!/usr/bin/env python3
import sys
print('STOPPED_' + sys.argv[sys.argv.index('--host') + 1])
'''


def run_stop(unmount, fail_name=''):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'bin').mkdir()
        for name, body in (('docker', DOCKER), ('timeout', TIMEOUT), ('remote.py', REMOTE)):
            f = d / 'bin' / name
            f.write_text(body)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        (d / 'rm.log').write_text('')
        (d / 'state').write_text('')
        (d / 'empty.env').write_text('')
        env = dict(os.environ,
                   PATH=f'{d}/bin:{os.environ["PATH"]}',
                   ENV_FILE=str(d / 'empty.env'),
                   REMOTE_PY=str(d / 'bin' / 'remote.py'),
                   LOG_DIR=str(d),
                   STUB_NAMES=';'.join(CONTAINERS),
                   STUB_STATE=str(d / 'state'),
                   STUB_RM_LOG=str(d / 'rm.log'),
                   STUB_FAIL=fail_name,
                   RM_TIMEOUT='10')
        args = ['bash', str(ROOT / 'stop.sh')] + (['--unmount'] if unmount else [])
        r = subprocess.run(args, env=env, capture_output=True, text=True, cwd=str(ROOT))
        removed = [ln for ln in (d / 'rm.log').read_text().split() if ln]
    return r, removed


def check(name, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail and not ok else ''))
    return ok


def main():
    default, default_removed = run_stop(unmount=False)
    unmount, unmount_removed = run_stop(unmount=True)
    stubborn, stubborn_removed = run_stop(unmount=True, fail_name='dsv41-nfs')
    results = [
        check('plain stop.sh runs', default.returncode == 0, default.stderr[-300:]),
        check('plain stop.sh keeps the NFS exporter', 'dsv41-nfs' not in default_removed,
              ','.join(default_removed)),
        check('plain stop.sh still sweeps other recipe containers',
              {'dsv41-head', 'dsv41-share-probe'} <= set(default_removed), ','.join(default_removed)),
        check('plain stop.sh reports the exporter as kept',
              'dsv41-nfs exporter' in default.stdout, default.stdout),
        check('plain stop.sh does not warn about the kept exporter',
              'still running on head: dsv41-nfs' not in default.stdout, default.stdout),
        check('--unmount runs', unmount.returncode == 0, unmount.stderr[-300:]),
        check('--unmount removes the NFS exporter', 'dsv41-nfs' in unmount_removed,
              ','.join(unmount_removed)),
        check('--unmount still sweeps the recipe containers',
              {'dsv41-head', 'dsv41-share-probe'} <= set(unmount_removed), ','.join(unmount_removed)),
        check('--unmount reports the removal',
              'remove NFS exporter dsv41-nfs' in unmount.stdout, unmount.stdout),
        check('--unmount no longer claims the exporter was kept',
              'dsv41-nfs exporter' not in unmount.stdout, unmount.stdout),
        check('--unmount is quiet about the exporter once it is gone',
              'still running on head: dsv41-nfs' not in unmount.stdout, unmount.stdout),
        check('an exporter that refuses to die is reported, not hidden',
              'still running on head: dsv41-nfs' in stubborn.stdout,
              f'removed={stubborn_removed} out={stubborn.stdout}'),
    ]
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
