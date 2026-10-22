#!/usr/bin/env python3
"""mesh-up.sh must not read an existing rule as missing (scripts/ring_mesh/plan.py).

mesh-up.sh runs under `set -euo pipefail` and checks each qdisc, filter and route with
`<show> | grep ...` before adding it. With `grep -q`, grep exits at the first match while the
show command is still writing; the writer then dies of SIGPIPE (141), pipefail turns the whole
check into a failure, and the `||` branch re-adds a rule that exists ("Filter already exists"),
which aborts mesh-up under `set -e`. The test generates mesh-up.sh for a small plan and runs it
against stub `tc`, `ip`, `systemctl` and `sleep` whose show output puts the match first and then
more than a pipe buffer of further lines, with everything already present: the script must exit 0
and never call an add.

No GPU, torch or network. usage: python3 tests/test_ring_mesh_pipefail.py
"""
import importlib.util
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ring_mesh_plan", ROOT / "scripts/ring_mesh/plan.py")
plan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_module)

# show output: the matching line first, then ~3 MB more, far past the 64 KiB pipe buffer
FLOOD = "head -c 3000000 /dev/zero | tr '\\0' 'x'; echo"
TC = f"""#!/bin/bash
case "$*" in
  *"qdisc show"*) echo "qdisc clsact ffff: parent ffff:fff1"; {FLOOD} ;;
  *"-s filter show"*) echo "filter protocol 0x88b5 pref 10 flower chain 0 handle 0x1 in_hw in_hw_count 1"; {FLOOD} ;;
  *"filter show"*) echo "filter protocol 0x88b5 pref 10 flower chain 0 handle 0x1"; {FLOOD} ;;
  *add*) echo "ADD tc $*" >> "$STUB_LOG"; echo "Error: Filter already exists." >&2; exit 2 ;;
esac
"""
IP = f"""#!/bin/bash
case "$*" in
  *"route show exact"*) echo "10.30.0.2 via 10.10.0.11 dev enp1s0f0np0"; {FLOOD} ;;
  *add*) echo "ADD ip $*" >> "$STUB_LOG"; echo "RTNETLINK answers: File exists" >&2; exit 2 ;;
esac
"""
SYSTEMCTL = "#!/bin/bash\nexit 0\n"
SLEEP = "#!/bin/bash\nexit 0\n"


class Fabric:
    @staticmethod
    def tc_rule_command(rule, add=True):
        return ["tc", "filter", "add" if add else "del", "dev", rule.ingress_netdev, "ingress", "pref",
                str(rule.preference), "protocol", "0x88b5", "handle", rule.handle, "flower"]

    @staticmethod
    def route_command(route, add=True):
        return ["ip", "route", "add" if add else "del", f"{route.destination_ipv4}/32", "via", "10.10.0.11"]


PLAN = SimpleNamespace(
    tc_rules=[SimpleNamespace(intermediate_rank=0, ingress_netdev="enp1s0f0np0", preference=10, handle="0x1"),
              SimpleNamespace(intermediate_rank=0, ingress_netdev="enp1s0f1np1", preference=11, handle="0x2")],
    routes=[SimpleNamespace(source_rank=0, destination_ipv4="10.30.0.2")],
    markers=[SimpleNamespace(source_rank=0, rdma_device="rocep1s0f0", udp_source_port=65535)],
)


class MeshUpPipefail(unittest.TestCase):
    def test_existing_rules_are_not_re_added(self):
        up, _down = plan_module.host_scripts(PLAN, Fabric, 0)
        self.assertIn("set -euo pipefail", up)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, body in (("tc", TC), ("ip", IP), ("systemctl", SYSTEMCTL), ("sleep", SLEEP)):
                f = d / name
                f.write_text(body)
                f.chmod(f.stat().st_mode | stat.S_IEXEC)
            script = d / "mesh-up.sh"
            script.write_text(up)
            log = d / "adds.log"
            env = dict(os.environ, PATH=f"{d}:{os.environ['PATH']}", STUB_LOG=str(log))
            r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)
            adds = log.read_text() if log.exists() else ""
        self.assertEqual(adds, "", f"re-added an existing rule:\n{adds}{r.stderr[-400:]}")
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("mesh-up OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
