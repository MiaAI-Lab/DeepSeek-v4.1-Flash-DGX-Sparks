#!/usr/bin/env python3
"""Plan the hardware-forwarded opposite-node paths that let RoCEnante run on a switchless ring.

    python3 scripts/ring_mesh/plan.py --sparkring ~/sparkring --out ring-mesh HEAD W1 W2 W3

HEAD W1 W2 W3 are SSH names of the four Sparks in tensor-parallel rank order (the head, then
WORKER_HOSTS). Each is inventoried over SSH with inventory.sh; the cabling is read from the
fabric subnets (each DAC is its own /24), the ring is numbered the way sparkring's planner
requires (physical port f0 clockwise), and FujitsuPolycom/sparkring's own planner
(spark_transport/fabric/cx7_hairpin_diagonal/fabric.py, tested at f16b5f4) builds the RoCEnante
selection: per rank two /32 routes to the opposite node via a neighbour, two hardware-only tc
redirect rules for the traffic it forwards, and two source markers. Writes into --out:

    fabric.json                sparkring topology document (the plan's input)
    mesh-up-<host>.sh          idempotent apply for that host (routes, clsact, rules, markers)
    mesh-down-<host>.sh        removes exactly what mesh-up created
    env.txt                    the EXTRA_CONTAINER_ENV additions, with the per-rank peer maps
                               translated from sparkring's ring numbering to the TP rank order

Nothing on the hosts is changed. See docs/switchless-ring.md for installation.
"""
import argparse
import ipaddress
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# sparkring's source-bound map: physical port f0 is the clockwise direction, function 0 is the
# first PCIe domain (rocep1s0f*) and function 1 the second (roceP2p1s0f*).
PORTS = {
    ("clockwise", 0): "enp1s0f0np0",
    ("clockwise", 1): "enP2p1s0f0np0",
    ("counter_clockwise", 0): "enp1s0f1np1",
    ("counter_clockwise", 1): "enP2p1s0f1np1",
}
HCAS = ("rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1")  # B12X_ROCE_HCA order


def inventory(host):
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash -s"], input=(HERE / "inventory.sh").read_bytes(),
                         capture_output=True, timeout=60, check=True).stdout
    return json.loads(out.decode())


def check_host(host, inv):
    """Problems that stop the mesh from working on this host (empty list when fine)."""
    bad = []
    for nd, p in inv["ports"].items():
        if p["state"] != "up" or not p["ipv4"]:
            bad.append(f"{host} {nd}: link {p['state']} ip {p['ipv4'] or '-'}")
        if p["mtu"] != 9000:
            bad.append(f"{host} {nd}: MTU {p['mtu']} (9000 expected)")
        if p["gid3_netdev"] != nd or not p["gid3"].endswith(_gid_tail(p["ipv4"])):
            bad.append(f"{host} {nd}: RoCEv2 GID index 3 does not map this port's IPv4")
        if p["tc_offload"] != "on":
            bad.append(f"{host} {nd}: hw-tc-offload {p['tc_offload']}")
        if p["flow_steering_mode"] != "hmfs":
            bad.append(f"{host} {nd}: flow_steering_mode {p['flow_steering_mode']} (sparkring tested hmfs)")
    return bad


def _gid_tail(cidr):
    if not cidr:
        return "-"
    a, b, c, d = str(ipaddress.ip_interface(cidr).ip).split(".")
    return f"{int(a):02x}{int(b):02x}:{int(c):02x}{int(d):02x}"


def ring_order(invs, hosts):
    """Hosts in sparkring ring order, starting at the TP rank 0 host: each f0 port reaches the next host's f1."""
    net = {}
    for h in hosts:
        for nd, p in invs[h]["ports"].items():
            net.setdefault(ipaddress.ip_interface(p["ipv4"]).network, []).append((h, nd))
    peer = {}
    for ends in net.values():
        if len(ends) != 2:
            raise SystemExit(f"fabric subnet shared by {len(ends)} ports, expected one DAC per /24: {ends}")
        (h1, n1), (h2, n2) = ends
        peer[(h1, n1)], peer[(h2, n2)] = (h2, n2), (h1, n1)
    order = [hosts[0]]
    while True:
        cur = order[-1]
        nxt = []
        for f in (0, 1):
            h, nd = peer.get((cur, PORTS[("clockwise", f)]), (None, None))
            if nd != PORTS[("counter_clockwise", f)]:
                raise SystemExit(f"{cur} {PORTS[('clockwise', f)]} reaches {h} {nd}; sparkring needs every f0 port "
                                 f"cabled to the next node's f1 port on the same PCIe domain")
            nxt.append(h)
        if nxt[0] != nxt[1]:
            raise SystemExit(f"{cur}: its two f0 functions reach different nodes {nxt}")
        if nxt[0] == hosts[0]:
            break
        order.append(nxt[0])
    if sorted(order) != sorted(hosts) or len(order) != 4:
        raise SystemExit(f"not a four-node ring: {order}")
    return order


def topology(invs, order, state_root="/run/dsv41-mesh"):
    ranks = []
    for r, h in enumerate(order):
        ports = {}
        for d in ("clockwise", "counter_clockwise"):
            ports[d] = []
            for f in (0, 1):
                nd = PORTS[(d, f)]
                p = invs[h]["ports"][nd]
                ports[d].append({
                    "function": f, "netdev": nd, "rdma_device": p["rdma"],
                    "ipv4_cidr": str(ipaddress.ip_interface(p["ipv4"]).ip) + "/32", "mac": p["mac"],
                    "peer_rank": (r + (1 if d == "clockwise" else -1)) % 4,
                    "peer_direction": "counter_clockwise" if d == "clockwise" else "clockwise", "peer_function": f})
        ranks.append({"rank": r, "ssh_alias": h, "management_netdev": "enP7s7", "ports": ports})
    return {"schema": "sparkring-cx7-hardware-diagonal-fabric/v1", "status": "research-only", "group_id": 77,
            "socket_direct_functions": 2, "flow_label_base": 16383, "shared_diagonal_flow_label": True,
            "endpoint_route_strategy": "adjacent_gateway", "standard_ether_type": 2048, "marked_ether_type": 34997,
            "standard_udp_destination_port": 4791, "roce_gid_index": 3, "bounded_runtime_seconds": 7200,
            "expected_ethernet_mtu": 9000, "expected_roce_mtu": 4096,
            "orchestration": {"host_helper_path": "/UNSUPPORTED/external-fabric-orchestrator",
                              "remote_state_root": state_root},
            "ranks": ranks}


def peer_maps(native_args, order, hosts):
    """B12X_ROCE_PEER_HCA_MAPS for the TP rank order.

    native_args is sparkring's rocenante_native_path_arguments(plan): per ring rank, entries
    "dest,function,rdma_device,gid_index,hops". Path i of a peer is its function-i path, the same
    physical path at both ends, which is what the proxy pairs (local path i with the peer's path i).
    """
    tp_of = {h: hosts.index(h) for h in hosts}
    maps = [None] * 4
    for ring_rank, specs in native_args.items():
        paths = {}
        for spec in specs:
            dest, function, rdma, _gid, _hops = spec.split(",")
            paths.setdefault(tp_of[order[int(dest)]], {})[int(function)] = HCAS.index(rdma)
        maps[tp_of[order[int(ring_rank)]]] = ",".join(f"{p}={v[0]}/{v[1]}" for p, v in sorted(paths.items()))
    return ";".join(maps)


def host_scripts(plan, fabric, host_rank):
    """(mesh-up, mesh-down) for one host from sparkring's RoCEnante plan."""
    up = ["#!/usr/bin/env bash",
          "# Hardware-forwarded opposite-node paths for RoCEnante (generated by scripts/ring_mesh/plan.py). Idempotent.",
          "set -euo pipefail"]
    down = ["#!/usr/bin/env bash", "# Removes exactly what mesh-up.sh creates.", "set -uo pipefail",
            "systemctl stop 'dsv41-mesh-marker@*' 2>/dev/null || true"]
    rules = [x for x in plan.tc_rules if x.intermediate_rank == host_rank]
    qdiscs = sorted({x.ingress_netdev for x in rules})
    for q in qdiscs:
        up.append(f"tc qdisc show dev {q} | grep clsact >/dev/null || tc qdisc add dev {q} clsact")
    for x in rules:
        add = fabric.tc_rule_command(x, add=True)
        up.append(f"tc filter show dev {x.ingress_netdev} ingress pref {x.preference} | grep flower >/dev/null || {shlex.join(add)}")
        up.append(f"tc -s filter show dev {x.ingress_netdev} ingress pref {x.preference} | grep in_hw >/dev/null "
                  f"|| {{ echo 'rule pref {x.preference} on {x.ingress_netdev} is not in hardware'; exit 3; }}")
        down.append(f"tc filter del dev {x.ingress_netdev} ingress pref {x.preference} protocol 0x88b5 "
                    f"handle {x.handle} flower 2>/dev/null || true")
    for q in qdiscs:
        down.append(f'[ -z "$(tc filter show dev {q} ingress; tc filter show dev {q} egress)" ] '
                    f"&& tc qdisc del dev {q} clsact 2>/dev/null || true")
    for r in (x for x in plan.routes if x.source_rank == host_rank):
        up.append(f"ip route show exact {r.destination_ipv4}/32 | grep . >/dev/null || {shlex.join(fabric.route_command(r, add=True))}")
        down.append(f"ip route del {r.destination_ipv4}/32 2>/dev/null || true")
    devs = sorted({m.rdma_device for m in plan.markers if m.source_rank == host_rank})
    ports = sorted({m.udp_source_port for m in plan.markers})
    if ports != [65535]:
        raise SystemExit(f"unexpected marker UDP source ports {ports}; the marker unit assumes 65535")
    up.append(f"for d in {' '.join(devs)}; do systemctl start dsv41-mesh-marker@$d; done")
    up.append(f"sleep 2; for d in {' '.join(devs)}; do systemctl is-active --quiet dsv41-mesh-marker@$d "
              f"|| {{ echo marker $d not active; exit 4; }}; done")
    up.append("echo mesh-up OK")
    down.append("echo mesh-down done")
    return "\n".join(up) + "\n", "\n".join(down) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hosts", nargs=4, help="SSH names in TP rank order (head first)")
    ap.add_argument("--sparkring", required=True, help="FujitsuPolycom/sparkring checkout (tested at f16b5f4)")
    ap.add_argument("--out", default="ring-mesh")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.sparkring) / "spark_transport/fabric/cx7_hairpin_diagonal"))
    import fabric  # sparkring's planner

    invs = {h: inventory(h) for h in a.hosts}
    problems = [p for h in a.hosts for p in check_host(h, invs[h])]
    if problems:
        raise SystemExit("hosts not ready for the mesh:\n  " + "\n  ".join(problems))
    order = ring_order(invs, a.hosts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fabric.json").write_text(json.dumps(topology(invs, order), indent=2) + "\n", newline="\n")
    plan = fabric.build_rocenante_plan(fabric.build_plan(fabric.load_topology(out / "fabric.json")))
    for r, h in enumerate(order):
        up, down = host_scripts(plan, fabric, r)
        (out / f"mesh-up-{h}.sh").write_text(up, newline="\n")
        (out / f"mesh-down-{h}.sh").write_text(down, newline="\n")
    hairpin = min(int(p["hairpin_queue_size"] or 0) for h in a.hosts for p in invs[h]["ports"].values())
    cap = 262144 if hairpin >= 8192 else 81920
    maps = peer_maps(fabric.rocenante_native_path_arguments(plan), order, list(a.hosts))
    env = (f"DSV41_ROCE_RING=1 SGLANG_ROCE_ALLREDUCE=1 SGLANG_ROCE_MAX_SIZE={cap} DSV41_ROCE_GATHER={cap} "
           f"B12X_ROCE_HCA={','.join(HCAS)} B12X_ROCE_PEER_HCA_MAPS={maps} B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0")
    (out / "env.txt").write_text(env + "\n", newline="\n")
    print(f"ring order (sparkring numbering): {' -> '.join(order)} -> {order[0]}")
    print(f"hairpin_queue_size (smallest): {hairpin} -> RoCE size cap {cap} bytes")
    print(f"wrote {out}/fabric.json, mesh-up-*.sh, mesh-down-*.sh, env.txt\nEXTRA_CONTAINER_ENV additions:\n  {env}")


if __name__ == "__main__":
    main()
