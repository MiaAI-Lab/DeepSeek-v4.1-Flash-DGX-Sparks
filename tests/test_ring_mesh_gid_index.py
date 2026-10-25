#!/usr/bin/env python3
"""scripts/ring_mesh/plan.py checks and plans the RoCEv2 GID index it is given.

The IPv4-mapped GID is not always at index 3 (a NetworkManager link-local puts it at 5, a GID
drift at 4), and the fleet then sets NCCL_IB_GID_INDEX. plan.py must accept the same index
instead of refusing every port. CPU only: no SSH, sparkring or b12x.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ring_mesh_plan", ROOT / "scripts/ring_mesh/plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)

IPV4 = {"enp1s0f0np0": "192.168.100.1/24", "enp1s0f1np1": "192.168.102.1/24",
        "enP2p1s0f0np0": "192.168.101.1/24", "enP2p1s0f1np1": "192.168.103.1/24"}


def port(nd, ip, gid):
    return {"rdma": "x", "ipv4": ip, "mac": "4c:bb:47:00:00:01", "mtu": 9000, "state": "up",
            "gid": gid, "gid_netdev": nd, "tc_offload": "on", "hairpin_num_queues": "4",
            "hairpin_queue_size": "1024", "flow_steering_mode": "hmfs"}


def inventory(ipv4_gid=True):
    """One host as inventory.sh reports it for the GID index plan.py asked about."""
    ports = {}
    for nd, ip in IPV4.items():
        gid = ("0000:0000:0000:0000:0000:ffff:" + plan._gid_tail(ip) if ipv4_gid
               else "fe80:0000:0000:0000:fb6a:7b89:81df:639c")
        ports[nd] = port(nd, ip, gid)
    return {"host": "s1", "ports": ports}


class GidIndex(unittest.TestCase):
    def test_ipv4_gid_at_the_given_index_passes(self):
        self.assertEqual(plan.check_host("s1", inventory(), 5), [])

    def test_link_local_at_the_given_index_is_named(self):
        bad = plan.check_host("s1", inventory(ipv4_gid=False), 3)
        self.assertEqual(len(bad), 4)
        self.assertTrue(all("RoCEv2 GID index 3 does not map" in b for b in bad), bad)

    def test_topology_records_the_index(self):
        invs = {h: inventory() for h in ("s1", "s2", "s3", "s4")}
        topo = plan.topology(invs, ["s1", "s2", "s3", "s4"], 5)
        self.assertEqual(topo["roce_gid_index"], 5)

    def test_cli_default_is_three(self):
        src = (ROOT / "scripts/ring_mesh/plan.py").read_text()
        self.assertTrue('"--gid-index", type=int, default=3' in src, "plan.py has no --gid-index defaulting to 3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
