#!/usr/bin/env bash
# One JSON object describing this node's four ConnectX-7 fabric functions, for plan.py.
# Read-only; no root needed (devlink parameter reads are unprivileged). GID_INDEX picks the
# RoCEv2 GID to report (default 3; NCCL_IB_GID_INDEX on hosts whose IPv4 GID is elsewhere).
set -euo pipefail
g=${GID_INDEX:-3}
first=1
printf '{"host": "%s", "ports": {' "$(hostname)"
for pair in enp1s0f0np0:rocep1s0f0 enp1s0f1np1:rocep1s0f1 enP2p1s0f0np0:roceP2p1s0f0 enP2p1s0f1np1:roceP2p1s0f1; do
  nd=${pair%%:*}; rd=${pair##*:}
  [[ -d /sys/class/net/$nd && -d /sys/class/infiniband/$rd ]] || { echo "missing $nd/$rd" >&2; exit 1; }
  bdf=$(basename "$(readlink -f "/sys/class/infiniband/$rd/device")")
  ip=$(ip -4 -o addr show dev "$nd" | awk '{print $4}' | head -1)
  param() { devlink dev param show "pci/$bdf" name "$1" 2>/dev/null | sed -n 's/.*value \([0-9a-z]*\).*/\1/p' | head -1; }
  [[ $first == 1 ]] || printf ', '; first=0
  printf '"%s": {"rdma": "%s", "ipv4": "%s", "mac": "%s", "mtu": %s, "state": "%s", "gid": "%s", "gid_netdev": "%s", "tc_offload": "%s", "hairpin_num_queues": "%s", "hairpin_queue_size": "%s", "flow_steering_mode": "%s"}' \
    "$nd" "$rd" "$ip" "$(cat /sys/class/net/$nd/address)" "$(cat /sys/class/net/$nd/mtu)" "$(cat /sys/class/net/$nd/operstate)" \
    "$(cat /sys/class/infiniband/$rd/ports/1/gids/$g 2>/dev/null)" "$(cat /sys/class/infiniband/$rd/ports/1/gid_attrs/ndevs/$g 2>/dev/null)" \
    "$(ethtool -k "$nd" 2>/dev/null | awk '/hw-tc-offload/{print $2}')" \
    "$(param hairpin_num_queues)" "$(param hairpin_queue_size)" "$(param flow_steering_mode)"
done
printf '}}\n'
