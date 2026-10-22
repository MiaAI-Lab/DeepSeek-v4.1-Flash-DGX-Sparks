#!/usr/bin/env bash
# Set the mlx5 hairpin queue size (the forwarding buffer of the opposite-node paths) on the four
# fabric functions, one at a time, and wait for each to come back. Functions already at the value
# are skipped. hairpin_queue_size is a driverinit parameter: it resets at boot, and applying it
# re-initialises the function (~4 s), which drops every RDMA connection on it. Never run it with a
# new value while the engine is up. usage: hairpin.sh [PACKETS]   (default 8192; driver default 1024)
set -euo pipefail
V=${1:-8192}
for x in 0000:01:00.0:enp1s0f0np0:rocep1s0f0 0000:01:00.1:enp1s0f1np1:rocep1s0f1 \
         0002:01:00.0:enP2p1s0f0np0:roceP2p1s0f0 0002:01:00.1:enP2p1s0f1np1:roceP2p1s0f1; do
  B=pci/${x%%:en*}; r=${x#*:*:*:}; N=${r%%:*}; D=${r##*:}
  cur=$(devlink dev param show "$B" name hairpin_queue_size | sed -n 's/.*cmode driverinit value \([0-9]*\).*/\1/p')
  if [[ "$cur" == "$V" ]]; then echo "  $(hostname) $N: hairpin_queue_size already $V"; continue; fi
  devlink dev param set "$B" name hairpin_queue_size value "$V" cmode driverinit
  devlink dev reload "$B" action driver_reinit >/dev/null
  for i in $(seq 1 30); do
    [[ "$(cat /sys/class/net/$N/operstate 2>/dev/null)" == up ]] && ip -4 addr show "$N" | grep inet >/dev/null \
      && [[ "$(cat /sys/class/net/$N/mtu)" == 9000 ]] && grep -q "4: ACTIVE" "/sys/class/infiniband/$D/ports/1/state" 2>/dev/null \
      && [[ "$(cat /sys/class/infiniband/$D/ports/1/gids/3)" != 0000:0000:0000:0000:0000:0000:0000:0000 ]] && break
    sleep 2
  done
  [[ $i -lt 30 ]] || { echo "  $(hostname) $N: did not come back after re-init" >&2; exit 2; }
  echo "  $(hostname) $N: hairpin_queue_size $cur -> $V, back after $((i * 2)) s"
done
