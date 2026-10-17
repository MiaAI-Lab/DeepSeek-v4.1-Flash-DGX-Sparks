#!/usr/bin/env bash
# start-tp4-switchless.sh — start-tp4.sh for a switchless 4-Spark ring (no RoCE switch).
#
# Same engine, image and commands as start-tp4.sh, with a profile of its own:
#   .env.tp4-switchless   settings for this profile
#   state-tp4-switchless/ launch record, smoke result, api key, DSpark tables
#   logs-tp4-switchless/  engine log
# On first run the profile is copied from .env.tp4 (else .env.tp4.example) and the ring
# block from docs/switchless-ring.md is appended, so a switched .env.tp4 stays untouched.
# Edit the cluster section (IPs, hosts, interfaces) and NCCL_HOST_DIR before booting.
#
# Prerequisites (docs/switchless-ring.md): a NCCL build with the sparkring switchless-cycle
# patch at NCCL_HOST_DIR, the serving image built on every node, and the checkpoint on
# every node's local disk (NFS_SHARE=0: a ring has no fabric-wide NFS path).
#
# Usage: ./start-tp4-switchless.sh doctor | build | share | pack | serve | stop | status | logs | smoke
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_FILE="$ROOT/.env.tp4-switchless"
export ENV_EXAMPLE="$ROOT/.env.tp4.example"
export STATE_DIR="${STATE_DIR:-$ROOT/state-tp4-switchless}"
export LOG_DIR="${LOG_DIR:-$ROOT/logs-tp4-switchless}"
export SERVE_LOG="${SERVE_LOG:-$LOG_DIR/dsv41.log}"
export DSV41_LAUNCHER=tp4

if [[ ! -f "$ENV_FILE" ]]; then
  base="$ROOT/.env.tp4"; [[ -f "$base" ]] || base="$ENV_EXAMPLE"
  cp "$base" "$ENV_FILE"
  # Later assignments win when start.sh sources the file, so these override any switched values above.
  cat >> "$ENV_FILE" <<'RING'

# ─── switchless ring (appended by start-tp4-switchless.sh) ────────────────────
NCCL_SWITCHLESS_RING_ONLY=1
NCCL_HOST_DIR=$HOME/nccl-2.30.7
NCCL_ALGO=Ring
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
NCCL_P2P_LEVEL=SYS
DSV41_SERIAL_WEIGHT_LOAD=1
NFS_SHARE=0
RING
  echo "[dsv41] wrote $(basename "$ENV_FILE") from $(basename "$base") + ring block — edit IPs and NCCL_HOST_DIR" >&2
fi
exec "$ROOT/start.sh" "$@"
