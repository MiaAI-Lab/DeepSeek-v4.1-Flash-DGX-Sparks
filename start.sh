#!/usr/bin/env bash
# start.sh — DeepSeek-V4.1-Flash on 3× DGX Spark (GB10 / SM121)
#
# Upstream: https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000
#   4× RTX PRO 6000 Blackwell, TP4/EP4, native weights, NVMe Engram, DSpark.
# This wrapper remaps that stack onto the 3-Spark triangle:
#   spark1 10.0.0.1 (mia)  — rank 0, API :8888
#   spark2 10.0.0.2 (zurih)
#   spark3 10.0.0.3 (zurih)
#   TP=3 EP=3, RoCE NCCL, Engram on NVMe. spark2/spark3 read spark1 weights
#   over NFSv4 on ConnectX (vllm-fn-nfs / dsv41-nfs) — no local 476 GiB copy.
#
# Usage:
#   ./start.sh                 doctor → image → share → serve (default)
#   ./start.sh doctor          connectivity + GPU + checkpoint report
#   ./start.sh pull            pull lmsysorg/sglang:dev-dsv41 on all 3 nodes
#   ./start.sh build           docker build the Engram overlay image everywhere
#   ./start.sh download        hf download the pinned V4.1-Flash revision
#   ./start.sh share           export spark1 checkpoint; NFS volumes on spark2/3
#   ./start.sh mount           alias for share (legacy)
#   ./start.sh serve           launch workers then head
#   ./start.sh stop            tear down all 3 ranks
#   ./start.sh status          containers + API
#   ./start.sh logs [N]        tail head boot logs
#   ./start.sh logs worker<N> [lines]   (worker1, worker2, ...)
#   ./start.sh smoke           arithmetic (+ optional tools/vision)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Profiles: start.sh reads .env (3 Sparks, TP3); start-tp4.sh points ENV_FILE at
# .env.tp4 (4 Sparks, TP4) and gives that profile its own state/log dirs.
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
ENV_EXAMPLE="${ENV_EXAMPLE:-$ROOT/.env.example}"
if [[ ! -f "$ENV_FILE" ]]; then
  [[ -f "$ENV_EXAMPLE" ]] || { echo "missing $(basename "$ENV_EXAMPLE")" >&2; exit 1; }
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  echo "[dsv41] wrote $(basename "$ENV_FILE") from $(basename "$ENV_EXAMPLE") — edit IPs if needed"
fi
set -a
# shellcheck disable=SC1091
source "$ENV_FILE"
set +a

HEAD_IP="${HEAD_IP:-10.0.0.1}"
# Workers: WORKER_IPS="10.0.0.2 10.0.0.3 ..." (space or comma separated) and
# optionally WORKER_HOSTS="spark2 spark3 ..." (ssh names); the legacy
# WORKER1_IP/WORKER2_IP/WORKER1_HOST/WORKER2_HOST pairs still work for 3 nodes.
_list() { tr ',' ' ' <<<"$1"; }
if [[ -n "${WORKER_IPS:-}" ]]; then
  read -r -a WORKER_IPS <<<"$(_list "$WORKER_IPS")"
else
  WORKER_IPS=("${WORKER1_IP:-10.0.0.2}" "${WORKER2_IP:-10.0.0.3}")
fi
if [[ -n "${WORKER_HOSTS:-}" ]]; then
  read -r -a WORKER_HOSTS <<<"$(_list "$WORKER_HOSTS")"
else
  WORKER_HOSTS=()
  for _i in "${!WORKER_IPS[@]}"; do
    _v="WORKER$((_i + 1))_HOST"
    WORKER_HOSTS+=("${!_v:-${WORKER_IPS[$_i]}}")
  done
fi
[[ ${#WORKER_HOSTS[@]} -eq ${#WORKER_IPS[@]} ]] || { echo "WORKER_HOSTS and WORKER_IPS differ in length" >&2; exit 1; }
WORKER1_IP="${WORKER_IPS[0]}"; WORKER2_IP="${WORKER_IPS[1]:-}"   # legacy names, still read by files/nfs-share.sh
WORKER_USER="${WORKER_USER:-zurih}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}"
FABRIC_IFACE="${FABRIC_IFACE:-enp1s0f1np1}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
IB_HCA="${IB_HCA:-rocep1s0f0,rocep1s0f1}"
NCCL_NET="${NCCL_NET:-IB}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
NCCL_CONTAINER_DIR="${NCCL_CONTAINER_DIR:-/nccl}"
# Container path of the base image's pip-installed NCCL. A patched host build
# can be overlaid here (NCCL_OVERLAY_PIP) instead of being pushed onto
# LD_LIBRARY_PATH, which leaves two NCCL runtimes visible and makes DeepEP's
# check_nccl_so() abort before NCCL is ever initialised.
NCCL_PIP_SO="${NCCL_PIP_SO:-/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"
# Overlay the patched library; defaults to the switchless-ring switch below,
# so it can also be enabled on its own with NCCL_OVERLAY_PIP=1.
NCCL_OVERLAY_PIP="${NCCL_OVERLAY_PIP:-${NCCL_SWITCHLESS_RING_ONLY:-0}}"

MODEL_DIR="${MODEL_DIR:-$HOME/NewModels/DeepSeek-V4.1-Flash}"
COMMON_MODEL="${COMMON_MODEL:-/var/tmp/DeepSeek-V4.1-Flash}"
HF_REPO="${HF_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
HF_REVISION="${HF_REVISION:-fb2764a5cf321eaa5070ca8f9e892818f477c16d}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-48}"

BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:dev-dsv41}"
IMAGE="${IMAGE:-dsv41-3x-spark:local}"
HEAD_CTN="${HEAD_CTN:-dsv41-head}"
WORKER_CTN="${WORKER_CTN:-dsv41-worker}"
WORKER_DIR="${WORKER_DIR:-/home/${WORKER_USER}/dsv41-3x-spark}"

NNODES="${NNODES:-$(( ${#WORKER_IPS[@]} + 1 ))}"
[[ "$NNODES" -eq $(( ${#WORKER_IPS[@]} + 1 )) ]] || { echo "NNODES=$NNODES but ${#WORKER_IPS[@]} workers are configured" >&2; exit 1; }
TP_SIZE="${TP_SIZE:-3}"
EP_SIZE="${EP_SIZE:-$TP_SIZE}"
DIST_PORT="${DIST_PORT:-20000}"
PORT="${PORT:-8888}"
OFFLOAD_MODE="${OFFLOAD_MODE:-nvme}"
DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-4}"
# Engram misses are serviced by a pool: the GB10 NVMe does ~3.5k IOPS at QD=1 and
# ~112k at QD=64, and the callback blocks the compute stream the whole time.
DSV41_IO_THREADS="${DSV41_IO_THREADS:-96}"
DSV41_RESIDENT_SCALES="${DSV41_RESIDENT_SCALES:-0}"
DSV41_CACHE_WAYS="${DSV41_CACHE_WAYS:-4}"
DSV41_STATS_SECONDS="${DSV41_STATS_SECONDS:-60}"
# DSpark's ragged-verify scheduler is inert without a profiled SPS cost table.
# Build one against a running server with:
#   python -m sglang.benchmark.dspark_sps_profiler
# and drop it at $STATE_DIR/dspark_sps.json; every rank needs its own copy.
DSPARK_SPS_TABLE="${DSPARK_SPS_TABLE:-/state/dspark_sps.json}"
DSPARK_STS_TABLE="${DSPARK_STS_TABLE:-/state/dspark_sts.json}"
# Node-local repacked Engram shards (see scripts/pack_engram.py). ~68 GiB/node.
ENGRAM_DIR="${ENGRAM_DIR:-$HOME/dsv41-engram}"
WORKER_ENGRAM_DIR="${WORKER_ENGRAM_DIR:-$WORKER_DIR/engram}"
DSV41_PACKED_DIR="${DSV41_PACKED_DIR:-/engram}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-409600}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
# Rank 0 also carries the HTTP server, tokenizer and detokenizer (~1.3 GiB the
# workers never pay) and serves the checkpoint over NFS, so it runs out of host
# RAM first -- and on GB10 host RAM is GPU memory. Give it its own budget.
HEAD_MEM_FRACTION_STATIC="${HEAD_MEM_FRACTION_STATIC:-$MEM_FRACTION_STATIC}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-320000}"
SPEC_ALGO="${SPEC_ALGO:-DSPARK}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_QUICK="${SMOKE_QUICK:-1}"
API_KEY="${API_KEY:-}"
STATE_DIR="${STATE_DIR:-$ROOT/state}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
SERVE_LOG="${SERVE_LOG:-$LOG_DIR/dsv41.log}"
REMOTE_PY="$ROOT/scripts/remote.py"
NFS_VOLUME="${NFS_VOLUME:-dsv41-weights}"
NFS_SHARE="${NFS_SHARE:-1}"

_abs() { readlink -f "$1" 2>/dev/null || echo "$1"; }
MODEL_DIR="$(_abs "$MODEL_DIR")"
SSH_IDENTITY="$(_abs "$SSH_IDENTITY")"
NCCL_HOST_DIR="$(_abs "$NCCL_HOST_DIR")"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info() { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YELLOW}[!]${NC} $*"; }
err()  { echo "${RED}[x]${NC} $*" >&2; }
die()  { err "$@"; exit 1; }

mkdir -p "$LOG_DIR" "$STATE_DIR"

# shellcheck source=files/nfs-share.sh
source "$ROOT/files/nfs-share.sh"

remote_on() {
  local host="$1"; shift
  local to_args=()
  if [[ "${1:-}" == "--timeout" ]]; then
    to_args=(--timeout "$2"); shift 2
  elif [[ -n "${REMOTE_TIMEOUT:-}" ]]; then
    to_args=(--timeout "$REMOTE_TIMEOUT")
  fi
  python3 "$REMOTE_PY" --env-file "$ENV_FILE" \
    --host "$host" --user "$WORKER_USER" \
    --identity "$SSH_IDENTITY" \
    "${to_args[@]}" \
    "bash -lc $(printf '%q' "$*")"
}
remote_ok_on() { remote_on "$@" >/dev/null 2>&1; }

ssh_rsync_e() {
  printf 'ssh -i %q -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new' \
    "$SSH_IDENTITY"
}

ensure_ssh_keys() {
  local pub host
  pub=$(cat "${SSH_IDENTITY}.pub" 2>/dev/null || true)
  [[ -n "$pub" ]] || die "Missing ${SSH_IDENTITY}.pub"
  for host in "${WORKER_HOSTS[@]}"; do
    remote_on "$host" "mkdir -p ~/.ssh && chmod 700 ~/.ssh
      touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
      grep -qxF '$pub' ~/.ssh/authorized_keys || echo '$pub' >> ~/.ssh/authorized_keys"
  done
}

model_src() {
  if [[ -L "$COMMON_MODEL" ]]; then
    readlink -f "$COMMON_MODEL"
  elif [[ -d "$COMMON_MODEL" ]]; then
    echo "$COMMON_MODEL"
  else
    echo "$MODEL_DIR"
  fi
}

api_key() {
  # Empty / dummy / off → no --api-key (Spark 1 / Pi probe /v1/models without auth).
  local v="${API_KEY:-}"
  case "${v,,}" in
    ''|none|off|dummy|0) echo "" ;;
    *) echo "$v" ;;
  esac
}

gid_index_local() {
  local ip="$1" hex hca g i
  hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip" | tr . ' '))
  local IFS=','
  for hca in $IB_HCA; do
    hca="${hca// /}"
    [[ -d /sys/class/infiniband/"$hca" ]] || continue
    for g in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      i=${g##*/}
      [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"$i" 2>/dev/null)" == "RoCE v2" ]] || continue
      case "$(cat "$g")" in *ffff:"$hex") echo "$i"; return 0;; esac
    done
  done
  return 1
}

gid_index_remote() {
  local host="$1" ip="$2" hex
  hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip" | tr . ' '))
  remote_on "$host" "for hca in \$(echo $IB_HCA | tr ',' ' '); do
    [ -d /sys/class/infiniband/\$hca ] || continue;
    for g in /sys/class/infiniband/\$hca/ports/1/gids/*; do
      i=\${g##*/};
      [ \"\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] || continue;
      case \$(cat \$g) in *ffff:$hex) echo \$i; exit 0;; esac;
    done;
  done"
}

docker_common_args() {
  local -n _a=$1
  local src hip gid
  src=$(model_src)
  hip="$2"
  gid="$3"
  [[ -d "$src" ]] || die "model mount src missing: $src"
  _a+=(
    --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all
    --shm-size "${SHM_SIZE:-32g}"
    --ulimit "memlock=-1:-1" --ulimit stack=67108864
    --device /dev/infiniband:/dev/infiniband
    -v "$src:/models/DeepSeek-V4.1-Flash"
    -v "$STATE_DIR:/state"
    -v "$HOME/.cache:/root/.cache"
    -e "OFFLOAD_MODE=$OFFLOAD_MODE"
    -e "DSV41_CACHE_GIB=$DSV41_CACHE_GIB"
    -e "DSV41_IO_THREADS=$DSV41_IO_THREADS"
    -e "DSV41_RESIDENT_SCALES=$DSV41_RESIDENT_SCALES"
    -e "DSV41_CACHE_WAYS=$DSV41_CACHE_WAYS"
    -e "DSV41_STATS_SECONDS=$DSV41_STATS_SECONDS"
    -v "$ENGRAM_DIR:/engram"
    -e "DSV41_PACKED_DIR=$DSV41_PACKED_DIR"
    -e "DSPARK_SPS_TABLE=$DSPARK_SPS_TABLE"
    -e "DSPARK_STS_TABLE=$DSPARK_STS_TABLE"
    -e "DSV41_SOURCE=/models/DeepSeek-V4.1-Flash"
    -e "MODEL_PATH=/models/DeepSeek-V4.1-Flash"
    -e "STATE_PATH=/state"
    -e "SERVER_PORT=$PORT"
    -e "NNODES=$NNODES"
    -e "TP_SIZE=$TP_SIZE"
    -e "EP_SIZE=$EP_SIZE"
    -e "DIST_INIT_ADDR=${HEAD_IP}:${DIST_PORT}"
    -e "CONTEXT_LENGTH=$CONTEXT_LENGTH"
    -e "MEM_FRACTION_STATIC=$HEAD_MEM_FRACTION_STATIC"
    -e "MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
    -e "CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE"
    -e "MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS"
    -e "CUDA_GRAPH_MAX_BS_DECODE=$MAX_RUNNING_REQUESTS"
    -e "SPEC_ALGO=$SPEC_ALGO"
    -e "DSPARK_BLOCK_SIZE=$DSPARK_BLOCK_SIZE"
    -e "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
    -e "SKIP_PREPARE=$SKIP_PREPARE"
    -e "SKIP_VERIFY=$SKIP_VERIFY"
    -e "WARMUP=${WARMUP:-1}"
    -e "HOST=0.0.0.0"
    -e "NCCL_NET=$NCCL_NET"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    -e "NCCL_IB_HCA=$IB_HCA"
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    -e "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
    -e "NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
    -e "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}"
    -e "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}"
    -e "NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=$NCCL_DEBUG"
    -e "NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-4194304}"
    -e "NCCL_LL128_BUFFSIZE=${NCCL_LL128_BUFFSIZE:--2}"
    -e "NCCL_PROTO=${NCCL_PROTO:-LL,LL128,Simple}"
    -e "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-32}"
    -e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}"
    -e "DSV41_MXFP8_BACKEND=${DSV41_MXFP8_BACKEND:-b12x}"
    -e "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=${SGLANG_FLASHINFER_MOE_FUSED_FINALIZE:-1}"
    -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
    -e "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    -e "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0"
    -e "DSV41_TP_PAD=${DSV41_TP_PAD:-1}"
    -e "VLLM_HOST_IP=$hip"
    -e "HOST_IP=$hip"
    -e "NCCL_IB_GID_INDEX=$gid"
  )
  if [[ -n "${API_KEY}" ]]; then
    _a+=(-e "API_KEY=$API_KEY")
  fi
  if [[ -n "${EXTRA_SGLANG_ARGS:-}" ]]; then
    _a+=(-e "EXTRA_SGLANG_ARGS=$EXTRA_SGLANG_ARGS")
  fi
  if [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == "1" ]]; then
    _a+=(-e "NCCL_SWITCHLESS_RING_ONLY=1")
    _a+=(-e "NCCL_ALGO=${NCCL_ALGO:-Ring}")
    _a+=(-e "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}")
    _a+=(-e "NCCL_IB_SUBNET_PREFIX_LEN=${NCCL_IB_SUBNET_PREFIX_LEN:-24}")
    _a+=(-e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}")
    _a+=(-e "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}")
  fi
  if [[ -f "$NCCL_HOST_DIR/libnccl.so.2.30.7" || -f "$NCCL_HOST_DIR/libnccl.so.2" ]]; then
    local _nccl_so="libnccl.so.2.30.7"
    [[ -f "$NCCL_HOST_DIR/$_nccl_so" ]] || _nccl_so="libnccl.so.2"
    if [[ "$NCCL_OVERLAY_PIP" == "1" ]]; then
      _a+=(-v "$NCCL_HOST_DIR/$_nccl_so:$NCCL_PIP_SO:ro")
    else
      _a+=(-v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro" -e "LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR")
    fi
  fi
}

# Every rank builds its own planner, so the SPS/STS calibration has to exist on
# every node -- the head's copy in $STATE_DIR is the source of truth.
push_spec_tables() {
  local name local_path host
  for name in dspark_sps.json dspark_sts.json; do
    local_path="$STATE_DIR/$name"
    for host in "${WORKER_IPS[@]}"; do
      if [[ -f "$local_path" ]]; then
        local payload
        payload=$(base64 -w0 <"$local_path")
        # Every rank derives its verify schedule from its own copy of this file.
        # A rank that disagrees with the others builds a different verify shape
        # and the TP collective hangs, so a failed push is fatal, not a warning.
        remote_on "$host" "mkdir -p $WORKER_DIR/state && echo $payload | base64 -d > $WORKER_DIR/state/$name" \
          || die "could not push $name to $host: ranks would disagree on the DSpark verify schedule"
        info "pushed $name to $host"
      else
        # Likewise for a stale copy left from an earlier run.
        remote_on "$host" "rm -f $WORKER_DIR/state/$name" \
          || die "could not clear a stale $name on $host"
      fi
    done
  done
}

# Extra `-e NCCL_...` lines for a switchless ring: a 4-node ring where opposite
# nodes (rank0 <-> rank2) have no direct fabric path, so NCCL's Tree and PAT
# transports cannot be set up. Only a NCCL carrying the sparkring
# switchless-cycle patch understands NCCL_SWITCHLESS_RING_ONLY; stock NCCL
# ignores unknown NCCL_* variables, hence the explicit opt-in.
# See docs/switchless-ring.md.
switchless_ring_env_lines() {
  [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == "1" ]] || return 0
  printf '        -e NCCL_SWITCHLESS_RING_ONLY=1 \\\n'
  printf '        -e NCCL_ALGO=%s \\\n' "${NCCL_ALGO:-Ring}"
  printf '        -e NCCL_SKIP_TREE_CONNECT=%s \\\n' "${NCCL_SKIP_TREE_CONNECT:-1}"
  printf '        -e NCCL_IB_SUBNET_PREFIX_LEN=%s \\\n' "${NCCL_IB_SUBNET_PREFIX_LEN:-24}"
  printf '        -e NCCL_MIN_NCHANNELS=%s \\\n' "${NCCL_MIN_NCHANNELS:-4}"
  printf '        -e NCCL_P2P_LEVEL=%s \\\n' "${NCCL_P2P_LEVEL:-SYS}"
}

worker_env_lines() {
  local wip="$1" wgid="$2" rank="$3"
  cat <<EOF
        -e NODE_RANK=$rank -e NNODES=$NNODES \\
        -e TP_SIZE=$TP_SIZE -e EP_SIZE=$EP_SIZE \\
        -e DIST_INIT_ADDR=$HEAD_IP:$DIST_PORT \\
        -e OFFLOAD_MODE=$OFFLOAD_MODE -e DSV41_CACHE_GIB=$DSV41_CACHE_GIB \\
        -e DSV41_IO_THREADS=$DSV41_IO_THREADS \\
        -e DSV41_RESIDENT_SCALES=$DSV41_RESIDENT_SCALES \\
        -e DSV41_CACHE_WAYS=$DSV41_CACHE_WAYS \\
        -e DSV41_STATS_SECONDS=$DSV41_STATS_SECONDS \\
        -v $WORKER_ENGRAM_DIR:/engram \\
        -e DSV41_PACKED_DIR=$DSV41_PACKED_DIR \\
        -e DSPARK_SPS_TABLE=$DSPARK_SPS_TABLE \\
        -e DSPARK_STS_TABLE=$DSPARK_STS_TABLE \\
        -e DSV41_SOURCE=/models/DeepSeek-V4.1-Flash \\
        -e MODEL_PATH=/models/DeepSeek-V4.1-Flash -e STATE_PATH=/state \\
        -e SERVER_PORT=$PORT -e HOST=0.0.0.0 \\
        -e CONTEXT_LENGTH=$CONTEXT_LENGTH \\
        -e MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC \\
        -e MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS \\
        -e CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE \\
        -e MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS \\
        -e CUDA_GRAPH_MAX_BS_DECODE=$MAX_RUNNING_REQUESTS \\
        -e SPEC_ALGO=$SPEC_ALGO -e DSPARK_BLOCK_SIZE=$DSPARK_BLOCK_SIZE \\
        -e SERVED_MODEL_NAME=$SERVED_MODEL_NAME \\
        -e SKIP_PREPARE=1 -e SKIP_VERIFY=1 -e SKIP_SMOKE=1 \\
        -e NCCL_NET=$NCCL_NET -e NCCL_IB_DISABLE=$NCCL_IB_DISABLE \\
        -e NCCL_IB_HCA=$IB_HCA -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \\
        -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \\
        -e NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -e NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE \\
        -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1} \\
        -e NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0} \\
        -e NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1} \\
        -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=$NCCL_DEBUG \\
        -e NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-4194304} -e NCCL_LL128_BUFFSIZE=${NCCL_LL128_BUFFSIZE:--2} \\
        -e NCCL_PROTO=${NCCL_PROTO:-LL,LL128,Simple} -e NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-32} \\
        -e NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT} \\
$(switchless_ring_env_lines)
        -e DSV41_MXFP8_BACKEND=${DSV41_MXFP8_BACKEND:-b12x} \\
        -e SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=${SGLANG_FLASHINFER_MOE_FUSED_FINALIZE:-1} \\
        -e PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False} \\
        -e NCCL_IB_GID_INDEX=$wgid \\
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0 \\
        -e DSV41_TP_PAD=${DSV41_TP_PAD:-1} \\
        -e HOST_IP=$wip -e VLLM_HOST_IP=$wip \\
EOF
}

cmd_doctor() {
  echo "=== doctor (DeepSeek-V4.1-Flash · ${NNODES}× Spark · TP${TP_SIZE} · SGLang) ==="
  echo "head:     $(hostname) / $(whoami) @ $HEAD_IP"
  echo "workers:  ${WORKER_USER}@${WORKER_HOSTS[*]}  (${WORKER_IPS[*]})"
  echo "image:    $IMAGE  (base $BASE_IMAGE)"
  echo "model:    $MODEL_DIR"
  echo "parallel: nnodes=$NNODES TP=$TP_SIZE EP=$EP_SIZE  ctx=$CONTEXT_LENGTH  util=$MEM_FRACTION_STATIC (head $HEAD_MEM_FRACTION_STATIC)"
  echo "offload:  $OFFLOAD_MODE  cache=${DSV41_CACHE_GIB}GiB  spec=$SPEC_ALGO-$DSPARK_BLOCK_SIZE  port=$PORT"
  echo
  local ok=0
  command -v docker >/dev/null || { warn "docker missing on head"; ok=1; }
  command -v nvidia-smi >/dev/null && info "GPU: $(nvidia-smi -L | head -1)" || { warn "nvidia-smi missing"; ok=1; }
  [[ -f "$SSH_IDENTITY" ]] && info "SSH key $SSH_IDENTITY" || { warn "SSH key missing"; ok=1; }
  local host_arch
  host_arch=$(uname -m)
  info "host arch: $host_arch (GB10 expects aarch64)"

  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    info "overlay image present ($(docker image inspect -f '{{.Architecture}}' "$IMAGE"))"
  else
    warn "image $IMAGE missing — ./start.sh build (pulls $BASE_IMAGE)"
  fi

  if [[ -f "$MODEL_DIR/config.json" ]]; then
    local n
    n=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${n:-0}" -ge "$EXPECTED_SHARDS" ]]; then
      info "checkpoint OK ($n / $EXPECTED_SHARDS shards, $(du -sh "$MODEL_DIR" | awk '{print $1}'))"
    else
      warn "partial checkpoint ($n / $EXPECTED_SHARDS) — ./start.sh download"
      ok=1
    fi
  else
    warn "checkpoint missing — ./start.sh download"
    ok=1
  fi

  if [[ "$OFFLOAD_MODE" == "ram" ]]; then
    warn "RAM Engram mode pins ~189 GiB into unified memory — will not fit on Spark. Use nvme."
    ok=1
  fi
  if nfs_rpc_ready 127.0.0.1; then
    local _mounts="" _i
    for _i in "${!WORKER_HOSTS[@]}"; do _mounts+=" ${WORKER_HOSTS[$_i]}→$(nfs_server_ip_for "${WORKER_HOSTS[$_i]}" "$_i" 2>/dev/null || echo '?')"; done
    info "NFSv4 listening on this host (workers should mount CX7:${_mounts})"
  else
    warn "NFSv4 not listening yet — ./start.sh share will start or reuse the exporter"
  fi
  info "weights: spark2/spark3 use docker NFS volume $NFS_VOLUME (head $MODEL_DIR); no rsync/SSHFS"
  if [[ "$TP_SIZE" -eq 3 ]]; then
    info "TP3 note: heads=64, o_groups=8, vocab=129280 are not divisible by 3; adapter/tp3_pad.py pads them"
    info "  (heads 64→96, groups 8→12, draft experts 128→129). experts=384 divides. Rank 2 holds padded shards only."
    info "  2 Sparks cannot hold the MXFP4 experts (290 GiB / 2 = 145 GiB > 121 GiB)."
  elif [[ "$TP_SIZE" -eq 4 ]]; then
    info "TP4 note: heads, o_groups, draft experts and vocab all divide by 4; no padding (DSV41_TP_PAD=${DSV41_TP_PAD:-0})."
  fi

  if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
    warn "GPU already has compute apps:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    warn "Stop the other stack (e.g. ./start.sh stop in glm-5.3-flash-sm120) before serving, or FORCE=1"
  fi

  local h
  for h in "${WORKER_HOSTS[@]}"; do
    if remote_on "$h" "hostname" >/tmp/dsv41-host-"$h".txt 2>/tmp/dsv41-ssh-"$h".err; then
      info "SSH $h OK → $(tr -d '\r' </tmp/dsv41-host-"$h".txt)"
      remote_on "$h" "command -v docker >/dev/null && nvidia-smi -L | head -1 && test -d /dev/infiniband && echo IB_OK" \
        || { warn "docker/GPU/IB check failed on $h"; ok=1; }
    else
      err "SSH to $h FAILED"
      cat /tmp/dsv41-ssh-"$h".err || true
      ok=1
    fi
  done

  if [[ "$ok" -ne 0 ]]; then
    if [[ "${DOCTOR_STRICT:-1}" == "1" && "${1:-}" == "strict" ]]; then
      die "doctor found blocking issues"
    fi
    warn "doctor: issues found"
    return 1
  fi
  info "doctor: ready"
}

cmd_download() {
  info "=== download $HF_REPO @$HF_REVISION → $MODEL_DIR ==="
  mkdir -p "$MODEL_DIR"
  local n
  n=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
  if [[ -f "$MODEL_DIR/config.json" && "${n:-0}" -ge "$EXPECTED_SHARDS" ]]; then
    info "checkpoint already present ($n safetensors) — skip"
  else
    command -v hf >/dev/null || die "hf CLI missing (pip install -U huggingface_hub[cli])"
    hf download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
  fi
  ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
  info "head: $COMMON_MODEL → $MODEL_DIR"
}

cmd_pull() {
  info "=== pull $BASE_IMAGE on all nodes ==="
  ensure_ssh_keys
  docker pull --platform linux/arm64 "$BASE_IMAGE"
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    info "pulling on $h ..."
    remote_on "$h" --timeout "${PULL_TIMEOUT:-0}" \
      "docker pull --platform linux/arm64 $(printf '%q' "$BASE_IMAGE")"
  done
  info "base image present on all nodes"
}

cmd_build() {
  info "=== build $IMAGE from $ROOT ==="
  if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    cmd_pull
  fi
  docker build -t "$IMAGE" "$ROOT"
  local img_arch
  img_arch=$(docker image inspect -f '{{.Architecture}}' "$IMAGE")
  [[ "$img_arch" == "arm64" ]] || die "expected arm64 image, got $img_arch"
  info "head built $IMAGE arch=$img_arch"

  ensure_ssh_keys
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    info "rsync recipe → $h:$WORKER_DIR"
    remote_on "$h" "mkdir -p $(printf '%q' "$WORKER_DIR")"
    rsync -aH --delete --exclude '.env' --exclude '.env.tp4' --exclude 'state' --exclude 'state-tp4' \
      --exclude 'logs' --exclude 'logs-tp4' --exclude 'models' \
      --exclude 'engram' \
      -e "$(ssh_rsync_e)" \
      "$ROOT/" "${WORKER_USER}@${h}:${WORKER_DIR}/"
    info "docker build on $h ..."
    remote_on "$h" --timeout "${BUILD_TIMEOUT:-0}" \
      "docker image inspect $(printf '%q' "$BASE_IMAGE") >/dev/null || docker pull --platform linux/arm64 $(printf '%q' "$BASE_IMAGE")
       cd $(printf '%q' "$WORKER_DIR") && docker build -t $(printf '%q' "$IMAGE") ."
  done
  info "overlay image on all 3 nodes"
}

cmd_share() {
  info "=== share spark1 checkpoint over NFSv4 on ConnectX ==="
  [[ -f "$MODEL_DIR/config.json" ]] || die "no checkpoint — ./start.sh download"
  ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
  ensure_ssh_keys
  nfs_ensure_server
  nfs_publish_model
  local i=0 h
  for h in "${WORKER_HOSTS[@]}"; do
    nfs_ensure_worker_volume "$h" "$i"
    if nfs_worker_has_model "$h"; then
      info "$h: $NFS_VOLUME has config.json"
    else
      die "$h cannot see the checkpoint over NFS. Check ACL (10.0.23.0/24 for spark3) and docker logs of the nfs exporter."
    fi
    i=$((i + 1))
  done
  info "spark2 + spark3 read $MODEL_DIR from spark1 over NFS (no local copy)"
}
cmd_mount() { cmd_share; }

cmd_sync() {
  die "rsync of this 476 GiB checkpoint will not fit spark3. Use ./start.sh share (NFSv4 from spark1)."
}

_busy_gpu() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'
}

cmd_serve() {
  DOCTOR_STRICT=0 cmd_doctor || true
  [[ -f "$MODEL_DIR/config.json" ]] || cmd_download
  ln -sfn "$MODEL_DIR" "$COMMON_MODEL"

  if _busy_gpu && [[ "${FORCE:-0}" != "1" ]]; then
    die "GPU is busy (glm53-exl3 or similar). Stop the other stack, or FORCE=1 ./start.sh serve"
  fi

  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    cmd_build
  fi

  local h need_share=0
  for h in "${WORKER_HOSTS[@]}"; do
    if ! nfs_worker_has_model "$h"; then
      need_share=1
    fi
    if ! remote_ok_on "$h" "docker image inspect $(printf '%q' "$IMAGE") >/dev/null 2>&1"; then
      info "image missing on $h — building"
      cmd_build
      break
    fi
  done
  if [[ "$need_share" -eq 1 || "$NFS_SHARE" == "1" ]]; then
    cmd_share
  fi

  API_KEY="$(api_key)"
  export API_KEY
  if [[ -n "$API_KEY" ]]; then
    echo "$API_KEY" > "$STATE_DIR/api-key"
    chmod 600 "$STATE_DIR/api-key"
  else
    rm -f "$STATE_DIR/api-key"
  fi

  local GID_HEAD gi g
  local -a WORKER_GIDS=()
  GID_HEAD=$(gid_index_local "$HEAD_IP" 2>/dev/null || true)
  GID_HEAD="${GID_HEAD:-${NCCL_IB_GID_INDEX:-3}}"
  for gi in "${!WORKER_IPS[@]}"; do
    g=$(gid_index_remote "${WORKER_HOSTS[$gi]}" "${WORKER_IPS[$gi]}" | tr -d '\r' || true)
    WORKER_GIDS+=("${g:-${NCCL_IB_GID_INDEX:-3}}")
  done
  info "RoCEv2 GID indexes: head=$GID_HEAD workers=${WORKER_GIDS[*]}"

  docker rm -f "$HEAD_CTN" >/dev/null 2>&1 || true
  for h in "${WORKER_HOSTS[@]}"; do
    remote_on "$h" "docker rm -f $WORKER_CTN >/dev/null 2>&1 || true" || true
  done

  push_spec_tables
  info "Starting workers (ranks 1..${#WORKER_IPS[@]}) first..."
  local idx=0 wip wgid rank
  for h in "${WORKER_HOSTS[@]}"; do
    wip="${WORKER_IPS[$idx]}"
    wgid="${WORKER_GIDS[$idx]}"
    rank=$((idx + 1))
    remote_on "$h" "
      set -e
      docker volume inspect $NFS_VOLUME >/dev/null || { echo 'MISSING docker volume $NFS_VOLUME on $h — run ./start.sh share'; exit 1; }
      test -d /dev/infiniband || { echo 'MISSING /dev/infiniband on $h'; exit 1; }
      mkdir -p $WORKER_DIR/state $WORKER_DIR/logs
      NCCL_VOL=''
      NCCL_ENV=''
      if [ -f \$HOME/nccl-2.30.7/libnccl.so.2.30.7 ]; then
        if [ \"$NCCL_OVERLAY_PIP\" = 1 ]; then
          NCCL_VOL=\"-v \$HOME/nccl-2.30.7/libnccl.so.2.30.7:$NCCL_PIP_SO:ro\"
        else
          NCCL_VOL=\"-v \$HOME/nccl-2.30.7:$NCCL_CONTAINER_DIR:ro\"
          NCCL_ENV='-e LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR'
        fi
      fi
      docker run -d --name $WORKER_CTN \
        --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all \
        --shm-size ${SHM_SIZE:-32g} \
        --ulimit memlock=-1:-1 --ulimit stack=67108864 \
        --device /dev/infiniband:/dev/infiniband \
        -v $NFS_VOLUME:/models/DeepSeek-V4.1-Flash:ro \
        -v $WORKER_DIR/state:/state \
        -v \$HOME/.cache:/root/.cache \
        \$NCCL_VOL \$NCCL_ENV \\
$(worker_env_lines "$wip" "$wgid" "$rank")
        -e API_KEY=$(printf '%q' "$API_KEY") \\
        -e EXTRA_SGLANG_ARGS=$(printf '%q' "${EXTRA_SGLANG_ARGS:-}") \\
        $(printf '%q' "$IMAGE") run
    "
    idx=$((idx + 1))
  done

  info "Starting head (rank 0, API :$PORT)..."
  local -a head_args=()
  docker_common_args head_args "$HEAD_IP" "$GID_HEAD"
  local head_cid=""
  head_cid=$(docker run -d --name "$HEAD_CTN" --restart=no \
    "${head_args[@]}" \
    -e NODE_RANK=0 \
    -e SKIP_SMOKE="$SKIP_SMOKE" \
    -e SMOKE_QUICK="$SMOKE_QUICK" \
    "$IMAGE" run)
  [[ -n "$head_cid" ]] || die "docker run produced no container id"
  info "head cid=${head_cid:0:12}"

  mkdir -p "$LOG_DIR"
  : >"$SERVE_LOG"

  _stop_logtail() {
    local p
    p=$(cat "$LOG_DIR/logtail.pid" 2>/dev/null || true)
    rm -f "$LOG_DIR/logtail.pid"
    if [[ -n "${p:-}" ]]; then
      # Children first (docker logs + tee), while they are still parented to the
      # subshell; killing the subshell first reparents them and they outlive us,
      # which kept the log streaming onto the terminal after "this script is done".
      pkill -TERM -P "$p" 2>/dev/null || true
      kill -TERM "$p" 2>/dev/null || true
    fi
    # Only the follower started by this script (exact command lines); never the engine.
    pkill -TERM -f "docker logs -f --tail=20 ${HEAD_CTN}" 2>/dev/null || true
    pkill -TERM -f "tee -a ${SERVE_LOG}" 2>/dev/null || true
  }

  # Live engine log on this TTY, also copied to $SERVE_LOG. Stopped when we
  # return to the shell (ready or fail) — the container keeps running.
  info "streaming engine log (returns to shell when the engine is up, smoke-tested and warmed up, or the head dies)..."
  ( docker logs -f --tail=20 "$HEAD_CTN" 2>&1 | tee -a "$SERVE_LOG" ) &
  echo $! >"$LOG_DIR/logtail.pid"
  trap '_stop_logtail' EXIT

  _dump_head_fail() {
    _stop_logtail
    trap - EXIT
    local st oom errstr
    st=$(docker inspect -f '{{.State.Status}}' "$HEAD_CTN" 2>/dev/null || echo missing)
    oom=$(docker inspect -f '{{.State.OOMKilled}}' "$HEAD_CTN" 2>/dev/null || echo '?')
    errstr=$(docker inspect -f '{{.State.Error}}' "$HEAD_CTN" 2>/dev/null || echo '')
    echo
    err "head is ${st} (oom=${oom}) cid=${head_cid:0:12}"
    [[ -n "$errstr" ]] && err "docker: $errstr"
    echo "---- $SERVE_LOG (tail) ----"
    tail -n 120 "$SERVE_LOG" 2>/dev/null || true
    echo "---- workers ----"
    local wh
    for wh in "${WORKER_HOSTS[@]}"; do
      echo "== $wh =="
      remote_on "$wh" "docker ps -a --filter name=$WORKER_CTN --format '{{.Names}} {{.Status}}'; docker logs --tail=40 $WORKER_CTN 2>/dev/null | tail -40" || true
    done
  }

  local i=0 st
  while (( i < ${READY_TIMEOUT:-360} )); do
    # Ready = boot.py printed its banner, i.e. /health answered AND the smoke test
    # and warm-up passed (WARMUP=0 / SKIP_SMOKE=1 skip them; then it is /health alone).
    if docker logs "$HEAD_CTN" 2>&1 | grep -q "^Ready: API on port ${PORT}" \
       || { [[ "${SKIP_SMOKE:-0}" == "1" ]] && curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }; then
      _stop_logtail
      trap - EXIT
      # Keep the engine log flowing into $SERVE_LOG after we return, detached from
      # this terminal (the TTY follower above is what was stopped). Without this the
      # file ends at readiness and a later failure leaves no trace once the
      # container is removed.
      ( setsid docker logs -f --since 1s "$HEAD_CTN" >>"$SERVE_LOG" 2>&1 </dev/null & ) 2>/dev/null
      echo
      info "API is up on :$PORT (smoke + warm-up passed) — engine keeps running, this script is done."
      cmd_status
      echo
      echo "  curl http://$HEAD_IP:$PORT/v1/chat/completions \\"
      if [[ -n "$API_KEY" ]]; then
        echo "    -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' \\"
      else
        echo "    -H 'Content-Type: application/json' \\"
      fi
      echo "    -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 19 + 23?\"}],\"chat_template_kwargs\":{\"thinking\":false}}'"
      echo
      [[ -n "$API_KEY" ]] && echo "  key:  $STATE_DIR/api-key" || echo "  auth: none (no API key)"
      echo "  logs: ./start.sh logs | ./start.sh logs -f | ./start.sh logs worker1"
      echo "  stop: ./stop.sh"
      return 0
    fi
    st=$(docker inspect -f '{{.State.Status}}' "$HEAD_CTN" 2>/dev/null || echo missing)
    case "$st" in
      running|created|restarting) ;;
      *)
        _dump_head_fail
        exit 1
        ;;
    esac
    i=$((i + 1))
    sleep 10
  done
  _stop_logtail
  trap - EXIT
  die "timed out waiting for :$PORT — ./start.sh logs"
}

cmd_stop() {
  exec "$ROOT/stop.sh" "$@"
}

cmd_status() {
  echo "== head =="
  docker ps --filter "name=$HEAD_CTN" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  echo
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    echo "== worker $h =="
    remote_on "$h" "docker ps --filter name=$WORKER_CTN --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' ; test -f $COMMON_MODEL/config.json && echo weights:OK || echo weights:MISSING" || warn "status SSH $h failed"
    echo
  done
  echo "== API =="
  local key
  key=$(cat "$STATE_DIR/api-key" 2>/dev/null || true)
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" \
     || { [[ -n "$key" ]] && curl -fsS --max-time 5 -H "Authorization: Bearer $key" "http://127.0.0.1:${PORT}/v1/models"; }; then
    echo
  else
    echo "(not responding on :$PORT)"
  fi
}

cmd_logs() {
  local who="${1:-head}" lines="${2:-120}"
  case "$who" in
    ''|head|[0-9]*)
      [[ "$who" =~ ^[0-9]+$ ]] && lines="$who"
      docker logs --tail="$lines" "$HEAD_CTN" 2>&1 || tail -n "$lines" "$SERVE_LOG"
      ;;
    worker*|w[0-9]*)
      local n="${who#worker}"; n="${n#w}"; n="${n:-1}"
      [[ "$n" =~ ^[0-9]+$ && "$n" -ge 1 && "$n" -le ${#WORKER_HOSTS[@]} ]] || die "no such worker: $who (1..${#WORKER_HOSTS[@]})"
      remote_on "${WORKER_HOSTS[$((n - 1))]}" "docker logs --tail=$lines $WORKER_CTN" || true
      ;;
    -f|follow)
      docker logs -f "$HEAD_CTN"
      ;;
    *)
      docker logs --tail="$lines" "$HEAD_CTN" 2>&1 || true
      ;;
  esac
}

cmd_smoke() {
  local key auth=()
  key=$(cat "$STATE_DIR/api-key" 2>/dev/null || true)
  [[ -n "$key" ]] && auth=(-H "Authorization: Bearer $key")
  info "smoke: 19+23 (thinking off)"
  curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    "${auth[@]}" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"What is 19 + 23? Reply only with the number.\"}]}"
  echo
}

usage() {
  sed -n '2,24p' "$0" | tr -d '#'
}

CMD="${1:-serve}"
shift || true
# Repack each rank's owned Engram rows onto node-local NVMe (~68 GiB/node).
# A miss then costs one local read instead of two, and on a worker it stops
# being an NFS round trip to the head. Idempotent: complete shards are skipped.
cmd_pack() {
  local src
  src=$(model_src)
  info "packing Engram shards: head $ENGRAM_DIR, workers $WORKER_ENGRAM_DIR"
  mkdir -p "$ENGRAM_DIR"
  docker run --rm --network host \
    -v "$src:/models/DeepSeek-V4.1-Flash:ro" \
    -v "$ENGRAM_DIR:/engram" \
    -e DSV41_SOURCE=/models/DeepSeek-V4.1-Flash \
    --entrypoint python3 "$IMAGE" \
    /opt/dsv41/scripts/pack_engram.py --rank 0 --tp "$TP_SIZE" --out /engram \
    || die "pack failed on head"
  local idx=1 host
  for host in "${WORKER_IPS[@]}"; do
    info "packing rank $idx on $host..."
    remote_on "$host" "mkdir -p $WORKER_ENGRAM_DIR && docker run --rm --network host \
      -v $NFS_VOLUME:/models/DeepSeek-V4.1-Flash:ro \
      -v $WORKER_ENGRAM_DIR:/engram \
      -e DSV41_SOURCE=/models/DeepSeek-V4.1-Flash \
      --entrypoint python3 $IMAGE \
      /opt/dsv41/scripts/pack_engram.py --rank $idx --tp $TP_SIZE --out /engram" \
      || die "pack failed on $host"
    idx=$((idx + 1))
  done
  info "Engram shards packed on all 3 nodes"
}

case "$CMD" in
  serve|start|"") cmd_serve "$@" ;;
  doctor) cmd_doctor strict ;;
  pull) cmd_pull ;;
  build) cmd_build ;;
  pack) cmd_pack ;;
  download) cmd_download ;;
  share|mount) cmd_share ;;
  sync) cmd_sync ;;
  stop) cmd_stop "$@" ;;
  status) cmd_status ;;
  logs) cmd_logs "$@" ;;
  smoke) cmd_smoke ;;
  -h|--help|help) usage ;;
  *) die "unknown command: $CMD (try ./start.sh help)" ;;
esac
