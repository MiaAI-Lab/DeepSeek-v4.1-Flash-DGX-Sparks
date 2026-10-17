#!/usr/bin/env bash
# stop.sh — tear down DeepSeek-V4.1-Flash SGLang on all 3 Sparks.
#
# Stops:
#   - dsv41-head on spark1 (rank 0 + API :8888)
#   - dsv41-worker on spark2 and spark3
#   - log-tail helper
#   - leftover sglang.launch_server in those containers
#
# Keeps:
#   - checkpoint on spark1
#   - overlay image dsv41-3x-spark:local
#   - shared NFSv4 exporter (vllm-fn-nfs) — Qwen/GLM still use it
#   - docker volume dsv41-weights unless you pass --unmount
#
# Usage:
#   ./stop.sh              stop serve on all 3 nodes
#   ./stop.sh --unmount    also drop worker NFS volumes (weights stay on spark1)
#   ./start.sh stop        same
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

UNMOUNT="${UNMOUNT:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --unmount|-u) UNMOUNT=1 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1 (try ./stop.sh --help)" >&2
      exit 1
      ;;
  esac
  shift
done

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a
fi

_abs() { readlink -f "$1" 2>/dev/null || echo "$1"; }

HEAD_CTN="${HEAD_CTN:-dsv41-head}"
WORKER_CTN="${WORKER_CTN:-dsv41-worker}"
PORT="${PORT:-8888}"
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
WORKER_USER="${WORKER_USER-zurih}"   # "" lets ~/.ssh/config pick each worker's user (as start.sh)
SSH_IDENTITY="$(_abs "${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}")"
NFS_VOLUME="${NFS_VOLUME:-dsv41-weights}"
IMAGE="${IMAGE:-dsv41-3x-spark:local}"
REMOTE_PY="${REMOTE_PY:-$ROOT/scripts/remote.py}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
RM_TIMEOUT="${RM_TIMEOUT:-30}"

GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; NC=$'\033[0m'
info() { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YELLOW}[!]${NC} $*"; }
err()  { echo "${RED}[x]${NC} $*" >&2; }

remote_on() {
  local host="$1"; shift
  local env_args=()
  [[ -f "$ENV_FILE" ]] && env_args=(--env-file "$ENV_FILE")
  python3 "$REMOTE_PY" "${env_args[@]}" \
    --host "$host" --user "$WORKER_USER" \
    --identity "$SSH_IDENTITY" \
    --timeout "${REMOTE_TIMEOUT:-60}" \
    "bash -lc $(printf '%q' "$*")"
}

_rm_ctn() {
  local name="$1"
  timeout "$RM_TIMEOUT" docker rm -f "$name" >/dev/null 2>&1 || true
}

_stop_sglang_in() {
  local ctn="$1"
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$ctn" || return 0
  docker exec "$ctn" bash -lc '
    pkill -TERM -f "[s]glang.launch_server" >/dev/null 2>&1 || true
    pkill -TERM -f "[s]glang.srt" >/dev/null 2>&1 || true
    sleep 1
    pkill -KILL -f "[s]glang.launch_server" >/dev/null 2>&1 || true
  ' 2>/dev/null || true
}

info "=== stop DeepSeek-V4.1-Flash (3× Spark SGLang) ==="

if [[ -f "$LOG_DIR/logtail.pid" ]]; then
  kill "$(cat "$LOG_DIR/logtail.pid")" 2>/dev/null || true
  rm -f "$LOG_DIR/logtail.pid"
fi
pkill -f "docker logs -f ${HEAD_CTN}" >/dev/null 2>&1 || true

info "head: SIGTERM sglang in $HEAD_CTN, then remove"
_stop_sglang_in "$HEAD_CTN"
_rm_ctn "$HEAD_CTN"
# anything else this recipe named
ids=$(docker ps -aq --filter "name=dsv41-" 2>/dev/null || true)
if [[ -n "$ids" ]]; then
  # shellcheck disable=SC2086
  timeout "$RM_TIMEOUT" docker rm -f $ids >/dev/null 2>&1 || true
fi

for h in "${WORKER_HOSTS[@]}"; do
  info "worker ${WORKER_USER:+$WORKER_USER@}$h: stop $WORKER_CTN"
  if remote_on "$h" "
    if docker ps --format '{{.Names}}' | grep -qx $(printf '%q' "$WORKER_CTN"); then
      docker exec $(printf '%q' "$WORKER_CTN") bash -lc '
        pkill -TERM -f \"[s]glang.launch_server\" >/dev/null 2>&1 || true
        sleep 1
        pkill -KILL -f \"[s]glang.launch_server\" >/dev/null 2>&1 || true
      ' 2>/dev/null || true
    fi
    timeout ${RM_TIMEOUT} docker rm -f $(printf '%q' "$WORKER_CTN") >/dev/null 2>&1 || docker rm -f $(printf '%q' "$WORKER_CTN") >/dev/null 2>&1 || true
    ids=\$(docker ps -aq --filter name=dsv41- 2>/dev/null || true)
    [ -n \"\$ids\" ] && docker rm -f \$ids >/dev/null 2>&1 || true
    echo STOPPED_$h
  " 2>/dev/null | grep -q "STOPPED_$h"; then
    info "  $h: container gone"
  else
    warn "  $h: SSH/docker cleanup failed (node unreachable?). GPU there may still be busy."
  fi
  if [[ "$UNMOUNT" == "1" ]]; then
    info "  $h: drop NFS volume $NFS_VOLUME"
    remote_on "$h" "docker volume rm $(printf '%q' "$NFS_VOLUME") >/dev/null 2>&1 || true" || true
  fi
done

echo
if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
   || curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  warn "something is still answering on :${PORT}"
else
  info "API down on :${PORT}"
fi

left=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^dsv41-' || true)
if [[ -n "$left" ]]; then
  warn "still running on head: $left"
else
  info "no dsv41-* containers on head"
fi

info "kept: spark1 weights, $IMAGE overlay, vllm-fn-nfs exporter"
[[ "$UNMOUNT" == "1" ]] || info "worker NFS volume $NFS_VOLUME kept (./stop.sh --unmount to drop it)"
info "start again with: ./start.sh"
