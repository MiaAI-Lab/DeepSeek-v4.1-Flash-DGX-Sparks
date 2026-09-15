# NCCL setup shared by the head and the generated worker Bash script.
# Sourced by start.sh; no probing or side effects at import time.

nccl_library() {
  local dir="$1" name
  for name in libnccl.so.2.30.7 libnccl.so.2; do
    if [[ -f "$dir/$name" && -r "$dir/$name" ]]; then
      printf '%s\n' "$dir/$name"
      return 0
    fi
  done
  return 1
}

nccl_mount_args() {
  local -n mounts=$1
  local library
  if library=$(nccl_library "$NCCL_HOST_DIR"); then
    if [[ "$NCCL_OVERLAY_PIP" == 1 ]]; then
      mounts+=(-v "$library:$NCCL_PIP_SO:ro")
    else
      mounts+=(-v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro"
               -e "LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR")
    fi
  elif [[ "$NCCL_OVERLAY_PIP" == 1 || "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2
    return 1
  fi
}

switchless_ring_args() {
  [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  local -n ring_args=$1
  ring_args+=(-e NCCL_SWITCHLESS_RING_ONLY=1
    -e "NCCL_ALGO=${NCCL_ALGO:-Ring}"
    -e "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}"
    -e "NCCL_IB_SUBNET_PREFIX_LEN=${NCCL_IB_SUBNET_PREFIX_LEN:-24}"
    -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}"
    -e "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}")
}

nccl_validate_config() {
  local flag
  for flag in NCCL_SWITCHLESS_RING_ONLY NCCL_OVERLAY_PIP DSV41_SERIAL_WEIGHT_LOAD; do
    case "${!flag:-0}" in 0|1) ;; *) echo "$flag must be 0 or 1" >&2; return 1;; esac
  done
  if [[ -n "${NCCL_WORKER_DIR:-}" && "$NCCL_WORKER_DIR" != /* ]]; then
    echo 'NCCL_WORKER_DIR must be absolute' >&2
    return 1
  fi
  [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  if [[ "$NNODES" != 4 || "$TP_SIZE" != 4 || "$EP_SIZE" != 4 ||
        "$NCCL_OVERLAY_PIP" != 1 || "${NCCL_ALGO:-Ring}" != Ring ||
        "$NCCL_NET" != IB || "$NCCL_IB_DISABLE" != 0 ||
        "${NCCL_IB_SUBNET_AWARE_ROUTING:-1}" != 1 ]]; then
    echo 'Ring requires NNODES=TP_SIZE=EP_SIZE=4, NCCL_OVERLAY_PIP=1, NCCL_ALGO=Ring, NCCL_NET=IB, NCCL_IB_DISABLE=0 and subnet-aware routing=1' >&2
    return 1
  fi
  local minimum="${NCCL_MIN_NCHANNELS:-4}" maximum="${NCCL_MAX_NCHANNELS:-32}"
  if [[ ! "$minimum" =~ ^[1-9][0-9]{0,2}$ || ! "$maximum" =~ ^[1-9][0-9]{0,2}$ ]] ||
     (( minimum > maximum )); then
    echo 'Ring requires positive NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS' >&2
    return 1
  fi
}

# MANAGEMENT IPs need not appear in the CX7 GID table. Find a common, nonzero
# IPv4 RoCE v2 index on all explicitly selected ports, or validate the override.
# The optional sysfs root is used by host-side fixtures.
ring_gid_index() {
  local sysfs="${1:-/sys/class/infiniband}" entry hca port state base g idx valid gid
  local -a entries=() ports=() candidates=()
  IFS=, read -r -a entries <<<"${IB_HCA#=}"
  for entry in "${entries[@]}"; do
    if [[ ! "$entry" =~ ^[a-zA-Z0-9_]+(:[1-9][0-9]*)?$ ]]; then
      echo 'Ring IB_HCA must list exact HCA names, optionally with :port' >&2
      return 1
    fi
    hca="${entry%%:*}"; port=1
    [[ "$entry" != *:* ]] || port="${entry##*:}"
    base="$sysfs/$hca/ports/$port"
    state=$(cat "$base/state" 2>/dev/null) || state=''
    if [[ "$state" != '4: ACTIVE' ]]; then
      echo "Ring: $hca:$port is not ACTIVE" >&2
      return 1
    fi
    ports+=("$base")
  done
  [[ ${#ports[@]} -gt 0 ]] || { echo 'Ring: IB_HCA is empty' >&2; return 1; }
  if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
    [[ "$NCCL_IB_GID_INDEX" =~ ^[0-9]+$ ]] || return 1
    candidates=("$NCCL_IB_GID_INDEX")
  else
    for g in "${ports[0]}"/gids/*; do candidates+=("${g##*/}"); done
  fi
  for idx in "${candidates[@]}"; do
    valid=1
    for base in "${ports[@]}"; do
      if [[ "$(cat "$base/gid_attrs/types/$idx" 2>/dev/null)" != 'RoCE v2' ]]; then
        valid=0; break
      fi
      gid=$(cat "$base/gids/$idx" 2>/dev/null) || gid=''
      if [[ ! "$gid" =~ ^0000:0000:0000:0000:0000:ffff:[[:xdigit:]]{4}:[[:xdigit:]]{4}$ ||
            "$gid" == *:0000:0000 ]]; then
        valid=0; break
      fi
    done
    if [[ "$valid" == 1 ]]; then printf '%s\n' "$idx"; return 0; fi
  done
  echo "Ring: no common IPv4 RoCE v2 GID on $IB_HCA (override: ${NCCL_IB_GID_INDEX:-auto})" >&2
  return 1
}

nccl_preflight() {
  local library
  if [[ "$NCCL_OVERLAY_PIP" == 1 || "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    library=$(nccl_library "$NCCL_HOST_DIR") || {
      echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2; return 1;
    }
    if [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] && ! grep -qa SWITCHLESS_RING_ONLY "$library"; then
      echo "NCCL: $library lacks SWITCHLESS_RING_ONLY; install the patched build" >&2
      return 1
    fi
    [[ "$NCCL_PIP_SO" == /* && "$NCCL_PIP_SO" != *:* && "$NCCL_HOST_DIR" == /* && "$NCCL_HOST_DIR" != *:* ]] || {
      echo 'NCCL overlay requires an absolute container path and paths without colons' >&2; return 1;
    }
    docker run --rm --pull=never --network none --entrypoint /bin/sh "$IMAGE" \
      -c 'test -f "$1"' sh "$NCCL_PIP_SO" || {
      echo "NCCL: $IMAGE lacks the pip library at $NCCL_PIP_SO" >&2; return 1;
    }
  fi
  if [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then ring_gid_index || return 1; fi
  return 0
}

# Home-relative head paths follow the worker's HOME, including custom subdirs.
# An explicit NCCL_WORKER_DIR (absolute) overrides that mapping on every worker.
nccl_worker_settings() {
  if [[ -n "${NCCL_WORKER_DIR:-}" ]]; then
    printf 'NCCL_HOST_DIR=%q\n' "$NCCL_WORKER_DIR"
  elif [[ "$NCCL_HOST_DIR" == "$HOME/"* ]]; then
    printf 'NCCL_HOST_DIR="$HOME"/%q\n' "${NCCL_HOST_DIR#"$HOME/"}"
  else
    printf 'NCCL_HOST_DIR=%q\n' "$NCCL_HOST_DIR"
  fi
  local key
  for key in NCCL_OVERLAY_PIP NCCL_PIP_SO NCCL_CONTAINER_DIR NCCL_SWITCHLESS_RING_ONLY IB_HCA IMAGE NCCL_IB_GID_INDEX; do
    printf '%s=%q\n' "$key" "${!key:-}"
  done
  declare -f nccl_library nccl_mount_args ring_gid_index nccl_preflight
}
