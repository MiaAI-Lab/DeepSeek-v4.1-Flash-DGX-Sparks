#!/usr/bin/env bash
# TP4 preflight: read-only, no restarts. Checks what the plan in
# docs/tp4-plan.md needs before the first arm boots:
#   ssh to every worker, one base image across all nodes, overlay image present,
#   packed TP4 Engram shards (engram-l{1,14}-r<rank>of4.bin) or enough disk to pack them.
# Usage: scripts/tp4/preflight.sh            (reads .env.tp4, else .env.tp4.example)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
ENV="${TP4_ENV_FILE:-$ROOT/.env.tp4}"; [[ -f "$ENV" ]] || ENV="$ROOT/.env.tp4.example"
set -a; source "$ENV"; set +a
read -r -a HOSTS <<<"$(tr ',' ' ' <<<"${WORKER_HOSTS:-spark2 spark3 spark4}")"
ID="$(readlink -f "${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}")"; U="${WORKER_USER:-zurih}"
WDIR="${WORKER_DIR:-/home/$U/dsv41-4x-spark}"; WENG="${WORKER_ENGRAM_DIR:-$WDIR/engram}"
HENG="${ENGRAM_DIR:-$HOME/dsv41-engram}"; BASE="${BASE_IMAGE:-lmsysorg/sglang:dev-dsv41}"; IMG="${IMAGE:-dsv41-4x-spark:local}"
NEED_GB=50   # two layers x ~23.6 GiB per rank at TP4, plus margin
fail=0; ok(){ echo "  ok   $*"; }; bad(){ echo "  FAIL $*"; fail=1; }
probe() { # host rank engram_dir -> prints base_id|overlay?|packed_count|free_gb
  local cmd="b=\$(docker image inspect $BASE -f '{{.Id}}' 2>/dev/null | cut -c8-19); o=\$(docker image inspect $IMG >/dev/null 2>&1 && echo yes || echo no);
    p=\$(ls $3/engram-l1-r$2of4.bin $3/engram-l14-r$2of4.bin 2>/dev/null | wc -l); d=$3; while [ ! -d \"\$d\" ]; do d=\$(dirname \"\$d\"); done; f=\$(df -BG --output=avail \"\$d\" 2>/dev/null | tail -1 | tr -dc 0-9); echo \"\$b|\$o|\$p|\$f\""
  if [[ "$1" == local ]]; then bash -c "$cmd"; else ssh -i "$ID" -o IdentitiesOnly=yes -o ConnectTimeout=5 -o BatchMode=yes "$U@$1" "$cmd"; fi
}
echo "env: $ENV   base: $BASE   overlay: $IMG"
declare -A BID
r=0; for h in local "${HOSTS[@]}"; do
  name=$([[ $h == local ]] && hostname || echo "$h"); dir=$([[ $h == local ]] && echo "$HENG" || echo "$WENG")
  if ! out=$(probe "$h" "$r" "$dir" 2>/dev/null) || [[ -z "$out" ]]; then bad "rank $r $name: unreachable over ssh"; r=$((r+1)); continue; fi
  IFS='|' read -r bid ov pk fr <<<"$out"; BID[$bid]+="$name "
  [[ -n "$bid" ]] && ok "rank $r $name: base $bid" || bad "rank $r $name: base image $BASE missing"
  [[ "$ov" == yes ]] && ok "rank $r $name: overlay $IMG present" || echo "  todo rank $r $name: overlay missing -> ./start-tp4.sh build"
  if [[ "$pk" == 2 ]]; then ok "rank $r $name: packed shards present in $dir"
  elif (( ${fr:-0} >= NEED_GB )); then echo "  todo rank $r $name: not packed yet, ${fr} GB free in $dir (need ~$NEED_GB) -> ./start-tp4.sh pack"
  else bad "rank $r $name: not packed and only ${fr:-?} GB free in $dir (need ~$NEED_GB)"; fi
  r=$((r+1))
done
if (( ${#BID[@]} > 1 )); then bad "base image differs between nodes:"; for k in "${!BID[@]}"; do echo "         $k on ${BID[$k]}"; done
  echo "         align them first (docker save $BASE | ssh <node> docker load), see docs/tp4-plan.md"; fi
echo; [[ $fail == 0 ]] && echo "preflight: OK (do the 'todo' items, then follow docs/tp4-plan.md)" || echo "preflight: FAILED"
exit $fail
