#!/usr/bin/env bash
# One TP4 experiment arm = one boot: .env.tp4 + scripts/tp4/arms/<ARM>.env, then checks,
# benchmark and quality gate, all saved under RESULTS_DIR.
#   scripts/tp4/arm.sh <ARM>            boot the arm, then measure
#   scripts/tp4/arm.sh baseline --no-boot   measure whatever TP4 config is serving now
# Env: RESULTS_DIR (default docs/results/tp4-<date>), BENCH_HOST (spark2), REPS (5),
#      PHASES (c1,c4,c8), NEEDLES (30,100)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
ARM="$1"; NOBOOT="${2:-}"
export RESULTS_DIR="${RESULTS_DIR:-docs/results/tp4-$(date +%Y%m%d)}"; mkdir -p "$RESULTS_DIR/envs"
export STATE_DIR="$ROOT/state-tp4" PHASES="${PHASES:-c1,c4,c8}"
BASE_ENV="$ROOT/.env.tp4"; [[ -f "$BASE_ENV" ]] || { echo ".env.tp4 missing: run ./start-tp4.sh doctor once (creates it from .env.tp4.example)"; exit 1; }
if [[ "$NOBOOT" != "--no-boot" ]]; then
  OV="$ROOT/scripts/tp4/arms/$ARM.env"; [[ -f "$OV" ]] || { echo "no such arm: $OV"; exit 1; }
  mapfile -t KV < <(grep -E '^[A-Z0-9_]+=' "$OV")
  BASE_ENV=.env.tp4 scripts/overnight/mkenv.sh "$ARM" "${KV[@]}" >/dev/null
  echo "[arm $ARM] settings changed vs .env.tp4:"; printf '  %s\n' "${KV[@]}"
  ./start-tp4.sh stop >/dev/null 2>&1 || true
  TP4_ENV_FILE="$RESULTS_DIR/envs/$ARM.env" ./start-tp4.sh serve > "$RESULTS_DIR/serve-$ARM.txt" 2>&1 \
    || { echo "[arm $ARM] boot FAILED, see $RESULTS_DIR/serve-$ARM.txt"; tail -40 "$RESULTS_DIR/serve-$ARM.txt"; exit 1; }
fi
LOG="$RESULTS_DIR/boot-$ARM-head.log"; docker logs dsv41-head > "$LOG" 2>&1
{
  echo "[arm $ARM] $(date)"
  echo "packed=True lines on rank 0: $(grep -c 'Exact nvme Engram.*packed=True' "$LOG") (expect 2; check workers: ./start-tp4.sh logs workerN)"
  grep -h -E 'max_total_num_tokens|memory calculation' "$LOG" | tail -2 | cut -c1-200
  grep -h -E '^\[(wo_a_w8|verify_cap|draft_head_fp8|autotune_keep)\]|Engram prefetch ARMED|folded into the draft|disagree' "$LOG" | sort | uniq -c | cut -c1-160
} | tee "$RESULTS_DIR/checks-$ARM.txt"
scripts/overnight/bench.sh "$ARM" "${REPS:-5}" 512
scripts/overnight/quality.sh "$ARM" "${NEEDLES:-30,100}"
echo "[arm $ARM] done: $RESULTS_DIR/{checks,bench,quality}-$ARM.txt"
