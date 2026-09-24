#!/usr/bin/env bash
# Run benchmarks/overnight_bench.py from a worker (off the head's CPU) and join the head's
# "Decode batch" acceptance lines for each phase window.
# Usage: scripts/overnight/bench.sh LABEL [reps] [max_tokens]
#   PHASES=c1,c4[,c8,...]  C1_WORKLOADS=prose,code,chat_sampled,prose2
#   BENCH_HOST=spark2 (worker that sends the load)  BASE_URL=http://10.0.0.1:8888 (head API)
#   STATE_DIR=state (api-key)   RESULTS_DIR=... (required)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
LABEL="$1"; REPS="${2:-5}"; MAXT="${3:-512}"
R="${RESULTS_DIR:?set RESULTS_DIR}"; mkdir -p "$R"
BENCH_HOST="${BENCH_HOST:-spark2}"; BASE_URL="${BASE_URL:-http://10.0.0.1:8888}"
ID="$HOME/.ssh/id_ed25519_shared"; U="${WORKER_USER:-zurih}"
REM="ssh -i $ID -o IdentitiesOnly=yes $U@$BENCH_HOST"
KEY=$(cat "${STATE_DIR:-state}/api-key" 2>/dev/null || true)
scp -q -i "$ID" benchmarks/overnight_bench.py "$U@$BENCH_HOST:/tmp/overnight_bench.py"
$REM "rm -f /tmp/ob-$LABEL.jsonl; API_KEY=$KEY PHASES=${PHASES:-c1,c4} C1_WORKLOADS=${C1_WORKLOADS:-prose,code,chat_sampled,prose2} BASE_URL=$BASE_URL python3 /tmp/overnight_bench.py /tmp/ob-$LABEL.jsonl $REPS $MAXT" | tr -d '\r' | tee "$R/bench-$LABEL.txt"
scp -q -i "$ID" "$U@$BENCH_HOST:/tmp/ob-$LABEL.jsonl" "$U@$BENCH_HOST:/tmp/ob-$LABEL.summary.json" "$R/"
python3 scripts/overnight/accept.py "$R/ob-$LABEL.summary.json" | tee -a "$R/bench-$LABEL.txt"
