#!/usr/bin/env bash
# Run benchmarks/overnight_quality.py from a worker. Usage: quality.sh LABEL [needle_k,...]
#   BENCH_HOST=spark2  BASE_URL=http://10.0.0.1:8888  RESULTS_DIR=... (required)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
LABEL="$1"; NEEDLES="${2:-}"
R="${RESULTS_DIR:?set RESULTS_DIR}"; mkdir -p "$R"
BENCH_HOST="${BENCH_HOST:-spark2}"; BASE_URL="${BASE_URL:-http://10.0.0.1:8888}"
ID="$HOME/.ssh/id_ed25519_shared"; U="${WORKER_USER:-zurih}"
scp -q -i "$ID" benchmarks/overnight_quality.py "$U@$BENCH_HOST:/tmp/overnight_quality.py"
ssh -i "$ID" -o IdentitiesOnly=yes "$U@$BENCH_HOST" "BASE_URL=$BASE_URL python3 /tmp/overnight_quality.py /tmp/oq-$LABEL.json '$NEEDLES'" | tee "$R/quality-$LABEL.txt"
scp -q -i "$ID" "$U@$BENCH_HOST:/tmp/oq-$LABEL.json" "$R/"
