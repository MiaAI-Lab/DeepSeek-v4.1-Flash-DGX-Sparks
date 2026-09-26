#!/usr/bin/env bash
# start-tp8.sh — DeepSeek-V4.1-Flash on 8× DGX Spark (TP8/EP2, switched RoCE fabric).
#
# Same engine, image and commands as start-tp4.sh, with a profile of its own:
#   .env.tp8      settings for this profile (copied from .env.tp8.example on first run)
#   state-tp8/    launch record, smoke result, api key, DSpark tables
#   logs-tp8/     engine log
# The TP3 (.env) and TP4 (.env.tp4) profiles are untouched, so one checkout can drive any
# fleet. TP8 runs the TP4 launcher path (DSV41_LAUNCHER=tp4) and the canary-roce image
# unchanged; everything TP8-specific is a setting in .env.tp8.example (see docs/tp8.md).
#
# Usage: ./start-tp8.sh doctor | build | share | pack | serve | stop | status | logs | smoke
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_FILE="$ROOT/.env.tp8"
export ENV_EXAMPLE="$ROOT/.env.tp8.example"
export STATE_DIR="${STATE_DIR:-$ROOT/state-tp8}"
export LOG_DIR="${LOG_DIR:-$ROOT/logs-tp8}"
export SERVE_LOG="${SERVE_LOG:-$LOG_DIR/dsv41.log}"
export DSV41_LAUNCHER=tp4
exec "$ROOT/start.sh" "$@"
