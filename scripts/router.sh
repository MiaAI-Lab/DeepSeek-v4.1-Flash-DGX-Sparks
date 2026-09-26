#!/usr/bin/env bash
# router.sh — one OpenAI-compatible endpoint in front of several replicas (e.g. two TP4 groups on
# eight Sparks), using the SGLang router that ships in the serving image. See docs/tp8.md,
# "TP8 or 2 × TP4?".
#
#   scripts/router.sh start http://HEAD_A:8888 http://HEAD_B:8888   # listens on :$ROUTER_PORT
#   scripts/router.sh stop | status
#
# ROUTER_PORT (8889), ROUTER_METRICS_PORT (29000, Prometheus), ROUTER_POLICY (round_robin),
# IMAGE (dsv41-4x-spark:canary-roce).
# round_robin by default: with cache_aware, requests that share a prompt prefix all go to one
# replica, which then queues behind its slots while the other idles (measured: sparkDash prose c32
# 296 tok/s with cache_aware against 673 with round_robin). Use cache_aware only when shared
# prefixes are common and the load is already spread; power_of_two also balances well.
set -euo pipefail
NAME=dsv41-router
PORT="${ROUTER_PORT:-8889}"
METRICS_PORT="${ROUTER_METRICS_PORT:-29000}"
POLICY="${ROUTER_POLICY:-round_robin}"
IMAGE="${IMAGE:-dsv41-4x-spark:canary-roce}"

case "${1:-status}" in
  start)
    shift
    [[ $# -ge 2 ]] || { echo "usage: $0 start URL URL [URL...]" >&2; exit 2; }
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --network host --restart unless-stopped \
      --entrypoint python3 "$IMAGE" -m sglang_router.launch_router \
      --worker-urls "$@" --policy "$POLICY" --host 0.0.0.0 --port "$PORT" --prometheus-port "$METRICS_PORT" \
      --request-timeout-secs 1800 --max-payload-size 536870912 >/dev/null
    for _ in $(seq 1 60); do
      curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "router on :$PORT ($POLICY) -> $*"; exit 0; }
      sleep 2
    done
    echo "router did not become healthy; docker logs $NAME" >&2; exit 1 ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 || true; echo stopped ;;
  status) docker ps --filter "name=^$NAME\$" --format '{{.Names}} {{.Status}}'; curl -s -m 3 "http://127.0.0.1:$PORT/health" && echo ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
