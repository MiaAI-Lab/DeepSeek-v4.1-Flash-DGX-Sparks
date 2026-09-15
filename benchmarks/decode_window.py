#!/usr/bin/env python3
"""Mia-style decode-window sweep over the public OpenAI chat path.

Estimates the window for tokens 129-641 from character growth on greedy code
generation at C1/C2/C4/C8, plus wall-clock aggregate tok/s. Requires an idle server.

    BASE_URL=http://127.0.0.1:8888 python3 benchmarks/decode_window.py
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from .openai_serving import request_headers
except ImportError:  # direct script invocation
    from openai_serving import request_headers

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8888").rstrip("/")
MODEL = os.environ.get("SERVED_MODEL_NAME", "deepseek-v4.1-flash")
OUT = Path(os.environ.get("OUT_DIR", "docs/results/decode-window-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
INSTRUCTION = (
    "Write a complete Python module implementing an LRU cache with a doubly "
    "linked list and dictionary. Include get, put, delete, iteration, resize, "
    "clear, invariant validation and detailed docstrings. Then provide ten "
    "usage examples. Return code only. Implement all methods fully."
)


def loads():
    req = urllib.request.Request(BASE + "/v1/loads", headers=request_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        rows = json.load(r).get("loads")
        if not rows or any(k not in rows[0] for k in ("num_running_reqs", "num_waiting_reqs")):
            raise RuntimeError("cannot verify idle state: missing load counters")
        return rows[0]


def require_idle():
    snap = loads()
    running = snap.get("num_running_reqs") or 0
    waiting = snap.get("num_waiting_reqs") or 0
    if running or waiting:
        raise SystemExit(f"server not idle: running={running} waiting={waiting}")
    return snap


def stream_one(index, concurrency):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{index}:{concurrency}\n{INSTRUCTION}"}],
        "temperature": 0,
        "max_tokens": 4096,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=request_headers(),
    )
    events = []
    char_count = 0
    ttft = None
    done = False
    usage = {}
    finish = None
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError("server returned an SSE error")
            now = time.monotonic() - start
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                char_count += len(piece)
                if ttft is None:
                    ttft = now
            if choice.get("finish_reason"):
                finish = choice.get("finish_reason")
            if event.get("usage"):
                usage = event["usage"]
            events.append({"seconds": now, "chars": char_count, "finish": finish})
    if not done or finish is None:
        raise RuntimeError("incomplete SSE stream (missing finish reason or [DONE])")
    elapsed = time.monotonic() - start
    completion = usage.get("completion_tokens")
    # Approximate token timeline from char growth if usage arrives only at the end.
    # This is a character-scaled estimate, not exact token timestamps.
    # Final usage is exact for wall-clock aggregate throughput.
    chars = [e["chars"] for e in events]
    times = [e["seconds"] for e in events]
    window = None
    if not isinstance(completion, int) or completion <= 0:
        raise RuntimeError("missing positive completion_tokens in final usage")
    if completion >= 641 and chars and chars[-1] > 0:
        def at_tokens(n):
            target = n / completion * chars[-1]
            for t, c in zip(times, chars):
                if c >= target:
                    return t
            return None
        left, right = at_tokens(129), at_tokens(641)
        if left is not None and right is not None and right > left:
            window = 512 / (right - left)
    return {
        "index": index,
        "concurrency": concurrency,
        "seconds": round(elapsed, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "window_method": "character_scaled_estimate",
        "window_129_641_tps": round(window, 2) if window else None,
        "completion_tokens": completion,
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": finish,
        "chars": chars[-1] if chars else 0,
        "events": len(events),
    }


def wave(concurrency):
    idle = require_idle()
    wall0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda i: stream_one(i, concurrency), range(concurrency)))
    wall = time.monotonic() - wall0
    after = loads()
    completion = sum(r.get("completion_tokens") or 0 for r in rows)
    windows = [r["window_129_641_tps"] for r in rows if r.get("window_129_641_tps")]
    ttfts = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    return {
        "concurrency": concurrency,
        "wall_seconds": round(wall, 3),
        "successes": sum(1 for r in rows if r.get("completion_tokens")),
        "completion_tokens": completion,
        "aggregate_tok_s": round(completion / wall, 2) if wall else None,
        "median_window_tps": round(statistics.median(windows), 2) if windows else None,
        "mean_window_tps": round(statistics.mean(windows), 2) if windows else None,
        "median_ttft_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "finish_reasons": [r.get("finish_reason") for r in rows],
        "idle_before": {k: idle.get(k) for k in ("num_running_reqs", "num_waiting_reqs", "num_used_tokens")},
        "after": {k: after.get(k) for k in ("num_running_reqs", "num_waiting_reqs", "num_used_tokens")},
        "rows": rows,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "captured_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "base_url": BASE,
        "waves": [],
    }
    require_idle()
    for conc in (1, 2, 4, 8):
        print(f"wave C{conc} starting", flush=True)
        cell = wave(conc)
        report["waves"].append(cell)
        (OUT / "partial.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: cell[k] for k in (
            "concurrency", "wall_seconds", "aggregate_tok_s",
            "median_window_tps", "median_ttft_s", "successes"
        )}), flush=True)
        time.sleep(2)
        require_idle()
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(str(OUT), flush=True)


if __name__ == "__main__":
    main()
