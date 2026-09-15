#!/usr/bin/env python3
"""OpenAI-compatible serving matrix for a live DeepSeek-V4.1-Flash endpoint.

Unlike benchmarks/matrix.py and benchmarks/sweep.py this talks to
/v1/chat/completions (the public serving path), not the internal /generate
port. It records wall-clock aggregate tok/s, finish reasons and live
/v1/loads snapshots. Use decode_window.py for streaming TTFT.

    BASE_URL=http://127.0.0.1:8888 python3 benchmarks/openai_serving.py
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def env_url():
    return os.environ.get("BASE_URL", "http://127.0.0.1:8888").rstrip("/")


def request_headers():
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = os.environ.get("API_KEY", "")
    if key and key.lower() not in ("none", "off", "dummy", "0"):
        headers["Authorization"] = "Bearer " + key
    return headers


def request(url, payload=None, timeout=1800, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers=request_headers(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return response.status, json.loads(body) if body else {}


def loads(base):
    try:
        _, payload = request(base + "/v1/loads", timeout=10)
        rows = payload.get("loads") or []
        return rows[0] if rows else payload
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def chat(base, messages, *, max_tokens, temperature=0, thinking=False, tools=None, extra=None, timeout=1800):
    payload = {
        "model": os.environ.get("SERVED_MODEL_NAME", "deepseek-v4.1-flash"),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"thinking": thinking},
        "stream": False,
    }
    if temperature:
        payload["top_p"] = 0.95
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if extra:
        payload.update(extra)
    started = time.monotonic()
    status, body = request(base + "/v1/chat/completions", payload, timeout=timeout)
    elapsed = time.monotonic() - started
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    return {
        "http": status,
        "seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "tool_calls": message.get("tool_calls") or [],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens") or 0,
        "total_tokens": usage.get("total_tokens"),
    }


PROSE = [
    "a lighthouse keeper's last winter",
    "a city that only exists at night",
    "two rivals sharing a train compartment",
    "a violin found in a flooded cellar",
    "the first market day after a long war",
    "a cartographer who maps dreams",
    "a bakery run by retired sailors",
    "an orchard planted on a rooftop",
]


def wave(base, concurrency, max_tokens, thinking=False):
    snap_before = loads(base)
    wall0 = time.monotonic()

    def one(i):
        return chat(
            base,
            [{"role": "user", "content": (
                f"Write a vivid short story of about {max_tokens} words about {PROSE[i % len(PROSE)]}. "
                "Prose only, no headings."
            )}],
            max_tokens=max_tokens,
            thinking=thinking,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, range(concurrency)))
    wall = time.monotonic() - wall0
    snap_after = loads(base)
    completion = sum(r.get("completion_tokens") or 0 for r in rows)
    oks = sum(1 for r in rows if r.get("http") == 200)
    per = [(r["completion_tokens"] or 0) / r["seconds"] for r in rows if r.get("seconds")]
    return {
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "wall_seconds": round(wall, 3),
        "successes": oks,
        "completion_tokens": completion,
        "aggregate_tok_s": round(completion / wall, 2) if wall else None,
        "mean_per_request_tok_s": round(statistics.mean(per), 2) if per else None,
        "min_per_request_tok_s": round(min(per), 2) if per else None,
        "finish_reasons": [r.get("finish_reason") for r in rows],
        "seconds": [r.get("seconds") for r in rows],
        "loads_before": snap_before,
        "loads_after": snap_after,
        "rows": rows,
    }


def correctness(base):
    out = {}
    out["models"] = request(base + "/v1/models", timeout=10)[1]
    out["health_http"] = request(base + "/health", timeout=10)[0]
    out["arithmetic"] = chat(base, [{"role": "user", "content": "What is 19 + 23? Reply only with the number."}], max_tokens=32)
    out["thinking"] = chat(
        base,
        [{"role": "user", "content": "Reply with exactly OK99 and nothing else."}],
        max_tokens=128,
        thinking=True,
    )
    out["tools"] = chat(
        base,
        [{"role": "user", "content": "Call get_weather with city Vigo."}],
        max_tokens=128,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }],
    )
    return out


NEEDLE_WORDS = ["harbour", "lantern", "meadow", "copper", "violin", "orchard", "compass", "thistle"]


def needle(base, target_tokens, concurrency=1, seed=100):
    """Approximate a needle-in-haystack of ~target_tokens prompt tokens."""
    def build(idx, reps):
        para = (
            f"Entry {idx}-{{i}}: the {NEEDLE_WORDS[idx % len(NEEDLE_WORDS)]} keeper noted "
            f"{{n}} crates by the gate before dusk. "
        )
        body = "".join(para.format(i=i, n=(i * 17 + idx) % 997) for i in range(reps))
        return body + f"\nThe secret code word is PELICAN-{idx}.\n" + body[:800] + "\nWhat is the secret code word? Answer with the code word only."

    # calibrate
    cal = chat(base, [{"role": "user", "content": build(1, 40)}], max_tokens=8)
    prompt = cal.get("prompt_tokens") or 200
    per = max(1.0, (prompt - 40) / 40.0)
    reps = max(10, int((target_tokens - 120) / per))
    wall0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(
            lambda i: chat(base, [{"role": "user", "content": build(seed + i, reps)}], max_tokens=16, timeout=3600),
            range(concurrency),
        ))
    wall = time.monotonic() - wall0
    ok = 0
    for i, row in enumerate(rows):
        expected = f"PELICAN-{seed + i}"
        text = (row.get("content") or "").strip()
        row["expected"] = expected
        row["ok"] = expected == text
        ok += int(row["ok"])
    return {
        "target_tokens": target_tokens,
        "concurrency": concurrency,
        "calibrated_prompt_tokens": [r.get("prompt_tokens") for r in rows],
        "wall_seconds": round(wall, 3),
        "correct": ok,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-needles", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    base = env_url()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(args.out) if args.out else Path("docs/results") / f"openai-serving-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "base_url": base,
        "captured_utc": stamp,
        "loads_idle": loads(base),
        "correctness": correctness(base),
        "decode": [],
        "needles": [],
    }
    (folder / "partial.json").write_text(json.dumps(report, indent=2))
    print("correctness done", flush=True)
    for conc in (1, 2, 4, 8):
        if conc > args.max_concurrency:
            break
        cell = wave(base, conc, 256, thinking=False)
        report["decode"].append(cell)
        (folder / "partial.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: cell[k] for k in ("concurrency", "wall_seconds", "aggregate_tok_s", "successes")}), flush=True)
    if not args.skip_needles:
        for size, conc in ((32768, 1), (32768, 4), (131072, 1)):
            if conc > args.max_concurrency:
                continue
            cell = needle(base, size, concurrency=conc)
            report["needles"].append({k: v for k, v in cell.items() if k != "rows"})
            (folder / f"needle-{size}-c{conc}.json").write_text(json.dumps(cell, indent=2))
            (folder / "partial.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"needle": size, "c": conc, "correct": cell["correct"], "wall": cell["wall_seconds"], "prompt": cell["calibrated_prompt_tokens"]}), flush=True)
    (folder / "summary.json").write_text(json.dumps(report, indent=2))
    print(str(folder), flush=True)


if __name__ == "__main__":
    main()
