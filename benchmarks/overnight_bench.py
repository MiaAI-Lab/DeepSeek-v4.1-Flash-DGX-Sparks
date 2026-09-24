#!/usr/bin/env python3
"""Controlled decode benchmark for the overnight TP3 campaign.

Token counts come from the server's usage block (completion_tokens), never from
the number of SSE events. Timing uses the first streamed content/reasoning delta
as the first-token time, so:

    decode tok/s = (completion_tokens - 1) / (t_last - t_first)
    e2e tok/s    = completion_tokens / (t_last - t_send)

cN aggregate (PHASES=c1,c4,c8,...) = sum(completion_tokens) / (wave wall time), all N
requests (alternating prose and code) sent together. Prints UTC phase boundaries so acceptance/step lines in the head
log can be joined to each phase afterwards.

Usage: BASE_URL=http://10.0.0.1:8888 python3 overnight_bench.py OUT.jsonl [reps] [max_tokens]
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

BASE = os.environ.get("BASE_URL", "http://10.0.0.1:8888").rstrip("/")
MODEL = os.environ.get("SERVED_MODEL_NAME", "deepseek-v4.1-flash")
KEY = os.environ.get("API_KEY", "")

PROSE = ("Write a detailed, flowing essay of about 800 words on the history and engineering "
         "of lighthouses, from the Pharos of Alexandria to automated LED beacons. Use full "
         "paragraphs only: no lists, no headings, no markdown.")
CODE = ("Write a complete Python module implementing a thread-safe LRU cache using a doubly "
        "linked list and a dict, with get, put, delete, resize and a stats() method, full "
        "docstrings and type hints, followed by a unittest suite covering eviction order, "
        "resizing and concurrent access. Output only the code.")
PROSE2 = ("Tell the story of a small fishing village over one year as the seasons change, following "
          "three families. Write it as continuous literary prose of about 800 words, no headings or lists.")
CHAT = ("Explain to a curious fifteen-year-old how vaccines teach the immune system to "
        "recognise a virus, including memory B cells and T cells, and why booster shots "
        "exist. Be warm and concrete.")

WORKLOADS = {
    "prose": dict(prompt=PROSE, temperature=0.0),
    "code": dict(prompt=CODE, temperature=0.0),
    "chat_sampled": dict(prompt=CHAT, temperature=0.7, top_p=0.95),
    "prose2": dict(prompt=PROSE2, temperature=0.0),
}
# C1 workloads to run (prose2 was added at E2; earlier runs have the first three only)
C1_SET = os.environ.get("C1_WORKLOADS", "prose,code,chat_sampled,prose2").split(",")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def one(workload: str, max_tokens: int, tag: str) -> dict:
    w = WORKLOADS[workload]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": w["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": w["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    if "top_p" in w:
        body["top_p"] = w["top_p"]
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    finish = None
    text = []
    err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if piece:
                        t = time.perf_counter()
                        if t_first is None:
                            t_first = t
                        t_last = t
                        text.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except Exception as e:  # recorded, never silently dropped
        err = repr(e)
    t_end = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    rec = {"tag": tag, "workload": workload, "max_tokens": max_tokens, "finish": finish,
           "completion_tokens": ct, "prompt_tokens": (usage or {}).get("prompt_tokens"),
           "ttft_s": None if t_first is None else t_first - t0,
           "total_s": t_end - t0, "error": err, "text_head": "".join(text)[:160],
           "text_sha": __import__("hashlib").sha256("".join(text).encode()).hexdigest()[:16]}
    if ct and t_first is not None and t_last is not None and t_last > t_first and ct > 1:
        rec["decode_tps"] = (ct - 1) / (t_last - t_first)
        rec["e2e_tps"] = ct / (t_last - t0)
    return rec


def summarize(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def main():
    out = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    phases = os.environ.get("PHASES", "c1,c4").split(",")
    rows = []
    summary = {"base": BASE, "reps": reps, "max_tokens": max_tokens, "phases": {}}

    def log(r):
        rows.append(r)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")

    # Warm-up: one of each workload, discarded, plus one C4 wave.
    for w in C1_SET:
        one(w, 64, "warmup")
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda w: one(w, 64, "warmup"), ["prose", "code", "prose", "code"]))

    if "c1" in phases:
        for w in C1_SET:
            start = now_utc()
            rs = []
            for i in range(reps):
                r = one(w, max_tokens, f"c1-{w}")
                r["rep"] = i
                log(r)
                rs.append(r)
            summary["phases"][f"c1-{w}"] = {
                "start": start, "end": now_utc(),
                "decode_tps": summarize(rs, "decode_tps"), "e2e_tps": summarize(rs, "e2e_tps"),
                "ttft_s": summarize(rs, "ttft_s"),
                "completion_tokens": [r["completion_tokens"] for r in rs],
                "distinct_outputs": len({r["text_sha"] for r in rs}),
                "errors": [r["error"] for r in rs if r["error"]]}
    # cN phases (c4, c8, c16, ...): N concurrent requests, alternating prose and code, per wave.
    for ph in phases:
        if not (ph.startswith("c") and ph[1:].isdigit()) or ph == "c1":
            continue
        n = int(ph[1:])
        start = now_utc()
        waves = []
        mix = [("prose", "code")[i % 2] for i in range(n)]
        for i in range(reps):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda w: one(w, max_tokens, ph), mix))
            wall = time.perf_counter() - t0
            toks = sum(r["completion_tokens"] or 0 for r in rs)
            for r in rs:
                r["wave"] = i
                log(r)
            waves.append({"wall_s": wall, "tokens": toks, "agg_tps": toks / wall,
                          "per_stream_decode": [r.get("decode_tps") for r in rs],
                          "errors": [r["error"] for r in rs if r["error"]]})
        summary["phases"][ph] = {
            "start": start, "end": now_utc(),
            "agg_tps": summarize(waves, "agg_tps"),
            "per_stream_decode_median": statistics.median(
                [x for w in waves for x in w["per_stream_decode"] if x]),
            "waves": waves}
    with open(out.replace(".jsonl", "") + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for k, v in summary["phases"].items():
        if "agg_tps" in v:
            print(f"{k:16s} agg {v['agg_tps']['median']:.2f} tok/s (min {v['agg_tps']['min']:.2f} max {v['agg_tps']['max']:.2f}) "
                  f"per-stream {v['per_stream_decode_median']:.2f}  {v['start']}..{v['end']}")
        else:
            d = v["decode_tps"]
            print(f"{k:16s} decode {d['median']:.2f} tok/s (min {d['min']:.2f} max {d['max']:.2f} sd {d['stdev']:.2f}) "
                  f"ttft {v['ttft_s']['median']:.3f}s toks {v['completion_tokens']} distinct {v['distinct_outputs']} "
                  f"err {len(v['errors'])}  {v['start']}..{v['end']}")


if __name__ == "__main__":
    main()
