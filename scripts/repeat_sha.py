#!/usr/bin/env python3
"""Greedy repeatability on a running engine (test tool): the same prompts N times, same call pattern as
ab_inboot.py (max_tokens=1 call, then a max_tokens=N call; temperature 0, thinking off). Saves every text,
then per prompt reports distinct outputs, their counts and, for each non-canonical one, the first divergent
character with context and the steps / tokens of both. Works on any boot (no harness needed);
with --variant V on a DSV41_AB_VARIANTS boot it first pins graph set V (flag file + ack).

usage (head node): repeat_sha.py LABEL [--prompts prose|builtin|sparkdash-prose] [--repeats 100]
                                      [--max-tokens 400] [--variant V] [--logprobs]
Rows: ~/ab_inboot/repeat-LABEL-<time>.jsonl. Exit 1 if any prompt has more than one distinct output.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ab_inboot.py

PROSE = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
         "Describe a walk through an autumn forest in vivid detail.",
         "Explain to a child why the moon changes shape during the month."]
CODE = ["Write a Python class implementing an LRU cache with get and put in O(1).",
        "Write a C function that reverses a singly linked list, with comments.",
        "Write a JavaScript debounce function and a usage example."]
URL = "http://127.0.0.1:8888/v1/chat/completions"


def call(prompt, max_tokens, logprobs):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "return_spec_tokens_details": True, "chat_template_kwargs": {"thinking": False}}
    if logprobs:
        body.update(logprobs=True, top_logprobs=2)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    ch = out["choices"][0]
    msg = ch["message"]
    det = (out.get("sglext") or {}).get("spec_tokens_details") or {}
    return {"dt": round(time.time() - t0, 4), "tokens": out["usage"]["completion_tokens"],
            "steps": det.get("spec_verify_ct"), "finish": ch.get("finish_reason"),
            "text": (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or ""),
            "logprobs": ch.get("logprobs") if logprobs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--prompts", default="prose", choices=["prose", "builtin", "sparkdash-prose"])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--variant", type=int, default=None)
    ap.add_argument("--logprobs", action="store_true", help="top-2 logprobs (may change the engine path)")
    args = ap.parse_args()
    if args.prompts == "prose":
        prompts = PROSE
    elif args.prompts == "builtin":
        prompts = PROSE + CODE
    else:
        from ab_inboot import load_prompts
        prompts = [p for k, p in load_prompts("sparkdash") if k == "prose"]
    if args.variant is not None:
        from ab_inboot import Engine, state_dir_default
        ack = Engine(URL, state_dir_default("dsv41-head"), 20.0).switch(args.variant)
        print(f"pinned graph set {args.variant} (config {ack['config']})", flush=True)
    path = os.path.expanduser(f"~/ab_inboot/repeat-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for p in prompts:                                  # warm, off the record
        call(p, args.max_tokens, args.logprobs)
    rows = []
    with open(path, "w") as f:
        for r in range(args.repeats):
            for i, p in enumerate(prompts):            # interleaved: time-local causes hit every prompt
                call(p, 1, False)
                row = dict(call(p, args.max_tokens, args.logprobs), rep=r, prompt=i, t=time.time())
                row["sha"] = hashlib.sha1(row["text"].encode()).hexdigest()[:12]
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            if r % 10 == 9:
                print(f"rep {r + 1}/{args.repeats}", flush=True)
    bad = 0
    for i, p in enumerate(prompts):
        rs = [r for r in rows if r["prompt"] == i]
        c = collections.Counter(r["sha"] for r in rs)
        canon_sha = c.most_common(1)[0][0]
        canon = next(r for r in rs if r["sha"] == canon_sha)
        print(f"prompt {i}: {len(c)} distinct over {len(rs)} runs {dict(c)}")
        for r in rs:
            if r["sha"] == canon_sha:
                continue
            bad += 1
            a, b = canon["text"], r["text"]
            k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"  rep {r['rep']} at {time.strftime('%H:%M:%S', time.localtime(r['t']))}: first diff at char "
                  f"{k}/{len(a)} ({k / max(len(a), 1):.0%}); canon {canon['tokens']} tok {canon['steps']} steps, "
                  f"this {r['tokens']} tok {r['steps']} steps")
            print(f"    canon: ...{a[max(0, k - 60):k]!r} | {a[k:k + 40]!r}")
            print(f"    this : ...{b[max(0, k - 60):k]!r} | {b[k:k + 40]!r}")
    print(f"{bad} non-canonical outputs of {len(rows)}; rows in {path}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
