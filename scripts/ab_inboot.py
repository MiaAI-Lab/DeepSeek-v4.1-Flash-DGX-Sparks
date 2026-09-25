#!/usr/bin/env python3
"""Drive an in-boot A/B (adapter/ab_variant.py, DSV41_AB_VARIANTS>=2) from the head node's host.

A warm pass runs every prompt once at the full --max-tokens (prefix cache, Engram rows, allocator).
Then rounds, ABBA by default (--abab for plain order). In each round, each variant is switched on
through the flag file (rank 0 broadcasts it; the .ack confirms every TP rank switched, including a
MAX all-reduce of [v, -v, seq, -seq] over the TP group), and the same prompts are run greedy at
c=1, the step_probe.py way: TTFT from a max_tokens=1 call, decode time = total - TTFT,
step_ms = decode time / spec_verify_ct, tok/s = (tokens - 1) / decode time. The first
--drop-rounds (default 1) rounds are run but left out of the statistics.

Reports per variant the median step_ms / tok/s and mean tokens/step, and for every variant vs v0
the difference: per round the mean over prompts of the paired (same prompt) difference, then the
mean over rounds with a 95 % bootstrap CI that resamples ROUNDS (the unit that shares drift).

The run FAILS (exit 1) when a block replays no target or no draft graph from its own set, replays
another variant's set, the ranks disagree after a switch, or (unless --allow-output-diff) greedy
outputs differ between variants: both flag families are meant to be bit-identical.

usage (on the head node, engine booted with DSV41_AB_VARIANTS):
  python3 ab_inboot.py LABEL [--rounds 9] [--drop-rounds 1] [--max-tokens 400]
                             [--prompts builtin|sparkdash|FILE] [--abab] [--allow-output-diff]
                             [--state-dir DIR] [--url URL]
Raw rows: ~/ab_inboot/LABEL-<time>.jsonl; summary appended to ~/ab_inboot/summary.jsonl.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request

BUILTIN = {
    "prose": ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
              "Describe a walk through an autumn forest in vivid detail.",
              "Explain to a child why the moon changes shape during the month."],
    "code": ["Write a Python class implementing an LRU cache with get and put in O(1).",
             "Write a C function that reverses a singly linked list, with comments.",
             "Write a JavaScript debounce function and a usage example."],
}


def load_prompts(spec):
    if spec == "builtin":
        return [(k, p) for k, ps in BUILTIN.items() for p in ps]
    if spec == "sparkdash":
        js = open(os.path.expanduser("~/sparkDash/src/shared/llmPrompts.js")).read()

        def arr(name):
            m = re.search(r"export const " + name + r" = \[(.*?)\n\];", js, re.S)
            return [json.loads(x) for x in re.findall(r"^\s*(\"(?:[^\"\\\\]|\\\\.)*\"),?\s*$", m.group(1), re.M)]

        return [("prose", p) for p in arr("TEXT_PROMPTS")] + [("structural", p) for p in arr("STRUCTURAL_PROMPTS")]
    out = []
    for line in open(spec):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out.append((rec.get("kind", "file"), rec["prompt"]))
    return out


def state_dir_default(container):
    try:
        mounts = json.loads(subprocess.check_output(
            ["docker", "inspect", container, "--format", "{{json .Mounts}}"], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    for m in mounts:
        if m.get("Destination") == "/state":
            return m.get("Source")
    return None


class Engine:
    def __init__(self, url, state_dir, timeout):
        self.url, self.flag, self.timeout = url, os.path.join(state_dir, "ab_variant"), timeout
        self.ack = self.flag + ".ack"

    def read_ack(self):
        try:
            with open(self.ack) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def switch(self, variant):
        ack = self._switch(variant)
        ranks = ack.get("ranks")
        if ranks is not None and not ranks.get("agree"):
            raise RuntimeError(f"ranks disagree after switching to {variant}: {ranks}")
        return ack

    def _switch(self, variant):
        """Write '<variant> <token>' and wait until rank 0 acks that token (every rank switched)."""
        token = f"{os.getpid()}-{time.monotonic_ns()}"
        tmp = f"{self.flag}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(f"{variant} {token}\n")
        os.replace(tmp, self.flag)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            ack = self.read_ack()
            if ack and ack.get("token") == token:
                if ack.get("variant") != variant:
                    raise RuntimeError(f"ack for variant {ack.get('variant')}, wanted {variant}")
                return ack
            time.sleep(0.1)
        raise RuntimeError(f"no ack for variant {variant} in {self.timeout}s at {self.ack}: is the "
                           "engine booted with DSV41_AB_VARIANTS and /state mounted there?")

    def call(self, prompt, max_tokens):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0, "return_spec_tokens_details": True,
                "chat_template_kwargs": {"thinking": False}}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        out = json.load(urllib.request.urlopen(req, timeout=900))
        dt = time.time() - t0
        det = (out.get("sglext") or {}).get("spec_tokens_details") or {}
        msg = out["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
        return dt, out["usage"]["completion_tokens"], det.get("spec_verify_ct"), text


def bootstrap_ci(values, stat, n=5000, seed=12345):
    rng = random.Random(seed)
    k = len(values)
    vals = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def _ok(x):
    return x.get("steps") and x.get("step_ms") is not None and x.get("tps") is not None


def summarize(rows, nvar):
    out = {"variants": {}, "vs_v0": {}}
    for v in range(nvar):
        r = [x for x in rows if x["variant"] == v and _ok(x)]
        if not r:
            continue
        out["variants"][v] = {
            "n": len(r),
            "step_ms_median": round(statistics.median(x["step_ms"] for x in r), 3),
            "tps_median": round(statistics.median(x["tps"] for x in r), 2),
            "tok_per_step_mean": round(statistics.mean(x["tok_per_step"] for x in r), 4),
            "by_kind": {k: {"step_ms_median": round(statistics.median(x["step_ms"] for x in r if x["kind"] == k), 3),
                            "tps_median": round(statistics.median(x["tps"] for x in r if x["kind"] == k), 2)}
                        for k in sorted({x["kind"] for x in r})},
        }
    base = {(x["round"], x["prompt"]): x for x in rows if x["variant"] == 0 and _ok(x)}
    for v in range(1, nvar):
        pairs = [(base[(x["round"], x["prompt"])], x) for x in rows
                 if x["variant"] == v and _ok(x) and (x["round"], x["prompt"]) in base]
        rounds = sorted({b["round"] for _, b in pairs})
        if len(rounds) < 2:
            out["vs_v0"][v] = {"pairs": len(pairs), "rounds": len(rounds), "error": "fewer than 2 rounds"}
            continue
        res = {"pairs": len(pairs), "rounds": len(rounds)}
        for metric in ("step_ms", "tps"):
            d = [b[metric] - a[metric] for a, b in pairs]
            per_round = [statistics.mean(b[metric] - a[metric] for a, b in pairs if b["round"] == r) for r in rounds]
            rel_round = [statistics.mean((b[metric] - a[metric]) / a[metric] * 100 for a, b in pairs if b["round"] == r)
                         for r in rounds]
            lo, hi = bootstrap_ci(per_round, statistics.mean)
            rlo, rhi = bootstrap_ci(rel_round, statistics.mean)
            res[metric] = {"mean_diff": round(statistics.mean(per_round), 4), "ci95_rounds": [round(lo, 4), round(hi, 4)],
                           "per_round": [round(x, 4) for x in per_round],
                           "median_pair_diff": round(statistics.median(d), 4),
                           "mean_rel_pct": round(statistics.mean(rel_round), 3), "rel_ci95_rounds": [round(rlo, 3), round(rhi, 3)]}
        same = sum(a["text_sha"] == b["text_sha"] for a, b in pairs)
        res["identical_outputs"] = f"{same}/{len(pairs)}"
        res["output_diffs"] = len(pairs) - same
        res["tok_per_step_diff"] = round(statistics.mean(b["tok_per_step"] - a["tok_per_step"] for a, b in pairs), 4)
        out["vs_v0"][v] = res
    return out


def check_replays(delta, v, nvar):
    """Problems with one block's replays: its own target and draft sets must both have run, and no
    other variant's set (shared shapes, index nvar, are allowed)."""
    bad = []
    for kind in ("target", "draft"):
        counts = delta.get(kind)
        if not counts or counts[v] <= 0:
            bad.append(f"no {kind} replays from set {v}")
        elif any(n for j, n in enumerate(counts[:nvar]) if j != v):
            bad.append(f"{kind} replayed another set: {counts}")
    return bad


def replay_delta(a, b, nvar):
    """Replays per graph set between two acks; index nvar = shapes shared (captured once)."""
    out = {}
    for kind, counts in (b.get("replays") or {}).items():
        prev = (a.get("replays") or {}).get(kind, [0] * len(counts))
        out[kind] = [c - p for c, p in zip(counts, prev)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--drop-rounds", type=int, default=1, help="leading rounds left out of the statistics")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--prompts", default="builtin")
    ap.add_argument("--abab", action="store_true", help="same order every round (default: ABBA...)")
    ap.add_argument("--allow-output-diff", action="store_true", help="do not fail on differing greedy outputs")
    ap.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    ap.add_argument("--container", default="dsv41-head")
    ap.add_argument("--state-dir", default=None, help="host dir mounted at /state (default: docker inspect)")
    ap.add_argument("--ack-timeout", type=float, default=20.0)
    args = ap.parse_args()

    state_dir = args.state_dir or state_dir_default(args.container)
    if not state_dir:
        sys.exit("cannot find the /state mount; pass --state-dir")
    eng = Engine(args.url, state_dir, args.ack_timeout)
    prompts = load_prompts(args.prompts)
    outdir = os.path.expanduser("~/ab_inboot")
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")

    ack = eng.switch(0)
    nvar = int(ack["variants"])
    print(f"armed: {nvar} variants, config {ack['config']}, {len(prompts)} prompts, rounds {args.rounds}", flush=True)
    for _, p in prompts:                       # warm at full length, off the clock
        eng.call(p, 1)
        eng.call(p, args.max_tokens)

    rows = []
    try:
        with open(raw_path, "w") as raw:
            for rnd in range(args.rounds):
                order = list(range(nvar))
                if not args.abab and rnd % 2:
                    order.reverse()
                for v in order:
                    before = eng.switch(v)
                    for i, (kind, p) in enumerate(prompts):
                        ttft = eng.call(p, 1)[0]
                        total, toks, ct, text = eng.call(p, args.max_tokens)
                        dec = total - ttft
                        row = {"round": rnd, "variant": v, "prompt": i, "kind": kind, "ttft": round(ttft, 4),
                               "decode_s": round(dec, 4), "tokens": toks, "steps": ct,
                               "step_ms": round(dec / ct * 1000, 3) if ct else None,
                               "tps": round((toks - 1) / dec, 2) if ct and dec > 0 else None,
                               "tok_per_step": round((toks - 1) / ct, 4) if ct else None,
                               "text_sha": hashlib.sha1(text.encode()).hexdigest()[:12],
                               "t": round(time.time(), 2), "text": text}  # full text: locate a divergence
                        rows.append(row)
                        raw.write(json.dumps(row) + "\n")
                    after = eng.switch(v)      # re-ack: replay counters for this block
                    delta = replay_delta(before, after, nvar)
                    bad = check_replays(delta, v, nvar)
                    blk = [r for r in rows if r["round"] == rnd and r["variant"] == v and _ok(r)]
                    stats = (f"step_ms {statistics.median(r['step_ms'] for r in blk):.2f} "
                             f"tps {statistics.median(r['tps'] for r in blk):.1f}" if blk else "NO VALID ROWS")
                    print(f"round {rnd} v{v}: {stats} replays {delta}" + (f"  FAIL {bad}" if bad else ""), flush=True)
                    raw.write(json.dumps({"round": rnd, "variant": v, "replays": delta, "problems": bad}) + "\n")
                    if bad:
                        raise RuntimeError(f"round {rnd} v{v}: {bad}")
                    if not blk:
                        raise RuntimeError(f"round {rnd} v{v}: no request returned spec_verify_ct / tokens")
    finally:
        try:
            eng.switch(0)
        except RuntimeError as exc:
            print(f"could not restore variant 0: {exc}", flush=True)

    kept = [r for r in rows if r["round"] >= args.drop_rounds]
    summary = {"label": args.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "raw": raw_path,
               "config": ack["config"], "rounds": args.rounds, "dropped_rounds": args.drop_rounds,
               "max_tokens": args.max_tokens, "prompts": args.prompts, "abba": not args.abab,
               **summarize(kept, nvar)}
    # greedy outputs over every round, dropped ones included
    diffs_all = sum(r.get("output_diffs", 0) for r in summarize(rows, nvar)["vs_v0"].values())
    summary["output_diffs_all_rounds"] = diffs_all
    failed = (diffs_all and not args.allow_output_diff) or any("error" in r for r in summary["vs_v0"].values())
    summary["verdict"] = "FAIL" if failed else "ok"
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.join(outdir, "summary.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    if failed:
        sys.exit(f"FAIL: {diffs_all} greedy output difference(s) between variants" if diffs_all
                 else "FAIL: not enough rounds for a CI")


if __name__ == "__main__":
    main()
