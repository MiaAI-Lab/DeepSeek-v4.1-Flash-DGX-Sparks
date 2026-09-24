# Bringing the TP3 decode gains to the 4-Spark (TP4) profile

For whoever runs the 4-Spark fleet. On 3 Sparks, the 2026-09-24 campaign
([overnight-results.md](overnight-results.md)) gained C1 prose +17 %, code +39 %, sampled +29 % and
C4 +19 %. This plan tries the same stack on TP4 in **3 to 4 boots** (about 15 min each), plus one
measurement of the current configuration that needs no reboot.

## Why so few boots

- The adapters were already measured one at a time on TP3, and knapcio runs the same set on TP4.
  Here they go on together in one boot (arm A) instead of one boot each.
- Two open questions remain, and only they get their own boots:
  - does the stack help at TP4's memory settings (0.80 fraction, 8M KV, 8 slots)?
  - which `EP_SIZE` is best? EP1 was the largest single gain on TP3; knapcio found EP2 beats EP4 on TP4.
- The baseline is measured on whatever TP4 is serving now, and packing runs while it serves.
- `DSV41_AUTOTUNE_KEEP=1` in every arm: a rebooted arm reuses its tuned kernels, and results
  repeat across boots.

**The TP3 and TP4 profiles share spark1-3 and the same container names:** every arm's boot stops
whatever is serving on them, including the TP3 stack.

Everything below runs on spark1 from the repository root. Code changes (adapter fixes,
`start.sh` forwarding) are already in the image; only `.env.tp4` values differ between arms.

## 0. Once: align images, build, pack (no reboot of the serving stack)

```bash
./start-tp4.sh doctor              # creates .env.tp4 from .env.tp4.example if missing
scripts/tp4/preflight.sh           # ssh to all 4 nodes, same base image, overlay, shards or free disk
```

**Base image:** every result was measured on the Docker Hub digest
`lmsysorg/sglang:dev-dsv41@sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349`
(public, anonymous pull; arm64 image id `37939c26c0ba…`). The plain `dev-dsv41` **tag** has since
moved to a newer build (arm64 id `381b27ff…`), so don't go by the tag. The Dockerfile `FROM` and
`BASE_IMAGE` in `.env.tp4.example` carry the digest. If your `.env.tp4` was created earlier, set:

```bash
BASE_IMAGE=lmsysorg/sglang:dev-dsv41@sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349
```

then `./start-tp4.sh pull` (all nodes; ~33 GB once) and check each node:
`docker image inspect "$BASE_IMAGE" -f '{{.Id}}'` must print `sha256:37939c26c0ba…`.
Rerun `scripts/tp4/preflight.sh`: it checks the image named by `BASE_IMAGE`, so with the pin set it
checks the right one.

```bash
./start-tp4.sh build               # overlay dsv41-4x-spark:local on all 4 nodes
./start-tp4.sh pack                # engram-l{1,14}-r<rank>of4.bin, ~47 GiB per node, ~4 min per node
```

`pack` is safe while TP4 serves: it writes `.partial` files and renames them when complete. It
does add disk and NFS load, so don't run step 1's measurement at the same time. Serving only picks
up the shards at the next boot.

## 1. Baseline, no reboot

With the current TP4 configuration serving (if nothing is serving, `./start-tp4.sh` first):

```bash
scripts/tp4/arm.sh baseline --no-boot
```

This records C1 prose/prose2/code/sampled, C4 and C8, the quality gate and needles at 30k/100k.
The load is sent from spark2 (`BENCH_HOST`), not from the head. Results go to
`docs/results/tp4-<date>/`.

If the running TP4 had no packed shards (check its log for `packed=False`), the baseline includes
that cost; note it when comparing, since packing alone was worth +2-5 % on TP3.

## 2. Arm A: full stack, EP4 (boot 1)

```bash
scripts/tp4/arm.sh a-stack-ep4
```

It sets `wo_a` W8+MID+DROP, draft-head FP8, k=5 with `conf:0.1` and block verify, Engram prefetch
and autotune keep (see `scripts/tp4/arms/a-stack-ep4.env`), then boots, checks, benchmarks and runs
the quality gate. Check `checks-a-stack-ep4.txt`:

- `packed=True` twice on rank 0; also check every worker with `./start-tp4.sh logs worker1..3 | grep packed=`
- `[wo_a_w8] einsum bridge armed` and `released the bf16 copy of … wo_a weights`
- `[wo_a_w8] draft wo_a left on bf16`
- `[verify_cap] confidence present`, `Engram prefetch ARMED`, `draft … folded into the draft cuda graph`
- a `max_total_num_tokens` value (the KV pool)

**Pass:** quality 8/8 and 8/8 concurrent; needles found; C1 prose, code and C4 no worse than
baseline. On TP3 the stack alone (before EP1) gave prose +8 %, code +23 %, C4 +4 %.

**If it fails to boot or regresses:**
1. `scripts/tp4/arm.sh x-no-wo-a`: the stack without the `wo_a` twin. `wo_a` is the only piece
   that changes weight memory.
2. If quality fails: `scripts/tp4/arm.sh x-prefetch-check`, then look for
   `prefetch CHECK: … = [0, 0]` on every rank (anything nonzero is a bug; its timing is not valid).
3. Stop here and report. Don't continue to the EP arms.

## 3. Arm B: EP2 (boot 2)

```bash
scripts/tp4/arm.sh b-stack-ep2
```

## 4. Arm C: EP1 (boot 3, optional)

Run it if B beat A, or if B was within noise. EP1 won on TP3 because EP3 left ranks waiting on
each other's experts; with 4 ranks the gap to EP2 may be smaller.

```bash
scripts/tp4/arm.sh c-stack-ep1
```

## 5. Keep the winner (boot 4, only if the winner is not the arm serving now)

Copy the winning arm's values into `.env.tp4` (the arm file lists exactly what changed), then
`./start-tp4.sh`. The boot reuses that arm's tuned kernels. Re-run `scripts/tp4/arm.sh final --no-boot`
to confirm the numbers repeat, and add `NEEDLES=30,200,500 scripts/tp4/arm.sh final --no-boot` for long
context (the profile advertises 1M; the previous 1M needle pass predates these changes).
Then update `.env.tp4.example` and the README's TP4 row, and open a PR with `docs/results/tp4-<date>/`.

## Reading the numbers

- Compare arms on C1 prose, code and C4/C8 **and** on `step` in `bench-*.txt`. The step (ms per
  speculative step) is steadier than tok/s: prose tok/s moves a few % when the greedy text changes.
- Workloads and metric definitions: `benchmarks/overnight_bench.py`. Token counts come from the
  server's `usage` field, not SSE events.
- Rollback at any point: `./start-tp4.sh` with the original `.env.tp4` (arms never modify it; they
  write copies under `docs/results/tp4-<date>/envs/`, which is git-ignored).
