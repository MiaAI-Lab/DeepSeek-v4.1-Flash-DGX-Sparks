# Changelog

## Unreleased (2026-10-17 triage)

Ported from open pull requests with authorship preserved (`git log` shows the original authors):

- **GanyX19**, #55 / #54: loop abort ends a request with a named error instead of `finish_reason=stop`, and broken tool calls are repaired in the history (`adapter/loop_abort.py`, `history_repair`). Behaviour change: a loop abort is now an HTTP 500 / stream error event for every request, not a partial 200. `DSV41_LOOP_NAMED_EXIT=0` restores the old behaviour.
- **100menotu001**, #47: Anthropic Messages accepts `thinking: {"type":"enabled"}` without `budget_tokens`. #46: `start.sh` derives a missing `.pub`, matches `authorized_keys` by key, and names a missing image in `share`.
- **hushengkai**, #15: the image placeholder token in message text no longer raises (`/v1/messages` 500 loop).
- **BHCC2025**, #26: `pack` re-shares the checkpoint when a worker cannot see it.
- **Rusty's AI Lab (RustyAiLab)**, #5: `scripts/remote.py` honours `--env-file` for argument defaults.
- **njs-io**, #42: `download` added to the quick start.
- **colonel-otto**, #49: per-host SSH users (empty `WORKER_USER`) and the second-ring reproduction notes in `docs/switchless-ring.md`.
- **Rhys Jones (rhys101)**, #39 (subset): shared-expert K padding follows `TP_SIZE`; TP8 with `EP_SIZE=1` is refused up front. The rest of #39 is not merged.
- **Zbigniew Majewski (knapcio)**, #7 (partial): warning that the DSpark SPS table cannot be used with Engram.

Fixes from issues: #6/#12 the SPS table is ignored on this Engram model unless `DSPARK_SPS_ALLOW_ENGRAM=1` (reported by ecohash-co and hushengkai); #31 `--fp8-gemm-backend` is kept when `EXTRA_SGLANG_ARGS` is set (reported by motzudao).

Docs: switched-fabric notes (#2, #38), multi-session 1M chat and stale TP4 comments (#43, #27, #35), switchless-ring field findings (#40, #41), host-memory fragmentation at load (#23).
