#!/usr/bin/env python3
"""Host-side tests for the OpenAI serving harness helpers."""
import importlib.util
from pathlib import Path
import io
import itertools
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import decode_window as decode

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "openai_serving", ROOT / "benchmarks" / "openai_serving.py"
)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_env_url_strips_slash(monkeypatch=None):
    import os
    old = os.environ.get("BASE_URL")
    os.environ["BASE_URL"] = "http://127.0.0.1:8888/"
    try:
        assert mod.env_url() == "http://127.0.0.1:8888"
    finally:
        if old is None:
            os.environ.pop("BASE_URL", None)
        else:
            os.environ["BASE_URL"] = old


def test_prose_topics_cover_c8():
    assert len(mod.PROSE) >= 8


class Response(io.BytesIO):
    status = 200


def test_benchmark_auth_header_and_no_secret_in_result():
    calls = []
    def urlopen(req, timeout):
        calls.append(req)
        return Response(b'{"choices": [{"message": {"content": "42"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}')
    with patch.dict(os.environ, {"API_KEY": "fixture-token"}), patch.object(mod.urllib.request, "urlopen", urlopen):
        row = mod.chat("http://127.0.0.1:8888", [], max_tokens=32)
    assert calls[0].get_header("Authorization") == "Bearer fixture-token"
    assert "fixture-token" not in json.dumps(row)
    with patch.dict(os.environ, {"API_KEY": "off"}):
        assert "Authorization" not in mod.request_headers()


def stream(events, done=True):
    data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return Response(data + (b"data: [DONE]\n\n" if done else b""))


def test_stream_ttft_uses_first_content_and_labels_window_estimate():
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "x" * 200}}]},
        {"choices": [{"delta": {"content": "y" * 500}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 700}},
    ]
    ticks = itertools.count()
    with patch.object(decode.urllib.request, "urlopen", lambda *a, **k: stream(events)), patch.object(decode.time, "monotonic", lambda: next(ticks)):
        row = decode.stream_one(0, 1)
    assert row["ttft_s"] == 2
    assert row["window_129_641_tps"] == 512
    assert row["window_method"] == "character_scaled_estimate"
    assert row["completion_tokens"] == 700


def test_stream_rejects_errors_truncation_and_missing_usage():
    finished = {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]}
    usage = {"choices": [], "usage": {"completion_tokens": 2}}
    for events, done in (([{"error": {"message": "failed"}}], True), ([finished, usage], False), ([finished], True)):
        with patch.object(decode.urllib.request, "urlopen", lambda *a, **k: stream(events, done)):
            try:
                decode.stream_one(0, 1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("broken stream counted as success")


def test_loads_rejects_unknown_idle_state():
    with patch.object(decode.urllib.request, "urlopen", lambda *a, **k: Response(b'{"loads": []}')):
        try:
            decode.require_idle()
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing counters treated as idle")


def test_raw_benchmark_aggregates_match_final_usage():
    report = json.loads((ROOT / "docs/results/tp4-switchless-ring-20260915.json").read_text())
    for wave in report["decode_window_idle"]["waves"]:
        assert abs(wave["aggregate_tok_s"] - wave["completion_tokens"] / wave["wall_seconds"]) < 0.02
        assert wave["successes"] == wave["concurrency"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[OK]   {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
