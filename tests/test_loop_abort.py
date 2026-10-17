#!/usr/bin/env python3
"""Host-side tests for the decode-side n-gram / repeated-line abort."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from loop_abort import (  # noqa: E402
    DEFAULT_IDENTICAL,
    DEFAULT_NGRAM,
    DEFAULT_REPEATS,
    MATCHED_STOP,
    detect_repetition,
    install,
    load_config,
    repeated_line_suffix,
)


def _with_env(mapping, fn):
    old = {k: os.environ.get(k) for k in mapping}
    try:
        for k, v in mapping.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cfg(**kwargs):
    base = dict(
        abort=1, ngram=DEFAULT_NGRAM, ngram_max=256, repeats=DEFAULT_REPEATS,
        identical=DEFAULT_IDENTICAL, line_repeats=8, line_min=16, enabled=True)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_default_config():
    cfg = _with_env({
        'DSV41_LOOP_ABORT': None,
        'DSV41_LOOP_NGRAM': None,
        'DSV41_LOOP_REPEATS': None,
        'DSV41_LOOP_IDENTICAL': None,
        'DSV41_LOOP_LINE_REPEATS': None,
    }, load_config)
    assert cfg.enabled
    assert cfg.ngram == DEFAULT_NGRAM
    assert cfg.repeats == DEFAULT_REPEATS
    assert cfg.identical == DEFAULT_IDENTICAL


def test_abort_zero_disables():
    cfg = _with_env({'DSV41_LOOP_ABORT': '0'}, load_config)
    assert not cfg.enabled
    ids = [7, 8] * 200
    assert detect_repetition(ids, cfg=cfg) is None


def test_ngram_cycle_is_detected():
    unit = list(range(32, 64))
    ids = list(range(10)) + unit * 4
    hit = detect_repetition(ids, cfg=_cfg())
    assert hit is not None
    matched, kept = hit
    assert matched == MATCHED_STOP
    assert kept == 10 + 32  # prefix + one copy of the unit


def test_unique_sequence_is_not_a_loop():
    ids = list(range(400))
    assert detect_repetition(ids, cfg=_cfg()) is None


def test_short_punctuation_repeat_does_not_fire():
    # Four copies of a 6-token "</div>"-sized unit stays under min ngram=32.
    ids = [1, 2, 3, 4, 5, 6] * 4
    assert detect_repetition(ids, cfg=_cfg()) is None


def test_identical_token_run():
    ids = [9] * DEFAULT_IDENTICAL
    hit = detect_repetition(ids, cfg=_cfg())
    assert hit == (MATCHED_STOP, 1)
    # A long collapse keeps one token, not (n - threshold + 1) of the run.
    hit = detect_repetition([3] * 10 + [9] * 500, cfg=_cfg())
    assert hit == (MATCHED_STOP, 11)


def test_three_copies_do_not_fire():
    ids = list(range(32, 64)) * 3
    assert detect_repetition(ids, cfg=_cfg()) is None


def test_identical_zero_disables_run_check():
    ids = [9] * 200
    assert detect_repetition(
        ids, cfg=_cfg(identical=0, ngram=0, repeats=4, line_repeats=0)) is None


def test_repeated_lines():
    line = 'ctx.fillRect(car.x, car.y, 24, 12);'
    text = ('header\n' + (line + '\n') * 8)
    assert repeated_line_suffix(text, 8, 16)
    ids = list(range(80))
    hit = detect_repetition(ids, text=text, cfg=_cfg(ngram=32, repeats=4, identical=0))
    assert hit == (MATCHED_STOP, 80)


def test_short_or_distinct_lines_do_not_fire():
    assert not repeated_line_suffix('a\na\na\na\na\na\na\na\n', 8, 16)
    lines = '\n'.join(f'item {i} is unique enough' for i in range(12))
    assert not repeated_line_suffix(lines, 8, 16)


def test_install_finishes_looping_req():
    class FakeFinish:
        def __init__(self, matched):
            self.matched = matched

        def to_json(self):
            return {'type': 'stop', 'matched': self.matched}

    class FakeReq:
        def __init__(self):
            self.output_ids = list(range(40, 72)) * 4
            self.finished_reason = None
            self.finished_len = None
            self.rid = 't'
            self.tokenizer = None

        def finished(self):
            return self.finished_reason is not None

        def update_finish_state(self, new_accepted_len=1):
            return None

    class FakeMod:
        Req = FakeReq
        FINISH_MATCHED_STR = FakeFinish

    def run():
        install(FakeMod)
        req = FakeReq()
        req.update_finish_state(1)
        return req

    req = _with_env({'DSV41_LOOP_ABORT': '1'}, run)
    assert req.finished()
    assert req.finished_reason.matched == MATCHED_STOP
    assert req.finished_len == 32


def test_install_respects_already_finished():
    class FakeFinish:
        def __init__(self, matched):
            self.matched = matched

    class FakeReq:
        def __init__(self):
            self.output_ids = [1] * 80
            self.finished_reason = FakeFinish('length')
            self.finished_len = 80
            self.tokenizer = None

        def finished(self):
            return self.finished_reason is not None

        def update_finish_state(self, new_accepted_len=1):
            return None

    class FakeMod:
        Req = FakeReq
        FINISH_MATCHED_STR = FakeFinish

    def run():
        install(FakeMod)
        req = FakeReq()
        req.update_finish_state(1)
        return req

    req = _with_env({'DSV41_LOOP_ABORT': '1'}, run)
    assert req.finished_reason.matched == 'length'


def _named_exit_run(env):
    from http import HTTPStatus

    class FakeFinish:
        def __init__(self, matched):
            self.matched = matched

    class FakeAbort:
        def __init__(self, message=None, status_code=None, err_type=None):
            self.message = message
            self.status_code = status_code
            self.err_type = err_type

    class FakeReq:
        def __init__(self):
            self.output_ids = list(range(40, 72)) * 4
            self.finished_reason = None
            self.finished_len = None
            self.rid = 't'
            self.tokenizer = None

        def finished(self):
            return self.finished_reason is not None

        def update_finish_state(self, new_accepted_len=1):
            return None

    class FakeMod:
        Req = FakeReq
        FINISH_MATCHED_STR = FakeFinish
        FINISH_ABORT = FakeAbort

    def run():
        install(FakeMod)
        req = FakeReq()
        req.update_finish_state(1)
        return req

    return _with_env(env, run), FakeFinish, FakeAbort, HTTPStatus


def test_named_exit_is_a_named_error():
    req, _, FakeAbort, HTTPStatus = _named_exit_run({'DSV41_LOOP_ABORT': '1', 'DSV41_LOOP_NAMED_EXIT': None})
    assert isinstance(req.finished_reason, FakeAbort)
    # The streaming error path only fires for an HTTPStatus instance, not an int.
    assert isinstance(req.finished_reason.status_code, HTTPStatus)
    assert req.finished_reason.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert req.finished_reason.err_type == 'loop_abort'
    # err_type does not reach the stream; the name must be in the message itself.
    assert req.finished_reason.message.startswith('loop_abort:')
    assert 'art=ngram' in req.finished_reason.message and 'period=32' in req.finished_reason.message
    assert req.finished_len == 32


def test_named_exit_zero_restores_stop():
    req, FakeFinish, _, _ = _named_exit_run({'DSV41_LOOP_ABORT': '1', 'DSV41_LOOP_NAMED_EXIT': '0'})
    assert isinstance(req.finished_reason, FakeFinish)
    assert req.finished_reason.matched == MATCHED_STOP
    assert req.finished_len == 32


def test_detail_reports_art_and_period():
    from loop_abort import detect_repetition_detail
    cfg = load_config()
    assert detect_repetition_detail(list(range(40, 72)) * 4, cfg=cfg) == ('ngram', 32, 32)
    assert detect_repetition_detail([7] * DEFAULT_IDENTICAL, cfg=cfg) == ('identical', 1, 1)


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[OK]   {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
