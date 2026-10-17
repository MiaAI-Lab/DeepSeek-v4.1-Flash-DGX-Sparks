#!/usr/bin/env python3
"""Host-side tests for keeping broken assistant tool calls in the history."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

import history_repair  # noqa: E402
from history_repair import REPAIR_KEY, install, make_repairing_parser  # noqa: E402


def stock_parse(arguments):
    """Mirror of serving_chat.parse_tool_call_arguments (sglang, cad252b image)."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError('Assistant tool call function.arguments must be valid JSON.') from exc
    if not isinstance(parsed, dict):
        raise ValueError('Assistant tool call function.arguments must be a JSON object.')
    return parsed


def stock_normalize(message, *, strict=True):
    for item in message['tool_calls']:
        function = item['function']
        try:
            function['arguments'] = MOD.parse_tool_call_arguments(function['arguments'])
        except ValueError:
            if strict:
                raise


MOD = SimpleNamespace(parse_tool_call_arguments=stock_parse)

# Synthetic shapes of truncated arguments: open string, cut list, cut escape.
BROKEN = [
    '{"cmd": "cd /work && ls -la ; echo \'--- a ---\' ; cat a.txt ; cat b',
    '{"items": ["a", "b", "c',
    '{"items": [{"title": "a", "priority": "high"}, {"title": "b\\',
]


def _fresh(env=None):
    MOD.parse_tool_call_arguments = stock_parse
    old = os.environ.get('DSV41_HISTORY_REPAIR')
    try:
        if env is None:
            os.environ.pop('DSV41_HISTORY_REPAIR', None)
        else:
            os.environ['DSV41_HISTORY_REPAIR'] = env
        install(MOD)
    finally:
        if old is None:
            os.environ.pop('DSV41_HISTORY_REPAIR', None)
        else:
            os.environ['DSV41_HISTORY_REPAIR'] = old


def test_valid_object_unchanged():
    _fresh()
    assert MOD.parse_tool_call_arguments('{"cmd": "ls", "n": 2}') == {'cmd': 'ls', 'n': 2}


def test_broken_json_is_kept_complete():
    _fresh()
    for raw in BROKEN:
        out = MOD.parse_tool_call_arguments(raw)
        assert out == {REPAIR_KEY: raw}


def test_non_object_is_wrapped():
    _fresh()
    for raw in ('[]', '"x"', '3'):
        assert MOD.parse_tool_call_arguments(raw) == {REPAIR_KEY: raw}


def test_history_message_no_longer_raises():
    _fresh()
    msg = {'role': 'assistant', 'tool_calls': [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'exec_command', 'arguments': BROKEN[0]}},
        {'id': 'c2', 'type': 'function', 'function': {'name': 'exec_command', 'arguments': '{"cmd": "pwd"}'}}]}
    stock_normalize(msg)
    assert msg['tool_calls'][0]['function']['arguments'] == {REPAIR_KEY: BROKEN[0]}
    assert msg['tool_calls'][1]['function']['arguments'] == {'cmd': 'pwd'}


def test_disabled_keeps_stock_error():
    _fresh('0')
    try:
        MOD.parse_tool_call_arguments(BROKEN[0])
    except ValueError:
        return
    raise AssertionError('DSV41_HISTORY_REPAIR=0 must keep the stock ValueError')


def test_install_is_idempotent():
    _fresh()
    first = MOD.parse_tool_call_arguments
    install(MOD)
    assert MOD.parse_tool_call_arguments is first


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
