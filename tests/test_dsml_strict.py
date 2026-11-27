#!/usr/bin/env python3
"""Host-side tests for strict nonstream DSML tool batches (#51). No GPU, network or tool runs."""
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

import dsml_strict  # noqa: E402

D = '｜DSML｜'


class Result:
    def __init__(self, normal_text='', calls=()):
        self.normal_text, self.calls = normal_text, list(calls)


class NativeV41:
    """Mirror of DeepSeekV41Detector's nonstream path (sglang f80c91a4b, deepseekv32_detector.py)."""
    dsml_token, tool_calls_block_name, invoke_tag_name, parameter_tag_name = D, ' calls', ' invoke', ' parameter'
    bot_token, eot_token = f'<{D} calls>', f'</{D} calls>'
    invoke_start_token, invoke_end_token = f'<{D} invoke', f'</{D} invoke>'

    def detect_and_parse(self, text, tools):
        idx = text.find(self.bot_token)
        if idx == -1:
            return Result(text)
        names, calls = [t.function.name for t in tools], []
        for block in re.findall(rf'<{D} calls>(.*?)</{D} calls>', text, re.DOTALL):
            for m in re.finditer(rf'<{D} invoke\s+name="([^"]+)"\s*(?:/>|>(.*?)(?:</{D} invoke>|$))', block, re.DOTALL):
                body = (m.group(2) or '').strip()
                args = json.loads(body) if body.startswith('{') and body.endswith('}') else {
                    k: v.strip() if s == 'true' else json.loads(v) for k, s, v in re.findall(
                        rf'<{D} parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</{D} parameter>', body, re.DOTALL)}
                if m.group(1).strip() in names:
                    calls.append(NS(tool_index=names.index(m.group(1).strip()), name=m.group(1).strip(),
                                    parameters=json.dumps(args, ensure_ascii=False)))
        return Result(text[:idx].removesuffix('\n\n'), calls)

    def parse_streaming_increment(self, new_text, tools):
        return Result(new_text)


STOCK = NativeV41.detect_and_parse
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['value', 'count'],
          'properties': {'value': {'type': 'string'}, 'count': {'type': 'integer'}}}
TOOLS = [NS(function=NS(name='record_sample', parameters=SCHEMA)),
         NS(function=NS(name='loose', parameters={'type': 'object', 'properties': {'value': {}}}))]
NI, NIE, NP, NPE = f'<{D} invoke name=', f'</{D} invoke>', f'<{D} parameter name=', f'</{D} parameter>'
CALLS, CALLS_END = NativeV41.bot_token, NativeV41.eot_token
FIRST = f'{NI}"record_sample">{NP}"value">cedar-42{NPE}{NP}"count">7{NPE}{NIE}'
PLAIN = '<invoke name="record_sample"><parameter name="value">second</parameter><parameter name="count">8</parameter></invoke>'
ISSUE = FIRST + '\n' + PLAIN  # the reproduction in issue #51
EXPECTED = [{'value': 'cedar-42', 'count': 7}, {'value': 'second', 'count': 8}]


def _install(env=None):
    NativeV41.detect_and_parse = STOCK
    old = os.environ.pop('DSV41_DSML_STRICT', None)
    os.environ.update({'DSV41_DSML_STRICT': env} if env else {})
    try:
        dsml_strict.install(NS(DeepSeekV41Detector=NativeV41), result_cls=Result)
    finally:
        os.environ.pop('DSV41_DSML_STRICT', None)
        os.environ.update({'DSV41_DSML_STRICT': old} if old is not None else {})
    return NativeV41()


def _args(result):
    return [json.loads(c.parameters) for c in result.calls]


def test_issue_shapes_give_both_calls():
    assert STOCK(NativeV41(), ISSUE, TOOLS).calls == []  # stock regression
    det = _install()
    json_bodies = (f'{CALLS}\n{NI}"record_sample">{{"value": "cedar-42", "count": 7}}{NIE}\n'
                   '<invoke name="record_sample">{"value": "second", "count": 8}</invoke>\n' + CALLS_END)
    for text in (ISSUE, f'{CALLS}\n{ISSUE}\n{CALLS_END}', json_bodies, f'{ISSUE}\n{CALLS_END}'):
        for prose in ('', 'Checking.\n\n'):
            out = det.detect_and_parse(prose + text, TOOLS)
            assert _args(out) == EXPECTED, text
            assert out.normal_text == prose.strip() and [c.tool_index for c in out.calls] == [0, 0]


def test_native_batch_unchanged():
    text = (f'Hi\n\n{CALLS}\n{NI}"record_sample">{NP}"value" string="true">a <b>{NPE}'
            f'{NP}"count" string="false">3{NPE}{NIE}\n{CALLS_END}')
    stock, strict = STOCK(NativeV41(), text, TOOLS), _install().detect_and_parse(text, TOOLS)
    assert _args(strict) == _args(stock) == [{'value': 'a <b>', 'count': 3}]
    assert strict.normal_text == stock.normal_text == 'Hi'


def test_invalid_batches_are_atomic_and_recover():
    det = _install()
    bad = {
        'duplicate key': f'{NI}"record_sample">{{"value": "a", "value": "b", "count": 1}}{NIE}',
        'duplicate parameter': FIRST.replace(f'{NP}"count">7', f'{NP}"value">x{NPE}{NP}"count">7'),
        'unknown tool': ISSUE.replace('<invoke name="record_sample"', '<invoke name="drop_table"'),
        'wrong type': FIRST + '\n' + PLAIN.replace('>8<', '>"8"<'),
        'not json': FIRST + '\n' + PLAIN.replace('>8<', '>eight<'),
        'missing required': FIRST.replace(f'{NP}"count">7{NPE}', ''),
        'extra argument': f'{NI}"record_sample">{{"value": "a", "count": 1, "x": 0}}{NIE}',
        'non-finite': f'{NI}"record_sample">{{"value": "a", "count": NaN}}{NIE}',
        'truncated': FIRST + '\n' + PLAIN[:-9],
        'mismatched close': FIRST.replace(NIE, '</invoke>'),
        'markup in value': FIRST.replace('cedar-42', f'x{NP}"count">1'),
        'unclosed wrapper': f'{CALLS}\n{ISSUE}',
        'trailing prose': ISSUE + '\nDone.',
        'ambiguous type': f'{NI}"loose">{NP}"value">1{NPE}{NIE}',
    }
    for why, text in bad.items():
        out = det.detect_and_parse(text, TOOLS)
        assert out.calls == [] and out.normal_text == text, why
        assert _args(det.detect_and_parse(ISSUE, TOOLS)) == EXPECTED, f'recovery after {why}'


def test_streaming_untouched_disabled_and_idempotent():
    det = _install()
    first = NativeV41.detect_and_parse
    assert NativeV41.parse_streaming_increment.__qualname__ == 'NativeV41.parse_streaming_increment'
    dsml_strict.install(NS(DeepSeekV41Detector=NativeV41), result_cls=Result)
    assert NativeV41.detect_and_parse is first
    assert det.detect_and_parse('no tools here', TOOLS).normal_text == 'no tools here'
    _install('0')
    assert NativeV41.detect_and_parse is STOCK


def test_schema_subset():
    schema = {'$defs': {'n': {'type': 'integer'}}, 'properties': {'n': {'$ref': '#/$defs/n'}, 'k': {
        'enum': ['a', 'b']}, 'v': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}}}
    dsml_strict.check({'n': 2, 'k': 'a', 'v': None}, schema)
    passed = [v for v in ({'n': True}, {'n': 1.5}, {'k': 'c'}, {'v': 3}) if dsml_strict._passes(v, schema)]
    assert passed == [], passed


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[OK]   {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {exc!r}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
