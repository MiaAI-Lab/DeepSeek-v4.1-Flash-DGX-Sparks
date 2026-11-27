"""Strict nonstream parsing of complete DSML tool-call batches (issue #51).

V4.1 can mix native ``<｜DSML｜ invoke>`` and plain ``<invoke>`` tags, omit ``string`` or skip the
``calls`` wrapper; stock nonstream ``detect_and_parse`` drops those calls. From the first native
``calls``/``invoke`` opening, the rest of the text must be one batch: optional wrapper, invokes
closed by their own tag kind (JSON object or parameter tags), optional ``calls`` end, whitespace.
A missing ``string`` comes from a single-name schema property type. Tools must be offered and
arguments must pass the schema (type, enum, const, required, properties, additionalProperties,
items, anyOf, local $ref). A valid batch is rewritten as native JSON DSML for the stock
detector; anything else stays text with no calls. Streaming is untouched.
``DSV41_DSML_STRICT=0`` keeps the stock behaviour.
"""
import json
import logging
import math
import os
import re

logger = logging.getLogger(__name__)


class Reject(ValueError):
    """The text is not one complete, valid tool-call batch."""


def enabled():
    return os.environ.get('DSV41_DSML_STRICT', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def _pairs(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise Reject('duplicate JSON key')
    return dict(pairs)


def _finite(raw):
    if not math.isfinite(float(raw)):
        raise Reject(f'non-finite number {raw}')
    return float(raw)


_JSON = json.JSONDecoder(object_pairs_hook=_pairs, parse_float=_finite, parse_constant=_finite)


def _loads(raw, pos=0):
    try:
        return _JSON.raw_decode(raw, pos)
    except json.JSONDecodeError as exc:
        raise Reject(f'invalid JSON ({exc.msg})') from None


_TYPES = {
    'object': lambda v: isinstance(v, dict), 'array': lambda v: isinstance(v, list),
    'string': lambda v: isinstance(v, str), 'boolean': lambda v: isinstance(v, bool),
    'null': lambda v: v is None, 'number': lambda v: type(v) in (int, float),
    'integer': lambda v: type(v) is int or (type(v) is float and v.is_integer()),
}


def _same(a, b):
    return a == b and type(a) is type(b)


def _passes(value, schema, root=None):
    try:
        check(value, schema, root)
    except Reject:
        return False
    return True


def check(value, schema, root=None):
    """Raise Reject unless value satisfies the JSON-schema subset named in the module doc."""
    root = schema if root is None else root
    if schema is True or schema == {}:
        return
    if not isinstance(schema, dict):
        raise Reject('schema refuses the value')
    if '$ref' in schema:
        target = root
        for part in str(schema['$ref']).removeprefix('#/').split('/'):
            if not str(schema['$ref']).startswith('#/') or not isinstance(target, dict) or part not in target:
                raise Reject(f'unresolved $ref {schema["$ref"]!r}')
            target = target[part]
        check(value, target, root)
    kinds = schema.get('type')
    kinds = kinds if isinstance(kinds, list) else [kinds] if kinds else []
    if kinds and not any(_TYPES.get(k, lambda v: False)(value) for k in kinds):
        raise Reject(f'expected {"/".join(map(str, kinds))}')
    if 'enum' in schema and not any(_same(value, e) for e in schema['enum']):
        raise Reject('value not in enum')
    if 'const' in schema and not _same(value, schema['const']):
        raise Reject('value differs from const')
    if 'anyOf' in schema and not any(_passes(value, sub, root) for sub in schema['anyOf']):
        raise Reject('no anyOf branch matches')
    if isinstance(value, dict):
        props, extra = schema.get('properties') or {}, schema.get('additionalProperties', True)
        missing = [k for k in schema.get('required', []) if k not in value]
        if missing:
            raise Reject(f'missing required {missing}')
        for key, item in value.items():
            if key not in props and extra is False:
                raise Reject(f'unexpected argument {key!r}')
            check(item, props.get(key, extra if isinstance(extra, dict) else {}), root)
    if isinstance(value, list) and isinstance(schema.get('items'), dict):
        for item in value:
            check(item, schema['items'], root)


class _Batch:
    """One-shot parser for the text from the first native opening to the end."""

    def __init__(self, det, tools):
        names = [(det.dsml_token + n, n.strip()) for n in (
            det.tool_calls_block_name, det.invoke_tag_name, det.parameter_tag_name)]
        calls, invoke, param = ('|'.join(map(re.escape, pair)) for pair in names)
        self.bot = det.bot_token
        self.ends = [f'</{n}>' for n in names[0]]
        self.invoke = re.compile(rf'\s*<(?P<tag>{invoke})\s+name="(?P<name>[^"<>]+)"\s*(?P<sc>/?)>')
        self.param = re.compile(
            rf'\s*<(?P<tag>{param})\s+name="(?P<name>[^"<>]+)"(?:\s+string="(?P<string>[^"]*)")?\s*>')
        self.markup = re.compile(rf'</?(?:{calls}|{invoke}|{param})(?=[\s>/])')
        self.tools = {t.function.name: t.function.parameters for t in tools or []}

    def parse(self, text, pos):
        rest = text[pos:].lstrip()
        pos = len(text) - len(rest)
        wrapped = rest.startswith(self.bot)
        pos += len(self.bot) if wrapped else 0
        calls = []
        while match := self.invoke.match(text, pos):
            pos = self._invoke(text, match, calls)
        rest = text[pos:].lstrip()
        end = next((e for e in self.ends if rest.startswith(e)), '')
        if not calls or (wrapped and not end):
            raise Reject('no complete invoke' if not calls else 'calls block not closed')
        if rest[len(end):].strip():
            raise Reject('text after the batch')
        return calls

    def _invoke(self, text, match, calls):
        name = match.group('name').strip()
        if name not in self.tools:
            raise Reject(f'tool {name!r} was not offered')
        schema, pos, args = self.tools[name], match.end(), {}
        if not match.group('sc'):
            pos += len(text[pos:]) - len(text[pos:].lstrip())
            if text.startswith('{', pos):
                args, pos = _loads(text, pos)
                if not isinstance(args, dict):
                    raise Reject('JSON body is not an object')
            else:
                args, pos = self._params(text, pos, schema)
            close = f'</{match.group("tag")}>'
            pos += len(text[pos:]) - len(text[pos:].lstrip())
            if not text.startswith(close, pos):
                raise Reject(f'invoke {name!r} not closed by {close}')
            pos += len(close)
        check(args, {} if schema is None else schema)
        calls.append((name, args))
        return pos

    def _params(self, text, pos, schema):
        args = {}
        while match := self.param.match(text, pos):
            key, close = match.group('name'), f'</{match.group("tag")}>'
            end = text.find(close, match.end())
            raw = text[match.end():end]
            if end < 0 or self.markup.search(raw) or key in args:
                raise Reject(f'parameter {key!r} unclosed, duplicated or holding tag markup')
            args[key] = self._value(key, raw.strip(), match.group('string'), schema)
            pos = end + len(close)
        return args, pos

    @staticmethod
    def _value(key, raw, string, schema):
        if string is None:
            kind = (((schema or {}).get('properties') or {}).get(key) or {}).get('type')
            if kind == 'string':
                string = 'true'
            elif kind in ('integer', 'number', 'boolean', 'object', 'array', 'null'):
                string = 'false'
            else:
                raise Reject(f'no string attribute and no single schema type for {key!r}')
        if string == 'true':
            return raw
        value, end = _loads(raw) if string == 'false' else (None, -1)
        if end != len(raw):
            raise Reject(f'bad value or string="{string}" for {key!r}')
        return value


def native_batch(det, calls):
    """The calls as the native JSON DSML block the stock detector parses."""
    # '<' only occurs inside JSON strings; escaping it keeps closing tags out of the body.
    invokes = ''.join(f'{det.invoke_start_token} name="{name}">'
                      f'{json.dumps(args, ensure_ascii=False).replace("<", chr(92) + "u003c")}'
                      f'{det.invoke_end_token}\n' for name, args in calls)
    return f'{det.bot_token}\n{invokes}{det.eot_token}'


def make_strict_detect_and_parse(original, result_cls):
    def detect_and_parse(self, text, tools):
        starts = [i for i in (text.find(self.bot_token), text.find(self.invoke_start_token)) if i >= 0]
        if not starts:
            return original(self, text, tools)
        try:
            calls = _Batch(self, tools).parse(text, min(starts))
        except Exception as exc:  # Reject, or a parser bug: never return part of a batch
            logger.warning('DSV41 DSML strict: batch kept as text (%s)', exc)
            return result_cls(normal_text=text, calls=[])
        return original(self, text[:min(starts)] + native_batch(self, calls), tools)

    detect_and_parse.__wrapped__ = original
    return detect_and_parse


def install(module, result_cls=None):
    cls = getattr(module, 'DeepSeekV41Detector', None)
    if not enabled() or cls is None:
        logger.warning('DSV41 DSML strict not installed (%s)',
                       'DSV41_DSML_STRICT=0' if cls else 'no DeepSeekV41Detector')
        return
    if getattr(cls.detect_and_parse, '__wrapped__', None) is not None:
        return
    if result_cls is None:
        from sglang.srt.function_call.core_types import StreamingParseResult as result_cls
    cls.detect_and_parse = make_strict_detect_and_parse(cls.detect_and_parse, result_cls)
    logger.warning('DSV41 DSML strict: nonstream tool batches validated whole; '
                   'DSV41_DSML_STRICT=0 disables')
