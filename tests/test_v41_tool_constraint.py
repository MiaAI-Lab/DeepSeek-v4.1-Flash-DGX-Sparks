#!/usr/bin/env python3
"""required/named tool_choice on the V4.1 DSML detector (runtime/patch_v41_tool_constraint.py).

Runs inside the image against the real SGLang FunctionCallParser and xgrammar; no model or
GPU. For each forced choice it builds the constraint SGLang would send to xgrammar, compiles
it, and checks that the grammar (a) admits the block the parser can read and (b) rejects the
bare invoke that the stock dev-dsv41 constraint produced, which reached clients as
message.content with no tool_calls. Then it parses the admitted text back into tool calls.
"""
import json

from xgrammar import Grammar, StructuralTag, StructuralTagItem
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice, ToolChoiceFuncName
from sglang.srt.function_call.function_call_parser import FunctionCallParser

SCHEMA = {'type': 'object', 'properties': {'symbol': {'type': 'string'}}, 'required': ['symbol']}
TOOLS = [
    Tool(type='function', function=Function(name='lookup_symbol', description='Look up a symbol', parameters=SCHEMA)),
    Tool(type='function', function=Function(name='list_files', description='List files',
                                            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}})),
]
NAMED = ToolChoice(function=ToolChoiceFuncName(name='lookup_symbol'))
failures = []


def check(label, ok):
    print(('ok   ' if ok else 'FAIL ') + label)
    if not ok:
        failures.append(label)


def grammar_for(parser, choice, thinking=False, parallel=True):
    constraint = parser.get_structure_constraint(choice, parallel_tool_calls=parallel, thinking_mode=thinking)
    check(f'{choice!r} thinking={thinking} parallel={parallel}: structural_tag constraint',
          constraint is not None and constraint[0] == 'structural_tag')
    tag = constraint[1]
    check(f'{choice!r}: V4.1 native tag, not the legacy per-invoke fallback', isinstance(tag, StructuralTag))
    if isinstance(tag, StructuralTag):
        return Grammar.from_structural_tag(tag)
    # Stock base: compile the legacy tag exactly as XGrammarGrammarBackend.dispatch_structural_tag does,
    # so the remaining checks show what the stock grammar actually admits.
    legacy = json.loads(tag.model_dump_json(by_alias=True))
    items = [StructuralTagItem(begin=s['begin'], schema=json.dumps(s.get('schema') or {}), end=s['end'])
             for s in legacy['structures']]
    new_tag = StructuralTag.from_legacy_structural_tag(items, legacy['triggers'])
    new_tag.format.at_least_one = legacy.get('at_least_one', False)
    return Grammar.from_structural_tag(new_tag)


def main():
    parser = FunctionCallParser(TOOLS, 'deepseekv41')
    d = parser.detector
    invoke = lambda name, args: f'{d.invoke_start_token} name="{name}">{json.dumps(args)}{d.invoke_end_token}\n'
    prefix_nl = getattr(d, 'tool_calls_prefix', '\n\n')  # stock base lacks the attribute
    block = lambda *calls: f"{prefix_nl}{d.bot_token}\n{''.join(calls)}{d.eot_token}"
    one = invoke('lookup_symbol', {'symbol': 'WidgetFactory'})
    # What the stock constraint let the model emit (observed on a live TP4 deployment):
    bare = f'{d.invoke_start_token} name="lookup_symbol">{{"symbol": "WidgetFactory"}}{d.invoke_end_token}\n{d.eot_token}'

    for choice in (NAMED, 'required'):
        for thinking in (False, True):
            g = grammar_for(parser, choice, thinking=thinking)
            prefix = 'Need the symbol.</think>' if thinking else ''
            check(f'{choice!r} thinking={thinking}: accepts calls block', _is_grammar_accept_string(g, prefix + block(one)))
            check(f'{choice!r} thinking={thinking}: rejects bare invoke', not _is_grammar_accept_string(g, prefix + bare))
            check(f'{choice!r} thinking={thinking}: rejects prose', not _is_grammar_accept_string(g, prefix + 'I will look it up.'))

    g = grammar_for(parser, NAMED)
    check('named: rejects a different tool', not _is_grammar_accept_string(g, block(invoke('list_files', {'path': '.'}))))
    g = grammar_for(parser, 'required', parallel=True)
    two = block(one, invoke('list_files', {'path': 'src'}))
    check('required parallel: accepts two invokes', _is_grammar_accept_string(g, two))
    g = grammar_for(parser, 'required', parallel=False)
    check('required parallel=False: rejects two invokes', not _is_grammar_accept_string(g, two))

    text, calls = parser.parse_non_stream(block(one))
    check('parse: one call, no leftover markup',
          len(calls) == 1 and calls[0].name == 'lookup_symbol'
          and json.loads(calls[0].parameters) == {'symbol': 'WidgetFactory'} and d.invoke_start_token not in text)
    text, calls = parser.parse_non_stream(two)
    check('parse: two parallel calls', [c.name for c in calls] == ['lookup_symbol', 'list_files'])
    _, calls = parser.parse_non_stream(bare)
    print(f'info stock bare-invoke output parses to {len(calls)} call(s)')

    if failures:
        raise SystemExit(f'{len(failures)} failure(s)')
    print('PASS test_v41_tool_constraint')


if __name__ == '__main__':
    main()
