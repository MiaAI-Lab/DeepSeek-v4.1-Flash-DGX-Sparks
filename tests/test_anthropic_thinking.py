#!/usr/bin/env python3
"""Anthropic thinking: "enabled" without budget_tokens, against SGLang's own protocol module.

Runs where SGLang is installed (inside the serving image):
    python3 tests/test_anthropic_thinking.py
Without SGLang it skips with exit code 0.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import install_anthropic_protocol  # noqa: E402


def accepts(cls, **fields):
    try:
        cls(**fields)
        return True
    except Exception:
        return False


def main():
    try:
        protocol = importlib.import_module('sglang.srt.entrypoints.anthropic.protocol')
    except Exception as exc:
        print(f'skip: SGLang Anthropic protocol not importable ({exc.__class__.__name__})')
        return 0
    param = protocol.AnthropicThinkingParam
    request = protocol.AnthropicMessagesRequest
    body = dict(model='m', max_tokens=16, messages=[{'role': 'user', 'content': 'hi'}])

    # Stock shape check: refused before the hook (skipped if the image already loaded it).
    stock = not accepts(param, type='enabled')
    install_anthropic_protocol(protocol)

    checks = {
        'enabled without budget accepted': accepts(param, type='enabled'),
        'request with enabled, no budget accepted': accepts(request, **body, thinking={'type': 'enabled'}),
        'enabled with budget >= 1024 accepted': accepts(param, type='enabled', budget_tokens=2048),
        'enabled with budget < 1024 still refused': not accepts(param, type='enabled', budget_tokens=512),
        'disabled with budget still refused': not accepts(param, type='disabled', budget_tokens=2048),
        'adaptive still accepted': accepts(param, type='adaptive'),
        'adaptive with budget still refused': not accepts(param, type='adaptive', budget_tokens=2048),
    }
    if stock:
        checks['stock refused enabled without budget (before the hook)'] = True
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(('ok   ' if ok else 'FAIL ') + name)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
