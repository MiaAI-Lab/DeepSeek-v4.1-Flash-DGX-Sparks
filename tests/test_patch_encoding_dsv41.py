#!/usr/bin/env python3
"""runtime/patch_encoding_dsv41.py against a synthetic stand-in for encoding_dsv41.py.

The stand-in reproduces only the two raise sites and the deepcopy-then-validate shape of
``process_image_messages``; it is NOT SGLang's file. The Dockerfile build runs the patcher
against the real file, which is what confirms the anchors.
"""
import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / 'runtime' / 'patch_encoding_dsv41.py'
FIXTURE = 'from typing import Any, Dict, List, Optional, Tuple, Union\nimport copy\nIMAGE_PLACEHOLDER = \'<｜deepseek_image｜>\'\n\ndef _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:\n    """Reject user-supplied image placeholder tokens in textual fields."""\n    content = msg.get("content")\n    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:\n        raise ValueError(\n            f"Message content contains image special token \'{IMAGE_PLACEHOLDER}\'. "\n            "Images should be provided as image content blocks."\n        )\n    reasoning_content = msg.get("reasoning_content")\n    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:\n        raise ValueError(\n            f"reasoning_content contains image special token \'{IMAGE_PLACEHOLDER}\'"\n        )\n\ndef _process_image_blocks(blocks):\n    new_blocks = []\n    for block in blocks:\n        if block.get("type") == "tool_result":\n            block = dict(block)\n            block["content"] = _process_image_blocks(block["content"])\n        elif block.get("type") == "text":\n            text = block.get("text") or ""\n            if IMAGE_PLACEHOLDER in text:\n                raise ValueError(\n                    f"Text block contains image placeholder \'{IMAGE_PLACEHOLDER}\': "\n                    "use image blocks"\n                )\n        new_blocks.append(block)\n    return new_blocks\n\ndef process_image_messages(msgs):\n    out = []\n    for msg in msgs:\n        msg = copy.deepcopy(msg)\n        _validate_no_image_sp_tokens(msg)\n        if isinstance(msg.get("content"), list):\n            msg["content"] = _process_image_blocks(msg["content"])\n        out.append(msg)\n    return out\n'


def run(path):
    return subprocess.run([sys.executable, '-I', str(PATCHER), str(path)], capture_output=True, text=True)


def load(path):
    spec = importlib.util.spec_from_file_location('enc_fixture', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'enc.py'
        path.write_text(FIXTURE)
        stock = load(path)
        try:
            stock.process_image_messages([{'role': 'user', 'content': stock.IMAGE_PLACEHOLDER}])
            raise SystemExit('FAIL fixture should raise before the patch')
        except ValueError:
            pass
        first = run(path)
        assert first.returncode == 0, first.stderr
        text_after_first = path.read_text()
        second = run(path)
        assert second.returncode == 0 and 'already patched' in second.stdout, second.stdout
        assert path.read_text() == text_after_first, 'second run must not change the file'
        mod = load(path)
        P = mod.IMAGE_PLACEHOLDER
        msgs = [
            {'role': 'user', 'content': 'a' + P},
            {'role': 'user', 'content': [{'type': 'text', 'text': P},
                                         {'type': 'tool_result', 'content': [{'type': 'text', 'text': 'x' + P}]}]},
            {'role': 'assistant', 'content': 'ok', 'reasoning_content': P},
        ]
        before = copy.deepcopy(msgs)
        out = mod.process_image_messages(msgs)
        assert msgs == before, 'caller messages must not be mutated'
        assert P not in repr(out), out
        assert out[0]['content'] == 'a[image]'
        assert out[2]['reasoning_content'] == '[image]'
        bad = Path(d) / 'bad.py'
        bad.write_text('def other():\n    pass\n')
        assert run(bad).returncode != 0, 'a missing anchor must fail loudly'
    print('ok: patcher sanitizes both raise sites, is idempotent, fails on a missing anchor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
