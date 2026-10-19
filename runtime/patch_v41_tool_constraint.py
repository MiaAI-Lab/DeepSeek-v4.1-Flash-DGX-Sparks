#!/usr/bin/env python3
"""Backport the DeepSeek V4.1 tool-call grammar from sgl-project/sglang#39665.

The pinned ``lmsysorg/sglang:dev-dsv41`` base ships an older
``deepseekv41_detector.py`` whose ``get_structural_tag_name()`` returns ``None``
and which has no ``get_structural_tag()``. For ``tool_choice="required"`` and for
a named tool choice, ``FunctionCallParser.get_structure_constraint`` then falls
back to the legacy per-invoke structural tag. That grammar forces
``<｜DSML｜ invoke name="...">{json}</｜DSML｜ invoke>`` but never the
``\\n\\n<｜DSML｜ calls>\\n`` opener. The V3.2-lineage parser anchors on the calls
opener, so the response comes back as HTTP 200 with the DSML markup in
``message.content``, no ``tool_calls`` and ``finish_reason="stop"``. ``auto``
tool calling is unaffected because the model writes the opener itself.

sglang#39665 (commit 464fffbec8d0) adds ``DeepSeekV41Detector.get_structural_tag``,
which builds the V4.1 shape with the spaced tag names and forces the calls block
for required/named choice. It only uses structural-tag classes already present in
the base image's xgrammar 0.2.1. The later sglang#39026 switches to xgrammar's
native ``deepseek_v4_1`` style and needs xgrammar >= 0.2.7; it is not backported.

The vendored file runtime/sglang/deepseekv41_detector.py is byte-identical to
upstream at that commit (Apache-2.0, see LICENSE.sglang). The rest of the base
image already matches 464fffbec8d0 for deepseekv32_detector.py,
deepseekv4_detector.py and base_format_detector.py.

Idempotent. Exits non-zero unless the target is either the known stock file or
the backport, so a base-image change cannot silently drop or clobber the fix.
Optionally takes the target path as argv[1].
"""
import hashlib
import pathlib
import sys

DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/function_call/deepseekv41_detector.py"
)
STOCK_SHA256 = "862168f54a45a336f99d6389840a07e1be55bb589a99a030ecf109c2b09027e4"
BACKPORT_SHA256 = "1db7eb88dfed0530a4c85231e525da83a5fd009e4feb85610bffd94d4c8c3672"
BACKPORT = pathlib.Path(__file__).resolve().parent / "sglang" / "deepseekv41_detector.py"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    replacement = BACKPORT.read_bytes()
    if sha256(replacement) != BACKPORT_SHA256:
        print(f"[v41-tool-constraint] vendored backport hash mismatch: {BACKPORT}", file=sys.stderr)
        return 1
    current = sha256(target.read_bytes())
    if current == BACKPORT_SHA256:
        print(f"[v41-tool-constraint] already applied: {target}")
        return 0
    if current != STOCK_SHA256:
        print(
            f"[v41-tool-constraint] {target} is neither the stock dev-dsv41 file nor the "
            f"backport (sha256 {current}); re-check against sglang#39665/#39026",
            file=sys.stderr,
        )
        return 1
    target.write_bytes(replacement)
    print(f"[v41-tool-constraint] applied sglang#39665 V4.1 structural tag: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
