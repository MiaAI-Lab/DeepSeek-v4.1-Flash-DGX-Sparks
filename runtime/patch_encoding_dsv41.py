#!/usr/bin/env python3
"""Tolerate the DeepSeek image placeholder token in textual message fields.

`encoding_dsv41._validate_no_image_sp_tokens` raises ValueError when a message's *text*
contains the literal image placeholder token. Any conversation whose transcript carries
that token -- a model-emitted token replayed from history, or a log file fed back as
context -- therefore fails with HTTP 500 on the Anthropic endpoint (`/v1/messages`), and
because the token re-enters the transcript the failure is self-sustaining: every later
request in that conversation fails the same way.

Measured on a 4x DGX Spark fleet (TP4/EP2, Engram, 200G RoCE): 111 of 145
Anthropic-protocol requests failed while the literal was present, with the OpenAI endpoint
unaffected in production traffic.

This step rewrites the guard to sanitize the token into plain text and log it, instead of
rejecting the request. Real image content blocks are untouched.

Idempotent. Exits non-zero if the anchor is not found exactly once, so a base-image change
cannot silently drop the fix.
"""
import pathlib
import sys

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
)
# Present in any revision of this fix, so re-running on an already-patched image is a no-op.
MARKER = "sanitizing image placeholder token"

OLD = '''def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    """Reject user-supplied image placeholder tokens in textual fields."""
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError(
            f"Message content contains image special token '{IMAGE_PLACEHOLDER}'. "
            "Images should be provided as image content blocks."
        )
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            f"reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'"
        )
'''

NEW = '''def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    """Sanitize textual image placeholder tokens instead of rejecting the request.

    Raising here turns any conversation whose text carries the literal placeholder into a
    hard HTTP 500, and because the token then re-enters the transcript the failure is
    self-sustaining. Measured on a 4x DGX Spark fleet: 111/145 Anthropic-protocol
    requests failed before this patch. Rewrite the placeholder into plain text so the
    request can proceed.
    """
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        print("[encoding_dsv41] sanitizing image placeholder token in message content",
              flush=True)
        msg["content"] = content.replace(IMAGE_PLACEHOLDER, "[image]")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        print("[encoding_dsv41] sanitizing image placeholder token in reasoning_content",
              flush=True)
        msg["reasoning_content"] = reasoning_content.replace(IMAGE_PLACEHOLDER, "[image]")
'''


def main() -> int:
    if not TARGET.is_file():
        print("ERROR: target not found: %s (base image changed?)" % TARGET, file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if MARKER in src:
        print("already patched: %s" % TARGET)
        return 0
    n = src.count(OLD)
    if n != 1:
        print(
            "ERROR: expected exactly 1 anchor match in %s, found %d. "
            "Upstream encoding_dsv41.py changed -- re-derive the patch." % (TARGET, n),
            file=sys.stderr,
        )
        return 1
    TARGET.write_text(src.replace(OLD, NEW))
    print("patched: %s" % TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
