#!/usr/bin/env python3
"""Sanitize textual image-placeholder tokens instead of rejecting the request.

`encoding_dsv41` raises `ValueError` in two places when the literal image placeholder
(`[image]`) appears in a message's *text*:

1. ``_validate_no_image_sp_tokens()`` -- string-form ``content`` / ``reasoning_content``.
   On the Anthropic endpoint this is the one that fires; the observed failure was HTTP 500
   on `/v1/messages`, and because the token re-enters the transcript the loop is
   self-sustaining: every later turn in that conversation fails the same way. Measured on a
   4x DGX Spark fleet: 111 of 145 Anthropic-protocol requests failed while the literal was
   present.

2. ``_process_image_blocks()`` -- a ``{"type": "text", ...}`` block, including text nested
   inside a ``tool_result`` block (the function recurses into those). This is the path a tool
   that reads a log file containing the token comes back on, and it is reached by block-form
   content, which `/v1/chat/completions` accepts natively. Rejected with HTTP 400.

Both branches become sanitize-and-warn, so such requests keep serving. The rewrite is in
place on a ``copy.deepcopy``'d message (``process_image_messages`` copies before validating),
so it cannot leak into the caller's dict or a cache key. Real image content blocks are
untouched.

Idempotent, per anchor. Exits non-zero if an anchor no longer matches, so a base-image change
cannot silently drop the fix. Optionally takes the target path as argv[1].
"""
import pathlib
import sys

DEFAULT_TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
)
TAG = "# [encoding_dsv41-placeholder-sanitize]"
# Per-anchor sentinels. Deliberately not a single file-wide marker: the logger bootstrap
# below also carries TAG, so a marker test would be satisfied before anchor 1 was applied.
SENT1 = "sanitizing image placeholder token in message content"
SENT2 = "sanitizing image placeholder token in a text block"
RAISE1 = "Message content contains image special token"
RAISE2 = "Text block contains image placeholder"
LOGGER_BOOTSTRAP = "logger = logging.getLogger(__name__)  " + TAG + "\n"

# ------------------------------------------------------------------ anchor 1
OLD1 = """def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    \"\"\"Reject user-supplied image placeholder tokens in textual fields.\"\"\"
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
"""

NEW1 = """def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    \"\"\"Sanitize textual image placeholder tokens instead of rejecting the request.

    Raising here turns any conversation whose text carries the literal placeholder into a
    hard failure, and because the token then re-enters the transcript, a self-sustaining
    error loop. # [encoding_dsv41-placeholder-sanitize]
    \"\"\"
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        logger.warning(
            "encoding_dsv41: sanitizing image placeholder token in message content "
            "(upstream would raise) # [encoding_dsv41-placeholder-sanitize]"
        )
        msg["content"] = content.replace(IMAGE_PLACEHOLDER, "[image]")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        logger.warning(
            "encoding_dsv41: sanitizing image placeholder token in reasoning_content "
            "(upstream would raise) # [encoding_dsv41-placeholder-sanitize]"
        )
        msg["reasoning_content"] = reasoning_content.replace(IMAGE_PLACEHOLDER, "[image]")
"""


def ensure_logger(src: str) -> str:
    """Insert `import logging` and a module-level logger if the file has none."""
    if LOGGER_BOOTSTRAP in src:
        return src
    if "import logging" not in src:
        anchor = "from typing import Any, Dict, List, Optional, Tuple, Union\n"
        if src.count(anchor) != 1:
            raise SystemExit("logger: cannot find the typing import anchor")
        src = src.replace(anchor, "import logging\n" + anchor)
    anchor = "import logging\n"
    idx = src.index(anchor) + len(anchor)
    lines = src[idx:].splitlines(keepends=True)
    n = 0
    for ln in lines:
        if ln.startswith(("import ", "from ")) or not ln.strip():
            n += 1
        else:
            break
    at = idx + sum(len(x) for x in lines[:n])
    return src[:at] + "\n" + LOGGER_BOOTSTRAP + src[at:]


def patch_text_block(src: str) -> tuple:
    """Anchor 2: replace the raise in the `type == "text"` branch with a sanitize.

    The raise is the last statement of that branch, so the branch still falls through to
    `new_blocks.append(block)` with the sanitized copy.
    """
    lines = src.splitlines(keepends=True)
    hit = next((i for i, ln in enumerate(lines) if RAISE2 in ln), None)
    if hit is None:
        return src, False
    start = hit
    while start >= 0 and lines[start].strip() != "raise ValueError(":
        start -= 1
    end = hit
    while end < len(lines) and lines[end].strip() != ")":
        end += 1
    if start < 0 or end >= len(lines):
        raise SystemExit("anchor 2: could not bracket the raise block")
    ind = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    new = [
        ind + "logger.warning(\n",
        ind + "    \"encoding_dsv41: sanitizing image placeholder token in a text block \"\n",
        ind + "    \"(upstream would raise) # [encoding_dsv41-placeholder-sanitize]\"\n",
        ind + ")\n",
        ind + "block = dict(block)\n",
        ind + "block[\"text\"] = text.replace(IMAGE_PLACEHOLDER, \"[image]\")\n",
    ]
    return "".join(lines[:start] + new + lines[end + 1:]), True


def verify(src: str) -> list:
    problems = []
    if RAISE1 in src:
        problems.append("anchor 1 still raises")
    if RAISE2 in src:
        problems.append("anchor 2 still raises")
    if SENT1 not in src:
        problems.append("anchor 1 sanitize missing")
    if SENT2 not in src:
        problems.append("anchor 2 sanitize missing")
    if LOGGER_BOOTSTRAP not in src:
        problems.append("logger bootstrap missing")
    if "import logging\n" not in src:
        problems.append("import logging missing")
    return problems


def main() -> int:
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET)
    if not target.is_file():
        print("ERROR: target not found: %s (base image changed?)" % target, file=sys.stderr)
        return 1
    src = target.read_text()
    changed = []
    src = ensure_logger(src)
    did1, did2 = SENT1 in src, SENT2 in src
    if did1 and did2:
        print("already patched: %s" % target)
        return 0
    if not did1:
        if src.count(OLD1) != 1:
            print("ERROR: expected exactly 1 match for anchor 1, found %d -- re-derive the patch"
                  % src.count(OLD1), file=sys.stderr)
            return 1
        src = src.replace(OLD1, NEW1)
        changed.append("anchor1(string content)")
    if not did2:
        src, applied2 = patch_text_block(src)
        if not applied2:
            print("ERROR: anchor 2 not found -- base image changed, re-derive the patch",
                  file=sys.stderr)
            return 1
        changed.append("anchor2(text block / tool_result)")
    problems = verify(src)
    if problems:
        print("ERROR: post-patch verification failed: %s" % "; ".join(problems),
              file=sys.stderr)
        return 1
    target.write_text(src)
    print("patched %s: %s" % (target, ", ".join(changed) or "logger only"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
