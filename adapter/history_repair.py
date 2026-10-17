"""Keep a broken assistant tool call in the history instead of rejecting the request.

An agent client (openai-agents / litellm) that receives a truncated streamed tool call turns
it into a tool error ("Invalid JSON input for tool ... Please try again") and sends the next
turn with the broken call still in its history. Stock ``parse_tool_call_arguments`` raises
``ValueError`` for that history, which surfaces as HTTP 400 -- not retryable, so the agent
dies on a mistake it was built to recover from.

This hook replaces ``serving_chat.parse_tool_call_arguments`` (looked up by name from
``normalize_assistant_tool_call_arguments``). A valid JSON object passes unchanged. Invalid
JSON or a non-object is handed to the template as
``{"_unparseable_arguments": <raw text, complete>}`` and logged, so the model sees its own
broken attempt next to the client's error message and can call the tool again.
Set ``DSV41_HISTORY_REPAIR=0`` to keep the stock behaviour.
"""
import logging
import os

logger = logging.getLogger(__name__)

REPAIR_KEY = '_unparseable_arguments'


def enabled():
    raw = os.environ.get('DSV41_HISTORY_REPAIR', '1')
    return str(raw).strip() not in ('0', 'false', 'off', 'no')


def make_repairing_parser(original):
    def parse_tool_call_arguments(arguments):
        try:
            return original(arguments)
        except ValueError as exc:
            reason = 'not_object' if 'JSON object' in str(exc) else 'json'
            logger.warning(
                'DSV41 history repair: len=%s reason=%s',
                len(arguments), reason)
            return {REPAIR_KEY: arguments}

    parse_tool_call_arguments.__wrapped__ = original
    return parse_tool_call_arguments


def install(module):
    if not enabled():
        logger.warning('DSV41 history repair disabled (DSV41_HISTORY_REPAIR=0)')
        return
    original = getattr(module, 'parse_tool_call_arguments', None)
    if original is None:
        logger.warning('DSV41 history repair: parse_tool_call_arguments not found; not installed')
        return
    if getattr(original, '__wrapped__', None) is not None:
        return
    module.parse_tool_call_arguments = make_repairing_parser(original)
    logger.warning(
        'DSV41 history repair: broken assistant tool-call arguments become {%r: raw}; '
        'DSV41_HISTORY_REPAIR=0 disables', REPAIR_KEY)
