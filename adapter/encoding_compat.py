"""Publisher-aligned V4.1 prompt encoding: reasoning-effort table, thinking kwargs,
and a completion-token cap.

SGLang's OpenAI chat path only reads ``chat_template_kwargs.thinking``. vLLM and the
EXL3 recipe use ``enable_thinking``. Without an alias, ``enable_thinking: true`` is
ignored (SGLang's default thinking flag is off) and a client that only sends the
vLLM name cannot turn thinking on.

The V4.1 encoder's named effort tiers also disagree with DeepSeek's
``encoding/encoding.py`` (SGLang: low/high = 25/50; publisher: 50/75). The
sitecustomize hook remaps them; ``xhigh`` stays as 75.

Chat completions that omit ``max_tokens`` fill the remaining context (1M on TP4).
That is how a racing-animation trial ran to 714k tokens. The 2x EXL3 recipe does
not install a server cap either — vLLM does the same fill — but every smoke and
README example there sends ``max_tokens``. This adapter fills *and* clamps at
``DSV41_MAX_NEW_TOKENS`` (default 32768; 0 restores the uncapped engine behaviour).
"""
import logging
import os

logger = logging.getLogger(__name__)

# Publisher encoding/encoding.py REASONING_EFFORT_MAPPINGS, plus SGLang's xhigh alias.
PUBLISHER_REASONING_EFFORT = {'low': 50, 'high': 75, 'xhigh': 75, 'max': 100}
DEFAULT_MAX_NEW_TOKENS = 32768


def normalize_chat_template_kwargs(kwargs):
    """Copy ``enable_thinking`` onto ``thinking`` (and the reverse) in a new dict.

    ``thinking`` wins when both are present and disagree. ``None`` / empty input
    is returned unchanged so SGLang's ``SGLANG_DEFAULT_THINKING`` still applies.
    """
    if not kwargs:
        return kwargs
    out = dict(kwargs)
    has_thinking = 'thinking' in out
    has_enable = 'enable_thinking' in out
    if has_thinking and has_enable:
        thinking = bool(out['thinking'])
        enable = bool(out['enable_thinking'])
        if thinking != enable:
            logger.warning(
                'chat_template_kwargs thinking=%r disagrees with enable_thinking=%r; '
                'using thinking', out['thinking'], out['enable_thinking'])
        out['thinking'] = thinking
        out['enable_thinking'] = thinking
    elif has_enable:
        out['thinking'] = bool(out['enable_thinking'])
        out['enable_thinking'] = out['thinking']
    elif has_thinking:
        out['thinking'] = bool(out['thinking'])
        out['enable_thinking'] = out['thinking']
    return out


def configured_max_new_tokens():
    """Return the completion-token default/cap, or 0 to leave the engine uncapped."""
    raw = os.environ.get('DSV41_MAX_NEW_TOKENS', str(DEFAULT_MAX_NEW_TOKENS)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning('DSV41_MAX_NEW_TOKENS=%r is not an int; using %s',
                       raw, DEFAULT_MAX_NEW_TOKENS)
        return DEFAULT_MAX_NEW_TOKENS


def apply_request_max_tokens(request):
    """Fill omitted max_tokens / max_completion_tokens and clamp to the cap.

    Mutates ``request`` in place and returns it. ``limit <= 0`` is a no-op.
    ``max_completion_tokens`` wins when both fields are set, matching SGLang.
    """
    limit = configured_max_new_tokens()
    if limit <= 0 or request is None:
        return request
    completion = getattr(request, 'max_completion_tokens', None)
    tokens = getattr(request, 'max_tokens', None)
    current = completion if completion is not None else tokens
    if current is None:
        request.max_tokens = limit
        return request
    if current > limit:
        if completion is not None:
            request.max_completion_tokens = limit
        else:
            request.max_tokens = limit
    return request


def install_encoder(module):
    """Point encoding_dsv41.REASONING_EFFORT_MAPPINGS at the publisher table."""
    module.REASONING_EFFORT_MAPPINGS.clear()
    module.REASONING_EFFORT_MAPPINGS.update(PUBLISHER_REASONING_EFFORT)


def install_serving_chat(module):
    """Rewrite thinking kwargs and apply the completion-token cap on every chat request."""
    cls = module.OpenAIServingChat
    original = cls._process_messages

    def _process_messages(self, request, is_multimodal):
        request.chat_template_kwargs = normalize_chat_template_kwargs(
            request.chat_template_kwargs)
        return original(self, request, is_multimodal)

    cls._process_messages = _process_messages

    original_convert = getattr(cls, '_convert_to_internal_request', None)
    if original_convert is not None:
        def _convert(self, request, *args, **kwargs):
            apply_request_max_tokens(request)
            return original_convert(self, request, *args, **kwargs)

        cls._convert_to_internal_request = _convert

    limit = configured_max_new_tokens()
    logger.warning(
        'DSV41 chat_template_kwargs: enable_thinking is accepted as an alias for thinking'
        + (f'; omitted max_tokens defaults and caps at {limit}' if limit > 0
           else '; DSV41_MAX_NEW_TOKENS=0 leaves completions uncapped'))


def install_anthropic_protocol(module):
    """Accept Anthropic ``thinking: {"type": "enabled"}`` without ``budget_tokens``.

    SGLang's Messages protocol requires ``budget_tokens`` on ``enabled`` (the SDK's
    discriminated shape), but its serving layer never enforces a budget: anything other
    than ``disabled`` turns reasoning on, and ``output_config.effort`` sets the effort.
    Clients that send ``enabled`` with ``output_config.effort`` and no budget (DeepSeek
    Harness's DeepSeek provider does) got a 400 on every request. Without a budget,
    ``enabled`` now passes like ``adaptive``; a budget, when given, is still checked, and
    ``disabled`` and ``adaptive`` keep their rules.
    """
    cls = getattr(module, 'AnthropicThinkingParam', None)
    validators = getattr(getattr(cls, '__pydantic_decorators__', None), 'model_validators', {})
    entry = validators.get('_validate_thinking_shape')
    if entry is None:
        logger.warning('DSV41 Anthropic thinking: AnthropicThinkingParam shape check not found; unchanged')
        return
    original = entry.func

    def _validate_thinking_shape(self):
        if self.type == 'enabled' and self.budget_tokens is None:
            return self
        return original(self)

    entry.func = _validate_thinking_shape
    cls.model_rebuild(force=True)
    for name in ('AnthropicMessagesRequest', 'AnthropicCountTokensRequest'):
        model = getattr(module, name, None)
        if model is not None:
            model.model_rebuild(force=True)
    logger.warning('DSV41 Anthropic thinking: "enabled" is accepted without budget_tokens '
                   '(effort comes from output_config.effort; budgets are not enforced)')
