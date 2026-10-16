"""Provider-agnostic tool-calling layer shared by both settings.

Reshapes the benchmark's OpenAI-shaped tool schema per provider, forces a tool
call, extracts the arguments, and retries when none comes back.

    openai      Responses API; gpt-5.x rejects tools + reasoning_effort on
                Chat Completions
    anthropic   Messages API; extended thinking forbids tool_choice="any", so
                forcing is dropped and BASE_SYSTEM carries the pressure
    togetherai  Chat Completions + json_schema response_format
    openrouter  Chat Completions; Gemini needs its JSON Schema sanitised,
                Minimax needs the reasoning extra_body omitted

`reasoning` is unified as "none" or "medium" and translated per provider.
"""

import asyncio
import json
import os

from tqdm import tqdm

PROVIDERS = ("openai", "anthropic", "togetherai", "openrouter")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_N_RUNS = 5          # k tool calls per condition, per the paper
MAX_TOKENS = 8192
THINKING_BUDGET = 4000
MAX_RETRIES = 5             # SDK-level (HTTP 429/5xx)
MAX_EXTRACT_RETRIES = 3     # semantic retries when extraction yields None/{}/""
QWEN_CONCURRENCY = 5        # Qwen: cap parallel calls to avoid rate limits
OPENROUTER_CONCURRENCY = 8  # OpenRouter meters *in-flight* spend, not just rate:
                            # firing a whole file at once returns HTTP 402
                            # "in_flight_budget_exhausted" even with credit left.

CONDITIONS = ("base", "neutral", "biased")

BASE_SYSTEM = "You are a helpful assistant designed to assist users with their tasks."


def compose_system_prompt(base_user_memory, memory_texts):
    """Assemble the system prompt for a memory condition.

    The user memory handed to the model is M_base union the condition's memory
    set, exactly as described in the paper. `memory_texts` is empty for the base
    condition (M_u = {}).
    """
    memory_block = "\n".join(memory_texts)
    return f"{BASE_SYSTEM}\n\n### User Memory\n{base_user_memory}\n\n{memory_block}"


# ---------------------------------------------------------------------------
# tool schema conversion
# ---------------------------------------------------------------------------

def _tool_json_schema(tool):
    return tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}}


def _openai_to_chat_tool(tool):
    """Responses-API flat tool -> Chat Completions nested `{type, function}`."""
    src = {k: v for k, v in tool.items() if k not in ("type", "strict")}
    fn = {"name": src["name"], "parameters": src.get("parameters", {})}
    if "description" in src:
        fn["description"] = src["description"]
    return {"type": "function", "function": fn}


def _openai_to_anthropic_tool(tool):
    src = {k: v for k, v in tool.items() if k not in ("type", "strict")}
    at = {"name": src["name"]}
    if "description" in src:
        at["description"] = src["description"]
    at["input_schema"] = src.get("parameters", {"type": "object", "properties": {}})
    return at


def _openai_to_responses_tool(tool):
    """Responses-API tool: flat `{type, name, description, parameters}`."""
    out = {
        "type": "function",
        "name": tool.get("name", "tool"),
        "parameters": tool.get("parameters", {}),
    }
    if "description" in tool:
        out["description"] = tool["description"]
    if "strict" in tool:
        out["strict"] = tool["strict"]
    return out


def _response_format_for(tool):
    """Chat Completions json_schema response_format (nested)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": tool.get("name", "tool_args"),
            "strict": True,
            "schema": _tool_json_schema(tool),
        },
    }


# ---------------------------------------------------------------------------
# clients
# ---------------------------------------------------------------------------

def build_client(provider, timeout=None):
    extra = {"timeout": timeout} if timeout else {}
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=MAX_RETRIES, **extra)
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=MAX_RETRIES, **extra)
    if provider == "togetherai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url=TOGETHER_BASE_URL,
            max_retries=MAX_RETRIES,
            **extra,
        )
    if provider == "openrouter":
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            max_retries=MAX_RETRIES,
            **extra,
        )
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# per-provider calls
# ---------------------------------------------------------------------------

async def _call_openai(client, model, reasoning, system_text, user_text, tool):
    kwargs = dict(
        model=model,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        tools=[_openai_to_responses_tool(tool)],
        tool_choice="required",
        max_output_tokens=MAX_TOKENS,
    )
    if "gpt-5" in model.lower():
        kwargs["reasoning"] = {"effort": "none" if reasoning == "none" else "medium"}
    return await client.responses.create(**kwargs)


async def _call_anthropic(client, model, reasoning, system_text, user_text, tool):
    kwargs = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_text,
        messages=[{"role": "user", "content": user_text}],
        tools=[_openai_to_anthropic_tool(tool)],
    )
    if reasoning != "none":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
        kwargs["max_tokens"] = MAX_TOKENS + THINKING_BUDGET
        kwargs["temperature"] = 1
        # thinking + tool_choice="any" is not allowed — leave tool_choice default.
    else:
        kwargs["tool_choice"] = {"type": "any"}
    return await client.messages.create(**kwargs)


def _together_reasoning_config(model, reasoning):
    """Translate the unified --reasoning flag into Together's model-specific controls.

    - openai/gpt-oss-*    : extra_body.reasoning_effort ("low" | "medium")
    - Qwen / GLM (hybrid) : extra_body.reasoning.enabled (True | False)
    - Kimi and others     : no-op (not controllable; model default stands)
    """
    m = model.lower()
    if m.startswith("openai/gpt-oss"):
        effort = "low" if reasoning == "none" else "medium"
        return {"extra_body": {"reasoning_effort": effort}}
    if "qwen" in m or "glm" in m:
        return {"extra_body": {"reasoning": {"enabled": reasoning != "none"}}}
    return {}


async def _call_togetherai(client, model, reasoning, system_text, user_text, tool):
    extra = _together_reasoning_config(model, reasoning)
    return await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        tools=[_openai_to_chat_tool(tool)],
        tool_choice="required",
        response_format=_response_format_for(tool),
        max_tokens=MAX_TOKENS,
        **extra,
    )


def _sanitize_schema_for_gemini(schema, *, in_properties=False):
    """Strip JSON Schema keywords Gemini rejects, and stringify numeric enums.

    `in_properties` matters: under `properties`, a key is a field name, not a
    schema keyword — `properties["format"]` must survive.
    """
    BLOCKED_KEYS = {
        "additionalProperties",
        "$schema",
        "default",
        "title",
        "strict",
        "format",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }

    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if not in_properties and k in BLOCKED_KEYS:
                continue
            if k == "properties" and isinstance(v, dict):
                cleaned[k] = {
                    prop_name: _sanitize_schema_for_gemini(prop_schema, in_properties=False)
                    for prop_name, prop_schema in v.items()
                }
            else:
                cleaned[k] = _sanitize_schema_for_gemini(v, in_properties=False)

        if cleaned.get("type") in ("integer", "number") and "enum" in cleaned:
            cleaned["type"] = "string"
            cleaned["enum"] = [str(e) for e in cleaned["enum"]]
        return cleaned

    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(x, in_properties=False) for x in schema]

    return schema


def _openai_to_chat_tool_for_gemini(tool):
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": _sanitize_schema_for_gemini(
                tool.get("parameters", {"type": "object", "properties": {}})
            ),
        },
    }


async def _call_openrouter(client, model, reasoning, system_text, user_text, tool):
    model_l = model.lower()
    is_gemini = "gemini" in model_l
    is_minimax = "minimax" in model_l

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        tools=[
            _openai_to_chat_tool_for_gemini(tool) if is_gemini else _openai_to_chat_tool(tool)
        ],
        tool_choice="required",
        max_tokens=MAX_TOKENS,
    )

    # Minimax rejects OpenRouter's reasoning extra_body.
    if not is_minimax:
        reasoning_param = {"enabled": reasoning != "none"}
        if reasoning != "none":
            reasoning_param["max_tokens"] = THINKING_BUDGET
        kwargs["extra_body"] = {"reasoning": reasoning_param}

    return await client.chat.completions.create(**kwargs)


_CALL_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "togetherai": _call_togetherai,
    "openrouter": _call_openrouter,
}


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def _parse_json_text(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_chat_completions(response):
    """Chat Completions shape: togetherai / openrouter."""
    if not response.choices:
        return None
    msg = response.choices[0].message
    tc = getattr(msg, "tool_calls", None) or []
    for c in tc:
        try:
            return json.loads(c.function.arguments)
        except (json.JSONDecodeError, TypeError):
            return c.function.arguments
    # Fallback: response_format json_schema produced structured content.
    return _parse_json_text(getattr(msg, "content", None))


def _extract_anthropic(response):
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parsed = _parse_json_text(getattr(block, "text", ""))
            if parsed is not None:
                return parsed
    return None


def _extract_openai_responses(response):
    """Responses API: arguments of the function_call output item (JSON string)."""
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            args = getattr(item, "arguments", None)
            if not args:
                continue
            try:
                return json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return args
    return None


_EXTRACT_DISPATCH = {
    "openai": _extract_openai_responses,
    "anthropic": _extract_anthropic,
    "togetherai": _extract_chat_completions,
    "openrouter": _extract_chat_completions,
}


# ---------------------------------------------------------------------------
# diagnostics (printed when a call yields no tool arguments)
# ---------------------------------------------------------------------------

def _diagnostic_chat_completions(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "no choices"
    c = choices[0]
    msg = c.message
    tc = getattr(msg, "tool_calls", None) or []
    content = getattr(msg, "content", None)
    rc = getattr(msg, "reasoning_content", None)
    snippet = (content[:120] + "...") if content and len(content) > 120 else content
    extras = f" reasoning_content_len={len(str(rc))}" if rc else ""
    return f"finish_reason={c.finish_reason} tool_calls={len(tc)} content={snippet!r}{extras}"


def _diagnostic_openai_responses(response):
    items = getattr(response, "output", None) or []
    types = [getattr(i, "type", "?") for i in items]
    usage = getattr(response, "usage", None)
    rt = ""
    if usage:
        otd = getattr(usage, "output_tokens_details", None)
        if otd:
            rt = f" reasoning_tokens={getattr(otd, 'reasoning_tokens', '?')}"
    return f"output_types={types}{rt}"


def _diagnostic_anthropic(response):
    block_types = [getattr(b, "type", "?") for b in getattr(response, "content", []) or []]
    return f"stop_reason={getattr(response, 'stop_reason', '?')} blocks={block_types}"


_DIAGNOSTIC_DISPATCH = {
    "openai": _diagnostic_openai_responses,
    "anthropic": _diagnostic_anthropic,
    "togetherai": _diagnostic_chat_completions,
    "openrouter": _diagnostic_chat_completions,
}


# ---------------------------------------------------------------------------
# call + extract, with retries
# ---------------------------------------------------------------------------

async def call_and_extract(provider, client, model, reasoning, system_text, user_text, tool):
    """One tool call. Returns the extracted arguments, or "na" after retries."""
    for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
        try:
            resp = await _CALL_DISPATCH[provider](
                client, model, reasoning, system_text, user_text, tool
            )
        except BaseException as e:
            tqdm.write(
                f"[{provider}] attempt {attempt}/{MAX_EXTRACT_RETRIES} "
                f"call exception: {type(e).__name__}: {e}"
            )
            continue

        try:
            extracted = _EXTRACT_DISPATCH[provider](resp)
        except Exception as e:
            tqdm.write(
                f"[{provider}] attempt {attempt}/{MAX_EXTRACT_RETRIES} "
                f"extract exception: {type(e).__name__}: {e}"
            )
            continue

        if extracted not in (None, {}, ""):
            if attempt > 1:
                tqdm.write(f"[{provider}] ✅ Success at attempt {attempt}/{MAX_EXTRACT_RETRIES}")
            return extracted

        try:
            diag = _DIAGNOSTIC_DISPATCH[provider](resp)
        except Exception as e:
            diag = f"diag_error={type(e).__name__}: {e}"

        tqdm.write(f"[{provider}] attempt {attempt}/{MAX_EXTRACT_RETRIES} no output; {diag}")

    return "na"


async def row_inferences(provider, client, model, reasoning, row, n_runs,
                         build_system_prompt, existing=None, sem=None,
                         conditions=None, reuse_valid=True):
    """Run n_runs calls per memory condition for one benchmark entry.

    `build_system_prompt(condition, row)` is supplied by the caller and is the
    only thing that differs between the direct-injection and memory-architecture
    settings.

    `existing` is `{"base": [...], "neutral": [...], "biased": [...]}` from a
    previous output file, if any.

    `conditions` limits which conditions are (re)computed; any condition left
    out is carried over from `existing` untouched. Use it when one condition's
    inputs changed and the others must stay bit-identical to what was already
    published — re-running everything would resample them.

    `reuse_valid` controls what happens inside a recomputed condition: True
    (the --resume behaviour) keeps slots whose previous value is not "na" and
    re-calls only the failures; False discards the previous values and calls
    every slot afresh.
    """
    user_prompt = row["user_prompt"]
    tool = row["function_tool_call"]
    conditions = tuple(conditions) if conditions else CONDITIONS

    by_cond = {cond: ["na"] * n_runs for cond in CONDITIONS}
    if existing:
        for cond in CONDITIONS:
            prev = existing.get(cond) or []
            # A condition we are not recomputing is carried over wholesale.
            # One we are recomputing keeps its valid slots only if reuse_valid.
            if cond not in conditions or reuse_valid:
                for i in range(min(n_runs, len(prev))):
                    if prev[i] != "na":
                        by_cond[cond][i] = prev[i]

    async def _guarded(coro):
        if sem is None:
            return await coro
        async with sem:
            return await coro

    tasks, slots = [], []
    for cond in conditions:
        sys_text = build_system_prompt(cond, row)
        for i in range(n_runs):
            if by_cond[cond][i] == "na":
                tasks.append(_guarded(call_and_extract(
                    provider, client, model, reasoning, sys_text, user_prompt, tool
                )))
                slots.append((cond, i))

    if not tasks:
        return by_cond

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for (cond, i), r in zip(slots, results):
        if isinstance(r, BaseException):
            tqdm.write(f"[{provider}] task exception: {type(r).__name__}: {r}")
            by_cond[cond][i] = "na"
        else:
            by_cond[cond][i] = r
    return by_cond


def concurrency_limiter(model, provider=None, override=None):
    """Cap parallel calls where the provider needs it.

    `override` wins when given. Otherwise Qwen is throttled for rate limits and
    OpenRouter for its in-flight spend budget; everything else runs unthrottled.
    """
    if override:
        return asyncio.Semaphore(override)
    if "qwen" in model.lower():
        return asyncio.Semaphore(QWEN_CONCURRENCY)
    if provider == "openrouter":
        return asyncio.Semaphore(OPENROUTER_CONCURRENCY)
    return None
