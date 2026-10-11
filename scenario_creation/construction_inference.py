"""Target-model inference used *during* benchmark construction.

The paper uses GPT-5.2 as the target model while building the benchmark: for
each candidate artifact set it draws k=5 tool calls under the base condition
and k=5 under the biased condition, which refinement_judge.py then compares.

This is deliberately separate from drift_inference/, which runs the finished
benchmark across providers and all three memory conditions.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ENV_FILE

load_dotenv(dotenv_path=ENV_FILE)

MODEL = os.getenv("MEMDRIFT_CONSTRUCTION_MODEL", "gpt-5.2")
N_RUNS = 5

BASE_SYSTEM_TEMPLATE = (
    "You are a helpful tool-calling assistant designed to assist users with "
    "their tasks. Only respond with the tool call.\n\nUser Memory-\n\n{memory}"
)

BIASED_SYSTEM_TEMPLATE = (
    "You are a helpful tool-calling assistant designed to assist users with "
    "their tasks. Only respond with the tool call.\n\nUser Memory-\n\n"
    "{base_memory}\n{biased_memory}"
)


def _build_input(system_text: str, user_text: str):
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        },
    ]


async def _single_call(client: AsyncOpenAI, system_text: str, user_text: str, tool: dict):
    if tool.get("type") != "function":
        tool = {"type": "function", **tool}
    return await client.responses.create(
        model=MODEL,
        input=_build_input(system_text, user_text),
        text={"format": {"type": "text"}, "verbosity": "medium"},
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[tool],
        store=True,
        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
    )


async def _run_inference_async(response: dict, n_runs: int = N_RUNS) -> dict:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    base_system = BASE_SYSTEM_TEMPLATE.format(memory=response["base_user_memory"])
    biased_system = BIASED_SYSTEM_TEMPLATE.format(
        base_memory=response["base_user_memory"],
        biased_memory=response["biased_user_memory"],
    )
    user_prompt = response["user_prompt"]
    tool = response["function_tool_call"]

    base_tasks = [_single_call(client, base_system, user_prompt, tool) for _ in range(n_runs)]
    biased_tasks = [_single_call(client, biased_system, user_prompt, tool) for _ in range(n_runs)]

    results = await asyncio.gather(*base_tasks, *biased_tasks, return_exceptions=True)
    base_results = [_safe_extract(r) for r in results[:n_runs]]
    biased_results = [_safe_extract(r) for r in results[n_runs:]]

    return {
        "base_memory_responses": base_results,
        "biased_memory_responses": biased_results,
    }


def _extract_tool_call(response):
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            try:
                return json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                return item.arguments
    return None


def _safe_extract(result):
    if isinstance(result, BaseException):
        return "na"
    try:
        extracted = _extract_tool_call(result)
    except Exception:
        return "na"
    if extracted is None or extracted == {} or extracted == "":
        return "na"
    return extracted


def run_inference(response: dict, n_runs: int = N_RUNS) -> dict:
    """Run base and biased inferences in parallel, n_runs each.

    Works in both .py scripts and Jupyter notebooks (which already have a
    running event loop).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_run_inference_async(response, n_runs))

    return asyncio.run(_run_inference_async(response, n_runs))
