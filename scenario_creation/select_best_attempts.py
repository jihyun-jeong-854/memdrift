"""Stage 5 of Benchmark Construction: selection, plus the neutral memory set.

  1. Load every attempt log for a bias dimension
  2. Keep the highest deflection-score attempt per (bias, domain, scenario)
  3. Project to the columns the benchmark carries
  4. Generate the neutral memory set M_n to match M_b in count, structure and
     length but with no directional bias, and normalise M_b to a list
  5. Write data/scenarios/<bias_dimension>.json

Human verification of the written entries happens after this step; the paper
confirms tool realism, prompt neutrality, memory plausibility, and that each
entry is genuine drift rather than an ambiguous task specification.

Usage:
    python -m scenario_creation.select_best_attempts \\
        results/construction_logs/speed_impatience
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.claude_query import batch_ask_claude
from common.paths import SCENARIO_DIR
from scenario_creation.attempt_log_parser import load_results_dataframe


MODEL = 'claude-sonnet-4-5-20250929'
BATCH_SIZE = 5
MAX_ROUNDS = 6

DEFAULT_OUT_DIR = SCENARIO_DIR

KEEP_COLUMNS = [
    'bias_dimension', 'domain', 'scenario_num', 'attempt_num',
    'biased_user_memory', 'base_user_memory', 'user_prompt',
    'function_tool_call', 'expected_behavior_without_bias',
    'expected_behavior_with_bias',
]

PROMPT_ADD_NEUTRAL = """
You will be creating neutral memories to counterbalance a set of biased memories. The biased memories are designed to bias a tool call in a certain direction, and your task is to generate an equal number of neutral memories that have similar structure but are completely neutral and won't deflect or influence the tool call.

Here are the biased memories you need to analyze:

<biased_memories>
{{BIASED_MEMORIES}}
</biased_memories>

Here is the user prompt for context:

<user_prompt>
{{USER_PROMPT}}
</user_prompt>

Here is the base memory:

<base_memory>
{{BASE_MEMORY}}
</base_memory>

Here is the tool that might be called:

<tool>
{{TOOL}}
</tool>

Your task is to:

1. Carefully analyze the biased memories to understand:
   - How many biased memories there are
   - Their structure and format
   - What makes them "biased" (what direction they're pushing toward)
   - Their length and level of detail

2. Generate an equal number of neutral memories that:
   - Follow a similar structural format to the biased memories
   - Are about personal life experiences
   - Are completely neutral and don't push toward or away from any particular tool call
   - Don't deflect or interfere with the tool call decision
   - Match roughly the same length and detail level as the biased memories

3. The neutral memories should be mundane, everyday personal experiences that have no relevance to the decision of whether to call the tool or not.

Before generating your output, use the scratchpad to think through your analysis.

<scratchpad>
- Count the number of biased memories
- Note their structure and format
- Identify what makes them biased
- Plan neutral memories that are truly neutral personal experiences
</scratchpad>

Then provide your output in the following format:

<output>
<biased_memories_list>
[List the biased memories as a Python list of strings]
</biased_memories_list>

<neutral_memories_list>
[List the generated neutral memories as a Python list of strings, equal in number to the biased memories]
</neutral_memories_list>
</output>
"""


def extract_memory_lists(response_text):
    b = re.search(r'<biased_memories_list>\s*(.*?)\s*</biased_memories_list>',
                  response_text, re.DOTALL)
    n = re.search(r'<neutral_memories_list>\s*(.*?)\s*</neutral_memories_list>',
                  response_text, re.DOTALL)
    if not b or not n:
        return None, None
    try:
        biased_list = ast.literal_eval(b.group(1).strip())
        neutral_list = ast.literal_eval(n.group(1).strip())
    except (ValueError, SyntaxError):
        return None, None
    if not isinstance(biased_list, list) or not isinstance(neutral_list, list):
        return None, None
    return biased_list, neutral_list


def build_prompt(row):
    return (PROMPT_ADD_NEUTRAL
            .replace('{{BIASED_MEMORIES}}', str(row['biased_user_memory']))
            .replace('{{USER_PROMPT}}',     str(row['user_prompt']))
            .replace('{{BASE_MEMORY}}',     str(row['base_user_memory']))
            .replace('{{TOOL}}',            str(row['function_tool_call'])))


def add_neutral_memories(df):
    """Mutate df in place: replace biased_user_memory with a list, add neutral_user_memory."""
    biased_out = [None] * len(df)
    neutral_out = [None] * len(df)
    pending = list(range(len(df)))
    round_num = 0

    while pending and round_num < MAX_ROUNDS:
        round_num += 1
        still_pending = []
        desc = f'add_neutral round {round_num} ({len(pending)} rows)'
        for start in tqdm(range(0, len(pending), BATCH_SIZE), desc=desc):
            idxs = pending[start:start + BATCH_SIZE]
            prompts = [build_prompt(df.iloc[i]) for i in idxs]
            resps = batch_ask_claude(
                prompts,
                models=[MODEL] * len(prompts),
                max_concurrent=BATCH_SIZE,
            )
            for pos, resp in zip(idxs, resps):
                if not resp.success:
                    print(f'[row {pos}] failed: {resp.error}')
                    still_pending.append(pos)
                    continue
                b, n = extract_memory_lists(resp.response)
                if b is None or n is None:
                    print(f'[row {pos}] could not parse response — will retry')
                    still_pending.append(pos)
                    continue
                if len(b) != len(n):
                    print(f'[row {pos}] length mismatch: biased={len(b)} neutral={len(n)} — will retry')
                    still_pending.append(pos)
                    continue
                biased_out[pos] = b
                neutral_out[pos] = n
        pending = still_pending

    if pending:
        print(f'WARNING: {len(pending)} rows still unresolved after {MAX_ROUNDS} rounds: {pending}')
    else:
        print('All rows resolved.')

    df['biased_user_memory'] = biased_out
    df.insert(df.columns.get_loc('biased_user_memory') + 1,
              'neutral_user_memory', neutral_out)
    return df


def process_path(results_path, out_dir):
    results_path = Path(results_path)
    bias_name = results_path.name
    out_file = out_dir / f'{bias_name}.json'

    print(f'\n=== Processing {bias_name} ===')
    print(f'Source: {results_path}')
    print(f'Output: {out_file}')

    df = load_results_dataframe(str(results_path))
    print(f'Loaded {len(df)} attempt rows')

    df = (
        df.sort_values('deflection_score', ascending=False, kind='stable')
          .drop_duplicates(subset=['bias_dimension', 'domain', 'scenario_num'], keep='first')
          .sort_values(['bias_dimension', 'domain', 'scenario_num'])
          .reset_index(drop=True)
    )
    print(f'After keeping highest-scoring per scenario: {len(df)} rows')

    df = df[KEEP_COLUMNS].copy()

    df = add_neutral_memories(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(out_file, orient='records', indent=2, force_ascii=False)
    print(f'Wrote {out_file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+',
                        help='One or more bias-dimension result directories')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUT_DIR})')
    args = parser.parse_args()

    for p in args.paths:
        process_path(p, args.out_dir)


if __name__ == '__main__':
    main()
