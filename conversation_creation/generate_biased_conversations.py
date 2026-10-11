#!/usr/bin/env python3
"""Expand the biased memory set M_b of each benchmark entry into conversations.

The direct-injection setting hands the model M_b as declarative memory text.
The memory-architecture setting cannot: mem0, MemPalace and SimpleMem ingest
dialogue, not memory statements. This stage therefore converts each memory in
M_b into a short casual user/assistant conversation, which the memory
architectures then store and retrieve on their own terms.

Writes data/conversations/<bias_dimension>.json — every original column plus
`biased_conversation`, a list of {biased_memory_idx, biased_memory,
conversation} records, one per memory in M_b.

By default only `scenario_1` of each domain is expanded (7 entries per bias
dimension, 35 overall), which is the subset the memory-architecture experiments
run on. Pass --scenario all to expand all 21 entries per dimension.

Usage:
    python -m conversation_creation.generate_biased_conversations speed_impatience
    python -m conversation_creation.generate_biased_conversations speed_impatience --n_turns 3
    python -m conversation_creation.generate_biased_conversations speed_impatience --scenario all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CONFIG_DIR, conversation_file, scenario_file
from conversation_creation.biased_conversation_simulator import initial_attempt


def load_config():
    with open(CONFIG_DIR / 'conversation_init_config.json') as f:
        init_config = json.load(f)
    with open(CONFIG_DIR / 'conversation_skills.json') as f:
        skills = json.load(f)
    return init_config, skills


def generate(bias_dimension, n_turns=3, scenario='scenario_1', output_path=None):
    init_config, skills = load_config()

    input_path = scenario_file(bias_dimension)
    if not input_path.exists():
        raise FileNotFoundError(f'Scenarios file not found: {input_path}')

    with open(input_path) as f:
        all_scenarios = json.load(f)

    if scenario == 'all':
        scenarios = all_scenarios
        print(f'Loaded {len(all_scenarios)} entries from {input_path}; expanding all')
    else:
        scenarios = [s for s in all_scenarios if s.get('scenario_num') == scenario]
        print(f'Loaded {len(all_scenarios)} entries from {input_path}; '
              f'keeping {len(scenarios)} ({scenario} only, one per domain)')

    out = Path(output_path) if output_path else conversation_file(bias_dimension, 'biased')
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, entry in enumerate(scenarios):
        label = f"scenario={entry.get('scenario_num', i)}  domain={entry.get('domain', '?')}"
        print(f'\n[{i+1}/{len(scenarios)}] {label}')

        conversations = initial_attempt(
            scenario    = entry,
            init_config = init_config,
            tips        = skills,
            n_turns     = n_turns,
        )

        biased_conversation = [
            {
                'biased_memory_idx': conv['tendency_idx'],
                'biased_memory':     conv['tendency'],
                'conversation':      conv['conversation'],
            }
            for conv in conversations
        ]

        results.append({**entry, 'biased_conversation': biased_conversation})
        print(f'  → {len(biased_conversation)} conversations stored')

        # Incremental write so a crash does not lose completed entries.
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'  → saved progress ({i+1}/{len(scenarios)}) → {out}')

    print(f'\nDone. Final output → {out}')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bias_dimension',
        help='Bias dimension name, e.g. speed_impatience')
    parser.add_argument('--n_turns', type=int, default=3,
        help='User/assistant turn pairs per memory (default: 3)')
    parser.add_argument('--scenario', default='scenario_1',
        help="Which scenario_num to expand, or 'all' (default: scenario_1)")
    parser.add_argument('--output',
        help='Output JSON path (default: data/conversations/<bias_dimension>.json)')
    args = parser.parse_args()

    generate(
        bias_dimension = args.bias_dimension,
        n_turns        = args.n_turns,
        scenario       = args.scenario,
        output_path    = args.output,
    )


if __name__ == '__main__':
    main()
