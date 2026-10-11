"""Parses the per-attempt logs written by run_refinement_loop.py.

Walks one bias dimension's log tree —
results/construction_logs/<bias>/<domain>/<scenario>/attempt_*.txt — and
returns one DataFrame row per attempt, carrying the generated artifacts, the
construction target model's responses under both conditions, and the judge's
deflection score. select_best_attempts.py uses that score to pick the surviving
attempt per scenario.

Usage:
    python -m scenario_creation.attempt_log_parser results/construction_logs/speed_impatience
"""

import re
import json
from pathlib import Path

import pandas as pd


def _extract_tag(text, tag):
    m = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_top_level_field(block, field, remaining_fields):
    start_m = re.search(rf'^  "{re.escape(field)}": ', block, re.MULTILINE)
    if not start_m:
        return None
    start = start_m.end()

    end = len(block)
    for nf in remaining_fields:
        nf_m = re.search(rf'^  "{re.escape(nf)}": ', block[start:], re.MULTILINE)
        if nf_m:
            end = min(end, start + nf_m.start())
    close_m = re.search(r'^\}', block[start:], re.MULTILINE)
    if close_m:
        end = min(end, start + close_m.start())

    value = block[start:end].rstrip().rstrip(',').rstrip()

    if value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        inner = (inner.replace('\\"', '"')
                      .replace('\\n', '\n')
                      .replace('\\t', '\t')
                      .replace('\\\\', '\\'))
        return inner
    if value.startswith('{') or value.startswith('['):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _split_sections(content):
    markers = [
        'SCENARIO DETAILS',
        'DETAILED CONTEXT',
        'BASE MEMORY RESPONSES',
        'BIASED MEMORY RESPONSES',
        'JUDGE RESULT',
    ]
    positions = []
    for m in markers:
        mm = re.search(rf'^{re.escape(m)}\n-+\n', content, re.MULTILINE)
        if mm:
            positions.append((m, mm.end(), mm.start()))

    positions.sort(key=lambda x: x[1])
    sections = {}
    for i, (name, body_start, _) in enumerate(positions):
        if i + 1 < len(positions):
            end = positions[i + 1][2]
        else:
            eol = re.search(r'={5,}\s*\nEND OF LOG', content)
            end = eol.start() if eol else len(content)
        sections[name] = content[body_start:end]
    return sections


def _parse_response_blocks(section_text):
    blocks = re.findall(r'Response \d+:\s*\n(\{[\s\S]*?\n\})', section_text)
    out = []
    for b in blocks:
        try:
            out.append(json.loads(b))
        except json.JSONDecodeError:
            out.append(b)
    return out


def parse_attempt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = _split_sections(content)

    field_order = [
        'scratchpad',
        'biased_user_memory',
        'base_user_memory',
        'user_prompt',
        'function_tool_call',
        'expected_behavior_without_bias',
        'expected_behavior_with_bias',
    ]
    detailed = sections.get('DETAILED CONTEXT', '')
    parsed = {
        f: _extract_top_level_field(detailed, f, field_order[i + 1:])
        for i, f in enumerate(field_order)
    }

    parsed['base_memory_responses'] = _parse_response_blocks(
        sections.get('BASE MEMORY RESPONSES', '')
    )
    parsed['biased_memory_responses'] = _parse_response_blocks(
        sections.get('BIASED MEMORY RESPONSES', '')
    )

    judge = sections.get('JUDGE RESULT', '')

    score = _extract_tag(judge, 'deflection_score')
    if score is not None:
        try:
            score = int(score)
        except ValueError:
            pass
    parsed['deflection_score'] = score

    dflag = _extract_tag(judge, 'deflected_at_least_once')
    if isinstance(dflag, str):
        low = dflag.strip().lower()
        if low == 'true':
            dflag = True
        elif low == 'false':
            dflag = False
    parsed['deflected_at_least_once'] = dflag

    parsed['justification'] = _extract_tag(judge, 'justification')

    return parsed


def load_results_dataframe(bias_path):
    """
    Walk <bias_path>/<domain>/<scenario_n>/attempt_*.txt
    and return a pandas DataFrame with one row per attempt.
    The final path component is treated as the bias dimension.
    """
    bias_dir = Path(bias_path)
    rows = []

    for domain_dir in sorted(p for p in bias_dir.iterdir() if p.is_dir()):
        for scenario_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            for attempt_file in sorted(scenario_dir.glob('attempt_*.txt')):
                try:
                    p = parse_attempt_file(attempt_file)
                except Exception as e:
                    print(f'[WARN] failed to parse {attempt_file}: {e}')
                    continue

                rows.append({
                    'bias_dimension': bias_dir.name,
                    'domain': domain_dir.name,
                    'scenario_num': scenario_dir.name,
                    'attempt_num': attempt_file.stem,
                    'biased_user_memory': p.get('biased_user_memory'),
                    'base_user_memory': p.get('base_user_memory'),
                    'user_prompt': p.get('user_prompt'),
                    'function_tool_call': p.get('function_tool_call'),
                    'expected_behavior_without_bias': p.get('expected_behavior_without_bias'),
                    'expected_behavior_with_bias': p.get('expected_behavior_with_bias'),
                    'base_memory_responses': p.get('base_memory_responses'),
                    'biased_memory_responses': p.get('biased_memory_responses'),
                    'deflection_score': p.get('deflection_score'),
                    'deflected_at_least_once': p.get('deflected_at_least_once'),
                    'justification': p.get('justification'),
                })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.paths import CONSTRUCTION_LOG_DIR

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('results_path', nargs='?', default=None,
                        help='One bias dimension\'s log directory '
                             '(default: the only one under results/construction_logs)')
    args = parser.parse_args()

    path = args.results_path or str(CONSTRUCTION_LOG_DIR)
    df = load_results_dataframe(path)
    print(f'Loaded {len(df)} rows')
    print(df[['bias_dimension', 'domain', 'scenario_num', 'attempt_num',
              'deflection_score', 'deflected_at_least_once']].to_string())
