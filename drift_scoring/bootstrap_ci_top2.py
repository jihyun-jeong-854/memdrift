"""Per-cell CIs and paired gap tests for the top-2 variant.

Per cell (model x bias_dimension, n=21, and pooled n=105): mean, std, and a
percentile bootstrap 95% CI over scenarios (B=10000).

Gaps between two models are measured on the same scenarios, so the test is
paired — bootstrap CI of the mean paired difference, sign-flip permutation
p-value, BH-corrected. A gap survives when the CI excludes 0 and p_bh < 0.05.
Pooled cells also get a domain-cluster CI, resampling the 7 domains.

Usage:
    python -m drift_scoring.bootstrap_ci_top2
    python -m drift_scoring.bootstrap_ci_top2 --estimator cross_max
    python -m drift_scoring.bootstrap_ci_top2 --condition biased_minus_neutral
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SCORE_DIR as _SCORE_ROOT

# Per-file outputs of objective_drift_top2.py, and where the bootstrap lands.
SCORE_DIR = _SCORE_ROOT / "objective" / "top2"
OUT_PATH = SCORE_DIR / "bootstrap_stats.json"

DIMENSIONS = ["autonomy_self_reliance", "minimalism_conciseness", "resource_frugality",
              "risk_permissiveness", "speed_impatience"]

B = 10000
SEED = 20260728
CI = 0.95

# model config -> architecture family, for the cross-architecture comparison
FAMILY = {
    "gpt-5.2_medium": "openai_gpt", "gpt-5.4_medium": "openai_gpt",
    "gpt-5.4_none": "openai_gpt",
    "claude-sonnet-4-5_none": "anthropic_claude", "claude-sonnet-4-6_none": "anthropic_claude",
    "gemini-2.5-pro_medium": "google_gemini", "gemini-3.1-pro-preview_medium": "google_gemini",
    "Llama-3.3-70B-Instruct-Turbo_none": "meta_llama",
    "Qwen3.5-397B-A17B_medium": "alibaba_qwen",
    "Kimi-K2.5_medium": "moonshot_kimi",
}


# ------------------------------------------------------------------------ loading

def load_scores(estimator: str, condition: str):
    """-> {model: {scenario_key: value}}, plus the set of excluded files and why.

    scenario_key is (bias_dimension, domain, scenario_num) so the same scenario can
    be matched across models for the paired tests.
    """
    scores: dict[str, dict[tuple, float]] = defaultdict(dict)
    excluded: dict[str, str] = {}

    for path in sorted(SCORE_DIR.glob("*.json")):
        name = path.stem
        if name == OUT_PATH.stem:
            continue
        if "backup" in name:
            excluded[path.name] = "backup file"
            continue
        dim = next((d for d in DIMENSIONS if name.startswith(d)), None)
        if dim is None:
            excluded[path.name] = "filename does not start with a known bias dimension"
            continue
        model = name[len(dim) + 1:]

        data = json.loads(path.read_text())
        unscored = 0
        for row in data["rows"]:
            if "error" in row:
                unscored += 1
                continue
            value = row[condition][estimator]
            if value is None:
                unscored += 1
                continue
            scores[model][(row["bias_dimension"], row["domain"], row["scenario_num"])] = value
        if unscored:
            excluded[path.name] = (f"{unscored}/{len(data['rows'])} scenarios unscorable "
                                   f"(missing/'na' inference runs) -- model dropped")

    # a model is only comparable if every one of its files is complete
    incomplete = {m for m in scores
                  for f, why in excluded.items() if f.startswith(tuple(DIMENSIONS))
                  and f"_{m}.json" in f and "unscorable" in why}
    for m in incomplete:
        scores.pop(m, None)
    return dict(scores), excluded, sorted(incomplete)


# ---------------------------------------------------------------------- bootstrap

def boot_ci(values: list[float], rng: random.Random, b: int = B, ci: float = CI):
    """Percentile bootstrap CI for the mean."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return (None, None)
    n = len(vals)
    means = sorted(st.mean(rng.choices(vals, k=n)) for _ in range(b))
    lo = means[int((1 - ci) / 2 * b)]
    hi = means[min(b - 1, int((1 + ci) / 2 * b))]
    return (lo, hi)


def boot_ci_clustered(pairs: list[tuple[str, float]], rng: random.Random,
                      b: int = B, ci: float = CI):
    """Cluster bootstrap: resample whole domains, not individual scenarios."""
    clusters = defaultdict(list)
    for key, value in pairs:
        clusters[key].append(value)
    keys = list(clusters)
    if len(keys) < 2:
        return (None, None)
    means = []
    for _ in range(b):
        drawn = [v for k in rng.choices(keys, k=len(keys)) for v in clusters[k]]
        if drawn:
            means.append(st.mean(drawn))
    means.sort()
    n = len(means)
    return (means[int((1 - ci) / 2 * n)], means[min(n - 1, int((1 + ci) / 2 * n))])


def cell_stats(values: list[float], rng: random.Random, domains: list[str] | None = None):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    lo, hi = boot_ci(vals, rng)
    out = {
        "n": len(vals),
        "mean": st.mean(vals),
        "std": st.stdev(vals) if len(vals) > 1 else 0.0,
        "ci95": [lo, hi],
    }
    if domains is not None:
        clo, chi = boot_ci_clustered(list(zip(domains, vals)), rng)
        out["ci95_domain_cluster"] = [clo, chi]
    return out


def paired_gap(a: dict[tuple, float], b: dict[tuple, float], rng: random.Random):
    """Paired bootstrap CI + sign-flip permutation p for mean(a) - mean(b)."""
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    if len(diffs) < 2:
        return None
    n = len(diffs)
    observed = st.mean(diffs)

    boots = sorted(st.mean(rng.choices(diffs, k=n)) for _ in range(B))
    lo = boots[int((1 - CI) / 2 * B)]
    hi = boots[min(B - 1, int((1 + CI) / 2 * B))]

    # paired permutation: under H0 the sign of each scenario's difference is arbitrary
    extreme = sum(abs(st.mean([d if rng.random() < 0.5 else -d for d in diffs])) >= abs(observed)
                  for _ in range(B))
    p = (extreme + 1) / (B + 1)

    return {"n_paired": n, "delta": observed, "ci95": [lo, hi], "p_perm": p,
            "ci_excludes_zero": (lo > 0) or (hi < 0)}


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH-adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1  # BH rank, descending
        prev = min(prev, pvals[idx] * m / i)
        adj[idx] = prev
    return adj


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="cross_mean",
                    choices=["index_mean", "index_max", "cross_mean", "cross_max"])
    ap.add_argument("--condition", default="base_to_biased",
                    choices=["base_to_biased", "base_to_neutral", "biased_minus_neutral"])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rng = random.Random(SEED)
    scores, excluded, dropped = load_scores(args.estimator, args.condition)
    if not scores:
        raise SystemExit("no complete model configs found")

    # ---- cells: model x dimension, plus pooled
    cells = {}
    for model, per_key in sorted(scores.items()):
        entry = {}
        for dim in DIMENSIONS:
            keys = [k for k in per_key if k[0] == dim]
            entry[dim] = cell_stats([per_key[k] for k in keys], rng)
        keys = sorted(per_key)
        entry["pooled"] = cell_stats([per_key[k] for k in keys], rng,
                                     domains=[k[1] for k in keys])
        entry["family"] = FAMILY.get(model)
        cells[model] = entry

    # ---- cross-model gaps (all pairs, ranked by pooled mean)
    ranked = sorted(scores, key=lambda m: -cells[m]["pooled"]["mean"])
    model_gaps = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            gap = paired_gap(scores[a], scores[b], rng)
            if gap:
                model_gaps.append({"model_a": a, "model_b": b, **gap})

    # ---- cross-architecture gaps: average configs within a family per scenario
    fam_scores: dict[str, dict[tuple, float]] = defaultdict(dict)
    per_family_models = defaultdict(list)
    for model in scores:
        per_family_models[FAMILY.get(model, "unknown")].append(model)
    for family, members in per_family_models.items():
        keys = set.intersection(*(set(scores[m]) for m in members))
        for k in keys:
            fam_scores[family][k] = st.mean(scores[m][k] for m in members)

    fam_cells = {}
    for family, per_key in fam_scores.items():
        keys = sorted(per_key)
        fam_cells[family] = {**cell_stats([per_key[k] for k in keys], rng,
                                          domains=[k[1] for k in keys]),
                             "configs": sorted(per_family_models[family])}
    fam_ranked = sorted(fam_scores, key=lambda f: -fam_cells[f]["mean"])
    arch_gaps = []
    for i, a in enumerate(fam_ranked):
        for b in fam_ranked[i + 1:]:
            gap = paired_gap(fam_scores[a], fam_scores[b], rng)
            if gap:
                arch_gaps.append({"family_a": a, "family_b": b, **gap})

    # ---- BH correction within each gap family of tests
    for table in (model_gaps, arch_gaps):
        adj = benjamini_hochberg([g["p_perm"] for g in table])
        for gap, p in zip(table, adj):
            gap["p_bh"] = p
            gap["survives"] = bool(gap["ci_excludes_zero"] and p < args.alpha)

    out = {
        "config": {
            "estimator": args.estimator,
            "condition": args.condition,
            "n_bootstrap": B, "seed": SEED, "ci_level": CI, "alpha": args.alpha,
            "ci_method": "percentile bootstrap over scenarios",
            "gap_test": "paired: bootstrap CI + sign-flip permutation p, BH-corrected",
            "survives_rule": "bootstrap CI excludes 0 AND p_bh < alpha",
            "std_note": ("std is between-scenario heterogeneity, not uncertainty on the "
                         "mean; use ci95 for the latter"),
            "models_included": ranked,
            "models_dropped_incomplete": dropped,
            "files_excluded": excluded,
        },
        "cells": cells,
        "architecture_cells": fam_cells,
        "cross_model_gaps": model_gaps,
        "cross_architecture_gaps": arch_gaps,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- console view
    def ci(c):
        return "n/a" if c["ci95"][0] is None else f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"

    print(f"estimator={args.estimator}  condition={args.condition}  B={B}  seed={SEED}")
    if dropped:
        print(f"dropped (incomplete inference): {', '.join(dropped)}")
    print(f"\nPOOLED PER MODEL (n=105 scenarios)   mean  std   95% CI (scenario) "
          f"| 95% CI (domain-cluster)")
    for m in ranked:
        c = cells[m]["pooled"]
        dc = c.get("ci95_domain_cluster", [None, None])
        dcs = "n/a" if dc[0] is None else f"[{dc[0]:.3f}, {dc[1]:.3f}]"
        print(f"  {m:36} {c['mean']:.3f} {c['std']:.3f} {ci(c):>16} | {dcs}")

    print(f"\nPER ARCHITECTURE")
    for f in fam_ranked:
        c = fam_cells[f]
        print(f"  {f:20} n={c['n']:4} {c['mean']:.3f}  {ci(c):>16}  "
              f"configs={len(c['configs'])}")

    for label, table, ka, kb in (("CROSS-MODEL", model_gaps, "model_a", "model_b"),
                                 ("CROSS-ARCHITECTURE", arch_gaps, "family_a", "family_b")):
        surv = [g for g in table if g["survives"]]
        print(f"\n{label} GAPS: {len(surv)}/{len(table)} survive "
              f"(BH-corrected, alpha={args.alpha})")
        for g in sorted(table, key=lambda x: -abs(x["delta"])):
            mark = "SURVIVES" if g["survives"] else "   --   "
            print(f"  {mark}  {g[ka]:34} vs {g[kb]:34} "
                  f"d={g['delta']:+.3f} CI=[{g['ci95'][0]:+.3f},{g['ci95'][1]:+.3f}] "
                  f"p_bh={g['p_bh']:.4f}")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
