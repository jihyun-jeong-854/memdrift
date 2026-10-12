#!/usr/bin/env python3
"""Paired neutral/biased bar charts with scenario-bootstrap CIs.

One pair of bars per model: the light bar is the neutral condition, the
saturated bar the biased one, and the delta is printed above. Each model keeps
its own hue so a model is the same color wherever it appears across the paper's
figures — color follows the entity, never its rank, so a figure that drops a
model must not repaint the survivors. Order the models once, in MODEL_ORDER,
and leave it alone.

Two layouts, matching the two evaluation settings:

    --layout single   one panel, every model side by side (direct injection)
    --layout facet    one panel per memory architecture, same models in each

Error bars are the 95% percentile bootstrap over scenarios from
bootstrap_ci.py — uncertainty on the mean, not between-scenario spread. The
delta above each pair is the *paired* bootstrap, which is the quantity the
paper claims; reading significance off whether the two bars' error bars overlap
is the conservative wrong test and understates the effect.

Input : bootstrap_ci.json from drift_scoring.bootstrap_ci
Output: <out-dir>/<name>.{pdf,png}

Usage:
    python -m drift_scoring.plot_drift results/scores/objective/all_params/bootstrap_ci.json \\
        --layout single --panel-label "Direct Memory Injection" --name direct

    python -m drift_scoring.plot_drift results/scores/objective/all_params/bootstrap_ci.json \\
        --layout facet --facet-key setting --name memory_architectures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Validated reference palette, categorical slots 1-8 in fixed order.
# Worst adjacent pair: CVD dE 9.1, normal-vision dE 19.6 (gates 8 / 15).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e3e0"
SURFACE = "#fcfcfb"

# Stable hue assignment and display names. A model keeps its slot across every
# figure; adding one appends to the end rather than reshuffling.
MODEL_ORDER = [
    ("gpt-5.2", "GPT-5.2\nMed"),
    ("gpt-5.4", "GPT-5.4\nMed"),
    ("claude-sonnet-4-5", "Claude-Sonnet-\n4.5"),
    ("gemini-2.5-pro", "Gemini-2.5-\nPro Med"),
    ("gemini-3.1-pro-preview", "Gemini-3.1-\nPro Preview Med"),
    ("Kimi-K2.5", "Kimi-K2.5\nMed"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-\nA17B Med"),
    ("Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-\n70B Instruct"),
]
# Hue per model, chosen so the two models that appear *alone together* in the
# memory-architecture and ratio figures are maximally separable. The default
# slot order put GPT-5.4 on orange and Gemini-3.1 on magenta, which measures
# normal-vision dE 12.9 — below the 15 floor, i.e. genuinely hard to tell apart
# side by side. Orange vs violet is 37.6 (CVD 29.5). The remaining hues are
# permuted so every adjacent pair in display order still clears the gates:
# worst adjacent CVD 9.2, normal 15.6.
MODEL_COLOR = {
    "gpt-5.2":                      "#2a78d6",   # blue
    "gpt-5.4":                      "#eb6834",   # orange   <- focus model
    "claude-sonnet-4-5":            "#1baf7a",   # aqua
    "gemini-2.5-pro":               "#008300",   # green
    "gemini-3.1-pro-preview":       "#4a3aa7",   # violet   <- focus model
    "Kimi-K2.5":                    "#e87ba4",   # magenta
    "Qwen3.5-397B-A17B":            "#eda100",   # yellow
    "Llama-3.3-70B-Instruct-Turbo": "#e34948",   # red
}
MODEL_LABEL = dict(MODEL_ORDER)
# Grouped bars need the two-line names to fit under a tick; a horizontal row
# layout has the width for one line, and wrapping there just breaks the hyphens.
MODEL_LABEL_1L = {
    "gpt-5.2": "GPT-5.2", "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4-5": "Claude-Sonnet-4.5", "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro-Preview", "Kimi-K2.5": "Kimi-K2.5",
    "Qwen3.5-397B-A17B": "Qwen3.5-397B-A17B",
    "Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B-Instruct",
}

FACET_LABEL = {"mem0": "Mem0", "mempalace": "MemPalace", "simplemem": "SimpleMem",
               "direct": "Direct Memory Injection"}


def load(path, facet_key=None):
    rows = [r for r in json.load(open(path)) if r.get("dimension", "overall") == "overall"]
    if facet_key is None:
        return {None: rows}
    out = {}
    for r in rows:
        out.setdefault(r[facet_key], []).append(r)
    return out


def _label_ink(fill):
    """Black or white, whichever has more contrast against the fill.

    Hard-coding white means the lighter hues (yellow at 2.2:1, magenta at 2.7:1)
    carry unreadable labels, which in turn constrains which model may wear which
    color. Picking per bar removes that coupling.
    """
    r, g, b = (int(fill.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    on_white = (1.05) / (L + 0.05)
    on_black = (L + 0.05) / 0.05
    return "#ffffff" if on_white >= on_black else "#0b0b0b"


def _err(cells, key):
    mean = [c[f"{key}_mean"] for c in cells]
    lo = [m - c[f"{key}_ci_lo"] for m, c in zip(mean, cells)]
    hi = [c[f"{key}_ci_hi"] - m for m, c in zip(mean, cells)]
    return mean, [lo, hi]


def draw_panel(ax, rows, panel_label=None, show_values=True):
    rows = sorted(rows, key=lambda r: [m for m, _ in MODEL_ORDER].index(r["model"])
                  if r["model"] in MODEL_LABEL else 99)
    n = len(rows)
    x = list(range(n))
    w = 0.34
    gap = 0.02            # 2px surface gap between the paired fills

    n_mean, n_err = _err(rows, "neutral")
    b_mean, b_err = _err(rows, "biased")

    for i, r in enumerate(rows):
        c = MODEL_COLOR.get(r["model"], PALETTE[0])
        ax.bar(i - w / 2 - gap / 2, n_mean[i], w, color=c, alpha=0.30,
               edgecolor=c, linewidth=1.1, zorder=2)
        ax.bar(i + w / 2 + gap / 2, b_mean[i], w, color=c,
               edgecolor=SURFACE, linewidth=1.2, zorder=2)

    ax.errorbar([i - w / 2 - gap / 2 for i in x], n_mean, yerr=n_err, fmt="none",
                ecolor=TEXT_PRIMARY, elinewidth=1.1, capsize=3, zorder=4)
    ax.errorbar([i + w / 2 + gap / 2 for i in x], b_mean, yerr=b_err, fmt="none",
                ecolor=TEXT_PRIMARY, elinewidth=1.1, capsize=3, zorder=4)

    if show_values:
        for i, r in enumerate(rows):
            # neutral value above its own error bar
            ax.annotate(f"{n_mean[i]:.3f}", (i - w / 2 - gap / 2, n_mean[i] + n_err[1][i]),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=8, color=TEXT_PRIMARY)
            # biased value inside the bar — the bar is tall enough to hold it
            ax.annotate(f"{b_mean[i]:.3f}", (i + w / 2 + gap / 2, b_mean[i] * 0.5),
                        ha="center", va="center", fontsize=8.5,
                        color=_label_ink(MODEL_COLOR.get(r["model"], PALETTE[0])),
                        fontweight="bold")
            # The paired delta — the quantity the paper claims. Kept in ink, not
            # the series hue: the bar directly beneath already carries identity,
            # and the lighter slots (yellow at 2.1:1, aqua at 2.7:1) are not
            # legible as text on this surface.
            ax.annotate(f"+{r['delta_mean']:.3f}", (i + w / 2 + gap / 2, b_mean[i] + b_err[1][i]),
                        textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=9, color=TEXT_PRIMARY, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(r["model"], r["model"]) for r in rows], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.65, n - 0.35)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)

    if panel_label:
        ax.annotate(panel_label, (0.015, 0.955), xycoords="axes fraction",
                    ha="left", va="top", fontsize=10, fontweight="bold",
                    color=TEXT_PRIMARY,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0efec",
                              edgecolor=GRID, linewidth=0.8))


def draw_dumbbell(ax, rows, label_key=None, sort_by_delta=True):
    """Arrow from the neutral value to the biased value, one row per entity.

    Paired bars make the reader subtract two heights; the arrow *is* the
    subtraction. It also puts every neutral dot in one vertical band near zero,
    which is the figure's second claim — personal memory on its own moves
    nothing, so the spread on the right is content, not presence.

    Rows are sorted by effect size for reading order, but hue stays bound to the
    model: sorting must never repaint a series.
    """
    rows = list(rows)
    if sort_by_delta:
        rows.sort(key=lambda r: r["delta_mean"])
    y = list(range(len(rows)))

    for i, r in enumerate(rows):
        c = MODEL_COLOR.get(r["model"], PALETTE[0])
        n, b = r["neutral_mean"], r["biased_mean"]

        # CI whiskers first, so the marks sit on top of them.
        ax.plot([r["neutral_ci_lo"], r["neutral_ci_hi"]], [i, i],
                color=TEXT_SECONDARY, linewidth=1.0, alpha=0.55, zorder=2)
        ax.plot([r["biased_ci_lo"], r["biased_ci_hi"]], [i, i],
                color=TEXT_SECONDARY, linewidth=1.0, alpha=0.55, zorder=2)

        # The arrowhead is the terminus — a filled dot under it just collides.
        ax.annotate("", xy=(b, i), xytext=(n, i), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=c, linewidth=2.6,
                                    shrinkA=6, shrinkB=0, mutation_scale=20))
        ax.plot(n, i, "o", markersize=8, markerfacecolor=SURFACE,
                markeredgecolor=c, markeredgewidth=2, zorder=4)

        ax.annotate(f"+{r['delta_mean']:.3f}", (r["biased_ci_hi"], i),
                    textcoords="offset points", xytext=(9, 0), va="center",
                    fontsize=9, color=TEXT_PRIMARY, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r[label_key] if label_key else
                        MODEL_LABEL_1L.get(r["model"], r["model"])
                        for r in rows], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    return rows


def dumbbell_legend():
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=SURFACE, markeredgecolor="#6e6e6a", markeredgewidth=2,
               label=r"neutral memory  $d(\mathcal{M}_n)$"),
        Line2D([], [], marker=">", linestyle="none", markersize=10,
               color="#6e6e6a", label=r"biased memory  $d(\mathcal{M}_b)$"),
        Line2D([], [], color=TEXT_SECONDARY, linewidth=1.0, alpha=0.55,
               label="95% CI"),
    ]


DIM_LABEL = {
    "speed_impatience": "Speed /\nImpatience",
    "risk_permissiveness": "Risk\nPermissiveness",
    "minimalism_conciseness": "Minimalism /\nConciseness",
    "resource_frugality": "Resource\nFrugality",
    "autonomy_self_reliance": "Autonomy /\nSelf-Reliance",
    "overall": "Overall",
}
# Ordinal: no personal memory -> memory on an untrusted channel -> memory the
# model is told to trust. A sequential lightness ramp of the model's own hue,
# light to dark, matches that order; three unrelated hues would not.
PRIVILEGE_STEPS = [("neutral", 0.22), ("unprivileged", 0.55), ("privileged", 1.0)]


def draw_privilege(ax, rows, model, dims):
    by_dim = {r["dimension"]: r for r in rows if r["model"] == model}
    base = MODEL_COLOR.get(model, PALETTE[0])
    x = list(range(len(dims)))
    w = 0.26

    for j, (cond, alpha) in enumerate(PRIVILEGE_STEPS):
        off = (j - 1) * (w + 0.015)
        mean = [by_dim[d][f"{cond}_mean"] for d in dims]
        lo = [m - by_dim[d][f"{cond}_ci_lo"] for m, d in zip(mean, dims)]
        hi = [by_dim[d][f"{cond}_ci_hi"] - m for m, d in zip(mean, dims)]
        ax.bar([i + off for i in x], mean, w, color=base, alpha=alpha,
               edgecolor=base if alpha < 1 else SURFACE, linewidth=1.1, zorder=2)
        ax.errorbar([i + off for i in x], mean, yerr=[lo, hi], fmt="none",
                    ecolor=TEXT_PRIMARY, elinewidth=1.0, capsize=2.5, zorder=4)

    # One label per group: the privileged -> unprivileged reduction, which is
    # what the figure is for. "n.s." where the paired CI still contains zero.
    for i, d in enumerate(dims):
        r = by_dim[d]
        top = max(r[f"{c}_ci_hi"] for c, _ in PRIVILEGE_STEPS)
        ns = r["reduction_ci_hi"] >= 0
        ax.annotate(f"{r['reduction_mean']:+.3f}" + ("  n.s." if ns else ""),
                    (i, top), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8.5, fontweight="bold",
                    color=TEXT_SECONDARY if ns else TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABEL.get(d, d) for d in dims], fontsize=8.5)
    # Headroom for the reduction labels — Gemini's bars reach 0.88 and the
    # label sits above the error bar.
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.6, len(dims) - 0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    # Separate the pooled column from the per-dimension ones.
    if "overall" in dims:
        ax.axvline(dims.index("overall") - 0.5, color=GRID, linewidth=1.2)


def privilege_legend():
    """One shared legend in neutral ink.

    The lightness ramp means the same thing in every panel — only the hue
    changes — so repeating it per panel is noise, and with Gemini's bars
    reaching 0.88 a per-panel legend lands on top of the data.
    """
    names = {"neutral": r"neutral  $d(\mathcal{M}_n)$",
             "unprivileged": "biased, unprivileged channel",
             "privileged": "biased, privileged channel"}
    return [Patch(facecolor="#6e6e6a", alpha=a,
                  edgecolor="#6e6e6a" if a < 1 else SURFACE,
                  label=names[c]) for c, a in PRIVILEGE_STEPS]


def legend_handles():
    return [
        Patch(facecolor="#b8b8b4", edgecolor="#8a8a86", alpha=0.55,
              label=r"left: neutral drift rate $d(\mathcal{M}_n)$"),
        Patch(facecolor="#6e6e6a", edgecolor=SURFACE,
              label=r"right: biased drift rate $d(\mathcal{M}_b)$"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bootstrap_ci", type=Path,
                        help="bootstrap_ci.json from drift_scoring.bootstrap_ci")
    parser.add_argument("--layout", default="single",
                        choices=["single", "facet", "dumbbell", "privilege"])
    parser.add_argument("--facet-key", default="setting",
                        help="Column that defines a panel in facet layout")
    parser.add_argument("--facet-order", nargs="+", default=None)
    parser.add_argument("--panel-label", default=None,
                        help="Badge text for the single-panel layout")
    parser.add_argument("--name", default="drift")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    groups = load(args.bootstrap_ci,
                  args.facet_key if args.layout == "facet" else None)

    if args.layout == "privilege":
        rows = json.load(open(args.bootstrap_ci))
        models = [m for m, _ in MODEL_ORDER if any(r["model"] == m for r in rows)]
        dims = [d for d in DIM_LABEL if any(r["dimension"] == d for r in rows)]
        fig, axes = plt.subplots(len(models), 1,
                                 figsize=(1.55 * len(dims) + 2.2, 3.5 * len(models)))
        if len(models) == 1:
            axes = [axes]
        for ax, m in zip(axes, models):
            draw_privilege(ax, rows, m, dims)
            ax.set_ylabel(r"Drift rate $d(\mathcal{M})$", color=TEXT_SECONDARY)
            ax.set_title(MODEL_LABEL_1L.get(m, m), loc="left", pad=8,
                         fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY)
        fig.legend(handles=privilege_legend(), loc="upper center", ncol=3,
                   frameon=True, edgecolor=GRID, facecolor=SURFACE, fontsize=9,
                   bbox_to_anchor=(0.5, 1.05), columnspacing=1.6,
                   handlelength=1.4, handleheight=1.0)
    elif args.layout == "dumbbell":
        rows = [r for r in json.load(open(args.bootstrap_ci))
                if r.get("dimension", "overall") == "overall"]
        # When the rows carry a facet column (one row per setting x model), the
        # row label has to name both; otherwise the model alone identifies it.
        label_key = None
        if args.facet_key in rows[0]:
            for r in rows:
                r["_label"] = (f"{FACET_LABEL.get(r[args.facet_key], r[args.facet_key])}"
                               f"  ·  {MODEL_LABEL_1L.get(r['model'], r['model'])}")
            label_key = "_label"
        fig, ax = plt.subplots(figsize=(8.8, 0.52 * len(rows) + 2.0))
        draw_dumbbell(ax, rows, label_key)
        ax.set_xlabel(r"Drift rate $d(\mathcal{M})$", color=TEXT_SECONDARY)
        if args.panel_label:
            ax.set_title(args.panel_label, color=TEXT_PRIMARY, fontsize=11,
                         loc="left", pad=12, fontweight="bold")
        ax.legend(handles=dumbbell_legend(), frameon=False, fontsize=9, loc="lower right")
    elif args.layout == "single":
        rows = next(iter(groups.values()))
        fig, ax = plt.subplots(figsize=(1.55 * len(rows) + 1.6, 4.2))
        draw_panel(ax, rows, args.panel_label)
        ax.set_ylabel(r"Drift rate $d(\mathcal{M})$", color=TEXT_SECONDARY)
        fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
                   frameon=True, edgecolor=GRID, facecolor=SURFACE,
                   fontsize=9.5, bbox_to_anchor=(0.5, 1.10))
    else:
        keys = args.facet_order or sorted(groups)
        keys = [k for k in keys if k in groups]
        fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.0), sharey=True)
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            draw_panel(ax, groups[k], FACET_LABEL.get(k, k))
        axes[0].set_ylabel(r"Drift rate $d(\mathcal{M})$", color=TEXT_SECONDARY)
        fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
                   frameon=True, edgecolor=GRID, facecolor=SURFACE,
                   fontsize=9.5, bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = args.out_dir / f"{args.name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}")


if __name__ == "__main__":
    main()
