"""Rebuild report figures and evidence tables from accepted, blinded scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-luna")
CONDITIONS = tuple("ABCDEF")
DIMENSIONS = ("meaning", "imagery", "tone", "english", "faithful_fit", "poetic_fit")
LABELS = ("含义", "意象", "情感语气", "英语表达", "忠实译介适合度", "诗歌朗读适合度")
COLORS = ("#176B87", "#C46A35", "#6D5497")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        raise ValueError("No matched poem observations")
    rng = np.random.default_rng(20261025)
    means = rng.choice(a, size=(2000, len(a)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(a.mean()), "low": float(low), "high": float(high), "n_poems": len(a)}


def extract_evidence(rows):
    primary = [r for r in rows if r["judge_variant"] == "primary" and r["split"] == "test"]
    grouped = defaultdict(list)
    for r in primary:
        grouped[(r["poem_id"], r["model"], r["condition"])].append(r)
    means = {
        key: {d: float(np.mean([float(r[d]) for r in group])) for d in DIMENSIONS}
        for key, group in grouped.items()
    }
    poems = sorted({p for p, _, _ in means})
    # Each poem is the sampling unit. Repetitions have already been averaged.
    quality, context, purpose = [], [], []
    for model in MODELS:
        for condition in CONDITIONS:
            for dim in DIMENSIONS:
                values = [means[(p, model, condition)][dim] for p in poems]
                quality.append({"model": model, "condition": condition, "dimension": dim, **mean_ci(values)})
        for a, b in (("C", "E"), ("D", "F")):
            for dim in DIMENSIONS:
                values = [means[(p, model, b)][dim] - means[(p, model, a)][dim] for p in poems]
                context.append({"model": model, "contrast": f"{b}-{a}", "dimension": dim, **mean_ci(values)})
        for a, b in (("C", "D"), ("E", "F")):
            swaps, expected, ties = 0, 0, 0
            categories = {k: 0 for k in ("both_up", "both_down", "reading_up_faith_down",
                                        "faith_up_reading_down", "one_tie", "both_tie")}
            for p in poems:
                faith = means[(p, model, b)]["faithful_fit"] - means[(p, model, a)]["faithful_fit"]
                poetic = means[(p, model, b)]["poetic_fit"] - means[(p, model, a)]["poetic_fit"]
                swaps += faith * poetic < 0
                expected += faith < 0 and poetic > 0
                ties += faith == 0 or poetic == 0
                if faith == 0 and poetic == 0:
                    category = "both_tie"
                elif faith == 0 or poetic == 0:
                    category = "one_tie"
                elif faith > 0 and poetic > 0:
                    category = "both_up"
                elif faith < 0 and poetic < 0:
                    category = "both_down"
                elif poetic > 0:
                    category = "reading_up_faith_down"
                else:
                    category = "faith_up_reading_down"
                categories[category] += 1
            purpose.append({"model": model, "contrast": f"{b}-{a}", "n_poems": len(poems),
                            "opposite_preferences": int(swaps), "faithful_vs_reading_direction": int(expected),
                            "tie_on_either_goal": int(ties), "categories": categories})
    indexed = {(r["judge_variant"], r["task_id"]): r for r in rows}
    agreement = []
    for variant in ("secondary", "reversed"):
        pairs = [(r, indexed[("primary", r["task_id"])]) for r in rows
                 if r["judge_variant"] == variant and ("primary", r["task_id"]) in indexed]
        for dim in DIMENSIONS:
            if pairs:
                differences = [float(a[dim]) - float(b[dim]) for a, b in pairs]
                agreement.append({"variant": variant, "dimension": dim, "n_translations": len(pairs),
                                  "mean_difference": float(np.mean(differences)),
                                  "mean_absolute_difference": float(np.mean(np.abs(differences))),
                                  "exact_agreement": float(np.mean(np.asarray(differences) == 0))})
    coverage = []
    for model in MODELS:
        for condition in CONDITIONS:
            poem_rates = defaultdict(list)
            for (poem, m, c), group in grouped.items():
                if m != model or c != condition:
                    continue
                for r in group:
                    units = json.loads(r["critical_units_json"])
                    for status in ("present", "partial", "missing", "wrong"):
                        poem_rates[(poem, status)].append(sum(u["status"] == status for u in units) / len(units))
            for status in ("present", "partial", "missing", "wrong"):
                values = [np.mean(poem_rates[(p, status)]) for p in poems]
                coverage.append({"model": model, "condition": condition, "status": status, **mean_ci(values)})
    return {"aggregation": "Within poem/model/condition average repetitions; bootstrap 16 poem clusters, 2000 draws.",
            "quality": quality, "context": context, "purpose_preferences": purpose,
            "judge_sensitivity": agreement, "critical_unit_coverage": coverage}


def draw_figures(evidence, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    quality = {(r["model"], r["condition"], r["dimension"]): r for r in evidence["quality"]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout="constrained")
    x = np.arange(6)
    for ax, dim, label in zip(axes.flat, DIMENSIONS, LABELS):
        for i, (model, color) in enumerate(zip(MODELS, COLORS)):
            rs = [quality[(model, c, dim)] for c in CONDITIONS]
            ys = np.array([r["mean"] for r in rs])
            errs = np.array([[r["mean"]-r["low"] for r in rs], [r["high"]-r["mean"] for r in rs]])
            ax.errorbar(x+(i-1)*.12, ys, yerr=errs, marker="o", markersize=4, capsize=2,
                        color=color, label=model, linewidth=1.1, alpha=.9)
        ax.set(title=label, xticks=x, xticklabels=CONDITIONS, ylim=(.9, 5.12), yticks=[1, 2, 3, 4, 5])
        ax.grid(axis="y", alpha=.17)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.supxlabel("A 普通翻译 · B 文学角色 · C 忠实译介 · D 诗歌朗读 · E 译介+解读 · F 朗读+解读")
    fig.supylabel("评分（1—5；误差线为按诗词重采样的95%区间）")
    fig.savefig(output_dir / "quality_dimensions.png")
    plt.close(fig)

    context = {(r["model"], r["contrast"], r["dimension"]): r for r in evidence["context"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, contrast, title in zip(axes, ("E-C", "F-D"), ("忠实译介：增加中文解读", "诗歌朗读：增加中文解读")):
        for i, (model, color) in enumerate(zip(MODELS, COLORS)):
            rs = [context[(model, contrast, d)] for d in DIMENSIONS]
            xs = [r["mean"] for r in rs]
            errs = np.array([[r["mean"]-r["low"] for r in rs], [r["high"]-r["mean"] for r in rs]])
            ax.errorbar(xs, np.arange(6)+(i-1)*.17, xerr=errs, fmt="o", capsize=3, color=color, label=model)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set(title=title, yticks=np.arange(6), yticklabels=LABELS, xlabel="同一诗词的评分变化（增加解读 − 原条件）")
        ax.grid(axis="x", alpha=.16)
    axes[0].invert_yaxis()
    axes[1].legend(loc="best", fontsize=8)
    limits = [axes[0].get_xlim(), axes[1].get_xlim()]
    bound = max(abs(v) for lim in limits for v in lim)
    for ax in axes:
        ax.set_xlim(-bound, bound)
    fig.savefig(output_dir / "context_effects.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    categories = (("both_up", "两项均提高", "#4F8A78"),
                  ("both_down", "两项均降低", "#C77868"),
                  ("reading_up_faith_down", "朗读提高、译介降低", "#567EB6"),
                  ("faith_up_reading_down", "译介提高、朗读降低", "#9274AE"),
                  ("one_tie", "一项持平", "#B9A56C"),
                  ("both_tie", "两项均持平", "#D2D7DD"))
    purposes = {(r["model"], r["contrast"]): r for r in evidence["purpose_preferences"]}
    for ax, contrast, title in zip(axes, ("D-C", "F-E"),
                                   ("无中文解读：译介任务 → 朗读任务", "有中文解读：译介任务 → 朗读任务")):
        left = np.zeros(3)
        for key, label, color in categories:
            counts = np.array([purposes[(m, contrast)]["categories"][key] for m in MODELS])
            ax.barh(np.arange(3), counts, left=left, color=color, label=label, height=.6)
            for i, count in enumerate(counts):
                if count:
                    ax.text(left[i]+count/2, i, str(count), ha="center", va="center", fontsize=9)
            left += counts
        ax.set(title=title, yticks=np.arange(3), yticklabels=MODELS, xlim=(0, 16), xticks=np.arange(0, 17, 2))
        ax.invert_yaxis()
    axes[-1].legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=3, fontsize=9, frameon=False)
    fig.supxlabel("诗词数量（每模型16篇，先平均三次生成；比较两种用途适合度的评分变化）")
    fig.savefig(output_dir / "purpose_comparison.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, default=ROOT / "results" / "summary")
    p.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = p.parse_args()
    rows = read_csv(args.summary_dir / "judge_scores_long.csv")
    primary = [r for r in rows if r["judge_variant"] == "primary" and r["split"] == "test"]
    if len(primary) != 864 or len({r["task_id"] for r in primary}) != 864:
        raise ValueError("Report plots require all 864 distinct primary evaluations")
    evidence = extract_evidence(rows)
    (args.summary_dir / "figure_data.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    draw_figures(evidence, args.output_dir)
    print(json.dumps({"figures": [str(p) for p in sorted(args.output_dir.glob('*.png'))], "primary_translations": 864}, ensure_ascii=False))


if __name__ == "__main__":
    main()
