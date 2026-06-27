"""Phase 3 — Statistical analysis and figures.

Outputs:
  - data/results/accuracy_by_model.csv
  - data/results/accuracy_by_category.csv
  - paper/figures/fig1_accuracy_by_model.png
  - paper/figures/fig2_accuracy_by_category.png
  - paper/figures/fig3_scaling.png  (if model size data available)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "results" / "attribution_results.csv"
FIGURES_DIR = ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = 0.5

# Known parameter counts (billions) for scaling analysis
MODEL_PARAMS = {
    "meta/llama-3.1-8b-instruct": 8,
    "meta/llama-3.3-70b-instruct": 70,
    "mistralai/mistral-7b-instruct-v0.3": 7,
    "mistralai/mixtral-8x7b-instruct-v0.1": 47,
    "google/gemma-3-4b-it": 4,
    "google/gemma-3-12b-it": 12,
    "google/gemma-3-27b-it": 27,
    "microsoft/phi-4": 14,
    "qwen/qwen2.5-7b-instruct": 7,
    "qwen/qwen2.5-72b-instruct": 72,
}


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-instruct", "").replace("-it", "")


def accuracy_stats(group: pd.DataFrame) -> dict:
    decisive = group[group["model_answer"].isin(["A", "B"])]
    n = len(decisive)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan")}
    wins = decisive["correct"].sum()
    acc = wins / n
    lo, hi = proportion_confint(wins, n, alpha=0.05, method="wilson")
    p = binomtest(wins, n, BASELINE, alternative="two-sided").pvalue
    return {"n": n, "accuracy": acc, "ci_lo": lo, "ci_hi": hi, "p_value": p}


def plot_accuracy_by_model(stats_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue" if row["p_value"] < 0.05 else "salmon" for _, row in stats_df.iterrows()]
    bars = ax.barh(stats_df["short_name"], stats_df["accuracy"] * 100, color=colors)
    ax.errorbar(
        stats_df["accuracy"] * 100,
        range(len(stats_df)),
        xerr=[
            (stats_df["accuracy"] - stats_df["ci_lo"]) * 100,
            (stats_df["ci_hi"] - stats_df["accuracy"]) * 100,
        ],
        fmt="none", color="black", capsize=4,
    )
    ax.axvline(50, color="black", linestyle="--", linewidth=1.2, label="Chance baseline (50%)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("LLM Self-Attribution Accuracy by Model\n(blue = p<0.05 vs chance, red = not significant)")
    ax.set_xlim(0, 100)
    ax.legend()
    for bar, (_, row) in zip(bars, stats_df.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{row['accuracy']*100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig1_accuracy_by_model.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def plot_accuracy_by_category(cat_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cat_df["category"], cat_df["accuracy"] * 100, color="steelblue")
    ax.axhline(50, color="black", linestyle="--", linewidth=1.2, label="Chance baseline (50%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Self-Attribution Accuracy by Prompt Category")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_accuracy_by_category.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def plot_scaling(stats_df: pd.DataFrame) -> None:
    has_params = stats_df["params_b"].notna()
    sub = stats_df[has_params]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub["params_b"], sub["accuracy"] * 100, s=120, color="steelblue", zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row["short_name"], (row["params_b"], row["accuracy"] * 100),
                    textcoords="offset points", xytext=(6, 0), fontsize=8)
    ax.axhline(50, color="black", linestyle="--", linewidth=1.2, label="Chance baseline (50%)")
    ax.set_xlabel("Model size (B parameters)")
    ax.set_ylabel("Self-attribution accuracy (%)")
    ax.set_title("Self-Attribution vs Model Size")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_scaling.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("No results yet. Run 02_attribute.py first.")

    df = pd.read_csv(RESULTS_PATH)
    print(f"Loaded {len(df)} attribution results")
    print(f"Invalid/error answers: {(~df['model_answer'].isin(['A','B'])).sum()}")

    # Accuracy by model
    rows = []
    for model, group in df.groupby("model"):
        stats = accuracy_stats(group)
        rows.append({"model": model, "short_name": short_name(model),
                     "params_b": MODEL_PARAMS.get(model), **stats})
    stats_df = pd.DataFrame(rows).sort_values("accuracy", ascending=True).reset_index(drop=True)
    stats_df.to_csv(ROOT / "data" / "results" / "accuracy_by_model.csv", index=False)
    print("\n=== Accuracy by model ===")
    for _, r in stats_df.iterrows():
        sig = "**" if r["p_value"] < 0.05 else "  "
        print(f"{sig} {r['short_name']:30s} {r['accuracy']*100:5.1f}%  "
              f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}]  p={r['p_value']:.3f}  n={r['n']}")

    # Accuracy by category
    cat_rows = []
    for cat, group in df.groupby("category"):
        stats = accuracy_stats(group)
        cat_rows.append({"category": cat, **stats})
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(ROOT / "data" / "results" / "accuracy_by_category.csv", index=False)
    print("\n=== Accuracy by category ===")
    print(cat_df[["category", "accuracy", "n", "p_value"]].to_string(index=False))

    # Figures
    print("\nGenerating figures...")
    plot_accuracy_by_model(stats_df)
    plot_accuracy_by_category(cat_df)
    plot_scaling(stats_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
