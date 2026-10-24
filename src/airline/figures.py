"""Figures for the README and the project page, drawn from reports/*.json and the raw data.

    python -m airline.figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import ROOT, load_competition, load_original  # noqa: E402
from .run import REPORTS  # noqa: E402

IMG = ROOT / "docs" / "img"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["lgbm_raw", "lgbm_values", "xgb_values", "catboost_values", "rank_mean", "stack"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def models() -> None:
    blend = json.loads((REPORTS / "blend.json").read_text())["scores"]
    lb = {r["name"]: float(r["public_score"]) for r in json.loads((REPORTS / "leaderboard.json").read_text())}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    ypos = np.arange(len(ORDER))[::-1]
    ax = axes[0]
    ax.scatter([blend[n]["cv_auc"] for n in ORDER], ypos, s=60, color=BLUE, label="5-fold CV", zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.scatter([lb[n] for n in ORDER], ypos, s=60, color=ORANGE, marker="D", label="Kaggle public LB",
               zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_yticks(ypos, ORDER)
    ax.set_title("Competition data (synthetic)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("ROC AUC", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax = axes[1]
    ax.scatter([blend[n]["original_auc"] for n in ORDER], ypos, s=60, color=AQUA, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_title("Same models scored on the real survey (129,880 rows)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("ROC AUC", color=MUTED, fontsize=9)
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(IMG / "models.png", dpi=150)
    plt.close(fig)


def noise() -> None:
    st = json.loads((REPORTS / "original_study.json").read_text())["label_noise_equivalent"]
    c = pd.DataFrame(st["curve"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(c.flip_rate * 100, c.auc, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(st["stack_auc_competition_cv"], color=ORANGE, linewidth=1.5, linestyle="--")
    ax.text(c.flip_rate.max() * 100, st["stack_auc_competition_cv"] + 0.0015,
            f"competition CV {st['stack_auc_competition_cv']:.4f}", ha="right", color=MUTED, fontsize=9)
    r = st["flip_rate_matching_competition_auc"]
    ax.annotate(f"{r:.1%} of labels flipped", xy=(r * 100, c.set_index("flip_rate").auc[r]),
                xytext=(r * 100 + 1.2, st["stack_auc_original"] - 0.004), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("share of real-survey labels flipped at random (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("stack AUC on the real survey", color=MUTED, fontsize=9)
    ax.set_title("Random label noise needed to turn 0.994 into the competition's 0.961", loc="left",
                 fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(IMG / "noise.png", dpi=150)
    plt.close(fig)


def shift() -> dict:
    X, y, _ = load_competition()
    Xo, yo = load_original()
    bins = [-1, 0, 15, 60, 10_000]
    labels = ["0 min", "1-15", "16-60", "> 60"]
    col = "Departure Delay in Minutes"
    share = pd.DataFrame({
        "competition": pd.cut(X[col], bins, labels=labels).value_counts(normalize=True),
        "original": pd.cut(Xo[col], bins, labels=labels).value_counts(normalize=True),
    }).loc[labels]
    sat = pd.DataFrame({
        "competition": y.groupby(pd.cut(X[col], bins, labels=labels), observed=True).mean(),
        "original": yo.groupby(pd.cut(Xo[col], bins, labels=labels), observed=True).mean(),
    }).loc[labels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(labels))
    for ax, df, title in ((axes[0], share, "Share of passengers by departure delay"),
                          (axes[1], sat, "Share satisfied, by departure delay")):
        ax.bar(x - 0.2, df["competition"], 0.38, color=BLUE, label="competition (synthetic)")
        ax.bar(x + 0.2, df["original"], 0.38, color=AQUA, label="original survey")
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "shift.png", dpi=150)
    plt.close(fig)
    return {"delay_share": share.round(4).to_dict(), "satisfied_by_delay": sat.round(4).to_dict()}


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    models()
    noise()
    (REPORTS / "shift.json").write_text(json.dumps(shift(), indent=2))


if __name__ == "__main__":
    main()
