"""
Static diagrams for the report and slides:
  reports/figures/pipeline.png   Figure 1 - training/evaluation pipeline and the app's scan path
"""

from __future__ import annotations

import matplotlib.patches as mpatches

from src import plotstyle
from src.config import path


def _box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                         linewidth=1.4, edgecolor=color, facecolor="white"))
    ax.add_patch(mpatches.Rectangle((x, y + h - 0.012), w, 0.012, color=color, linewidth=0))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=plotstyle.INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.6, color=plotstyle.INK2,
            linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2, color=None, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color or plotstyle.INK2, lw=1.3, shrinkA=2, shrinkB=2))


def pipeline() -> None:
    plotstyle.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue, orange, green, violet = plotstyle.SERIES[0], plotstyle.SERIES[1], plotstyle.SERIES[2], plotstyle.SERIES[6]
    # lanes
    ax.add_patch(mpatches.FancyBboxPatch((0.005, 0.03), 0.655, 0.94, boxstyle="round,pad=0,rounding_size=0.02",
                                         facecolor="#f3f6fb", edgecolor="none"))
    ax.add_patch(mpatches.FancyBboxPatch((0.675, 0.03), 0.32, 0.94, boxstyle="round,pad=0,rounding_size=0.02",
                                         facecolor="#fdf3ee", edgecolor="none"))
    ax.text(0.02, 0.945, "OFFLINE: training and evaluation", fontsize=11, fontweight="bold", color=blue)
    ax.text(0.69, 0.945, "ONLINE: Streamlit app (on Scan)", fontsize=11, fontweight="bold", color=orange)

    w, h = 0.19, 0.12
    # row 1: data
    _box(ax, 0.02, 0.76, w, h, "Dataset A: Slither audited", "120,608 verified contracts\nraw Slither detector output", blue)
    _box(ax, 0.235, 0.76, w, h, "Clean + tokenise", "comments blanked (lines kept)\nliterals -> <STR>/<NUM>", blue)
    _box(ax, 0.45, 0.76, w, h, "Labels + de-duplication", "detectors -> 7 classes\nSHA-256 + MinHash-LSH groups", blue)
    # row 2
    _box(ax, 0.45, 0.56, w, h, "Group-aware split", "70 / 15 / 15 by clone group\n+ random split (E2)", blue)
    _box(ax, 0.235, 0.56, w, h, "TF-IDF features", "token 1-3-grams, fit on TRAIN\n+ SVD-512", blue)
    _box(ax, 0.02, 0.56, w, h, "Train + select", "prior, LR, tree, forest, KNN, MLP\nvalidation grid, F2 thresholds", violet)
    # row 3
    _box(ax, 0.02, 0.33, w, h, "Freeze: git tag model-v1", "model, thresholds, calibration\nbefore any test evaluation", violet)
    _box(ax, 0.235, 0.33, w, h, "Test split (E1, E2, E4)", "per-class P/R/F1, PR-AUC\nbootstrap intervals", green)
    _box(ax, 0.45, 0.33, w, h, "Dataset B: SmartBugs (E3)", "143 human-annotated contracts\nannotations stripped first", green)
    _box(ax, 0.235, 0.11, w * 2 + 0.025, h, "Explanation quality and errors (E3b, E5)",
         "line hit-rate@1/@3 vs annotated lines; latency; failure case studies", green)
    _arrow(ax, 0.21, 0.82, 0.235, 0.82)
    _arrow(ax, 0.425, 0.82, 0.45, 0.82)
    _arrow(ax, 0.545, 0.76, 0.545, 0.68)
    _arrow(ax, 0.45, 0.62, 0.425, 0.62)
    _arrow(ax, 0.235, 0.62, 0.21, 0.62)
    _arrow(ax, 0.115, 0.56, 0.115, 0.45)
    _arrow(ax, 0.21, 0.39, 0.235, 0.39)
    _arrow(ax, 0.425, 0.39, 0.45, 0.39)
    _arrow(ax, 0.33, 0.33, 0.33, 0.23)

    # app lane
    aw = 0.27
    ax_ = 0.70
    _box(ax, ax_, 0.79, aw, 0.1, "1. Paste / upload .sol", "validation: empty, too large, not Solidity", orange)
    _box(ax, ax_, 0.65, aw, 0.1, "2. Same cleaning + TF-IDF", "identical code path as training", orange)
    _box(ax, ax_, 0.51, aw, 0.1, "3. predict()", "probabilities, flags, risk level", orange)
    _box(ax, ax_, 0.37, aw, 0.1, "4. explain() - occlusion", "blank functions, then lines; rank by drop", orange)
    _box(ax, ax_, 0.23, aw, 0.1, "5. similar()", "3 nearest training contracts (cosine)", orange)
    _box(ax, ax_, 0.08, aw, 0.11, "6. Result page + report", "risk badge, bars, highlighted code,\ndownload, disclaimer", orange)
    for y in (0.79, 0.65, 0.51, 0.37, 0.23):
        _arrow(ax, ax_ + aw / 2, y, ax_ + aw / 2, y - 0.04)
    # frozen model -> app: routed below the evaluation boxes so it crosses nothing
    ax.plot([0.02, 0.008, 0.008, 0.668, 0.668], [0.39, 0.39, 0.055, 0.055, 0.56], color=violet, lw=1.3)
    _arrow(ax, 0.668, 0.56, 0.70, 0.56, color=violet)
    ax.text(0.30, 0.062, "frozen model files (models/) are loaded by the app", fontsize=8, color=violet)
    plotstyle.save(fig, path("figures_dir") / "pipeline.png")


if __name__ == "__main__":
    pipeline()
    print("figures written")
