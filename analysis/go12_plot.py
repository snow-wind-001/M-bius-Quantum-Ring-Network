"""Render the verified Go12 measurements, without model fitting or smoothing."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def main():
    data = json.loads((ROOT / "analysis/results/go12_summary.json").read_text())
    assert data["verified"]
    legacy = json.loads((ROOT / "analysis/results/go12_legacy_summary.json").read_text())
    assert legacy["verified"]
    data["table"]["legacy"] = {
        "wins": legacy["wins"], "win_rate": legacy["wins"] / legacy["games"],
        "per_seed_wins": [row["wins"] for row in legacy["rows"]],
    }
    methods = ("frozen", "legacy", "baseline", "ownership", "credit", "combined", "unprotected")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = ["Frozen", "Old cadence", "Baseline", "Ownership", "Credit 64", "Combined", "No protection"]
    colors = ["#b2b2b2", "#47765e", "#3f6e91", "#8b9a5b", "#8b79a5", "#b86d3e", "#878787"]
    x = np.arange(len(methods))
    rates = [100 * data["table"][method]["win_rate"] for method in methods]
    axes[0].bar(x, rates, color=colors, alpha=.85, width=.65)
    for index, method in enumerate(methods):
        row = data["table"][method]
        axes[0].scatter(index + np.linspace(-.19, .19, 5),
                        np.array(row["per_seed_wins"]) * 100 / 48,
                        s=18, facecolors="white", edgecolors="#333333", linewidths=.7, zorder=3)
        label_y = max(row["per_seed_wins"]) * 100 / 48 + 3.5
        axes[0].text(index, label_y, f'{row["wins"]}/240', ha="center", fontsize=8)
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Win rate (%)")
    axes[0].set_title("Final: 240 complete games per method\nDots: five independent training seeds", fontsize=11)
    for method, color in (("baseline", colors[2]), ("combined", colors[5])):
        rows = data["learning_curves_matched_120_games"][method]
        axes[1].plot([r["online_games"] for r in rows], [100 * r["win_rate"] for r in rows],
                     "o-", label=method.capitalize(), color=color, linewidth=1.8, markersize=5)
    zero = data["learning_curves_matched_120_games"]["baseline"][0]["win_rate"] * 100
    axes[1].axhline(zero, color=colors[0], linestyle="--", label="Frozen")
    axes[1].set_xticks([0, 4, 8, 16])
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Completed online self-play games")
    axes[1].set_ylabel("Win rate (%)")
    axes[1].set_title("Learning curve: same 120 evaluation games\n24 teacher games precede online learning", fontsize=11)
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("10 x 10 Go | Six orthogonal rings | 16 search simulations, depth 8", fontsize=12)
    directory = ROOT / "analysis/figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "go12_efficiency.png", dpi=180)
    svg_path = directory / "go12_efficiency.svg"
    fig.savefig(svg_path)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
