"""Render the actual seed-paired adaptation/retention trade-off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    global_result = json.loads((ROOT / "analysis/results/orthogonal_go_online_5seed.json").read_text())
    spatial_result = json.loads((ROOT / "analysis/results/orthogonal_go_online_spatial_5seed.json").read_text())
    names = {
        "frozen": "Frozen CNN", "spatial_sgd": "Update CNN", "stateless": "Stateless adapter",
        "identity": "Identity ring", "unistochastic": "Squared-magnitude ring",
        "orthogonal": "Signed ring", "orthogonal_ogd": "Signed ring + OGD",
        "identity_ogd": "Identity ring + OGD", "gru": "GRU",
    }
    rows = []
    for prefix, payload in (("Global", global_result), ("Spatial", spatial_result)):
        for method in payload["config"]["methods"]:
            rows.append((f"{prefix}: {names[method]}", method, payload["summary"][method]))
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "mqr-go-909"})
    figure, axes = plt.subplots(1, 2, figsize=(12, 7.2), sharey=True)
    for ax, key, title in zip(
        axes, ("b_nll_gain", "a_forgetting_nll"),
        ("New-task NLL gain (higher is better)", "Old-task NLL increase (lower is better)"),
    ):
        for y, (label, method, summary) in enumerate(rows):
            statistic = summary[key]
            mean = statistic["mean"]
            low, high = statistic["ci95"]
            color = "#b86024" if method.endswith("_ogd") else "#285f83"
            ax.errorbar(mean, y, xerr=np.array([[mean - low], [high - mean]]),
                        color=color, fmt="o", markersize=5, capsize=3, linewidth=1.5)
        ax.axvline(0, color="#626262", linewidth=0.8, linestyle="--")
        ax.axhline(8.5, color="#bbbbbb", linewidth=0.7)
        ax.set_title(title, loc="left", fontsize=11, pad=15)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Change in nats")
    axes[0].set_yticks(range(len(rows)), [row[0] for row in rows])
    axes[0].invert_yaxis()
    figure.suptitle("Small Go model: online adaptation and retention", x=0.31, ha="left", fontsize=15)
    figure.text(0.31, 0.922, "5 paired seeds; 95% descriptive bootstrap intervals; orange = OGD", color="#555555")
    figure.subplots_adjust(left=0.31, right=0.98, top=0.85, bottom=0.10, wspace=0.20)
    output = ROOT / "analysis/figures"
    output.mkdir(parents=True, exist_ok=True)
    svg_path = output / "orthogonal_go_tradeoff.svg"
    figure.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    figure.savefig(output / "orthogonal_go_tradeoff.png", dpi=160)
    print(output / "orthogonal_go_tradeoff.svg")


if __name__ == "__main__":
    main()
