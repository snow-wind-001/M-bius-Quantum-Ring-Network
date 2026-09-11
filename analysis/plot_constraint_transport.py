"""Plot measured conditional-ring memory and latency from verified results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = json.loads((ROOT / "analysis/results/go_constraint_summary.json").read_text())
    measured = raw["mechanisms"]["conditional"]
    lengths = sorted(map(int, measured))
    rows = [measured[str(length)] for length in lengths]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    blue, orange, gray = "#27658c", "#bd662c", "#656d76"
    for field, label, color in (
        ("autograd_saved_bytes", "Autograd: saved for backward", blue),
        ("transport_saved_bytes", "Transport: saved for backward", orange),
        ("transport_state_trace_bytes", "Transport: detached prefix states", gray),
    ):
        axes[0].plot(lengths, [row[field] / 1024 for row in rows],
                     marker="o", color=color, label=label, linewidth=2)
    axes[0].set(ylabel="Tensor storage (KiB)", xlabel="Observed prefix length",
                title="Memory accounts shown separately")
    axes[0].legend(frameon=False, fontsize=8)
    for field, label, color in (
        ("autograd_seconds", "Autograd", blue),
        ("transport_seconds", "Explicit transport", orange),
    ):
        axes[1].plot(lengths, [1000 * row[field] for row in rows],
                     marker="o", color=color, label=label, linewidth=2)
    axes[1].set(ylabel="Gradient construction (ms)", xlabel="Observed prefix length",
                title="Latency measured without memory hooks")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(lengths)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("Input-conditioned orthogonal rings | five seeds", fontsize=13)
    figure.text(0.5, 0.02,
                "Saved tensors include referenced inputs/parameters; this is not process RSS or total training memory.\n"
                "Prefix states may overlap saved tensors. CPU timings are not a dedicated deployment benchmark.",
                ha="center", fontsize=8, color=gray)
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    directory = ROOT / "analysis/figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        figure.savefig(directory / f"constraint_transport.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
