"""Render verified Go13 learning and stability/plasticity results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results'


def load(name):
    return json.loads((OUT / (name + '.json')).read_text())


def main():
    primary = load('go13_summary')
    transfer = load('go13_transfer_summary')
    assert primary['verified'] and transfer['verified']
    seeds = [911, 919, 929, 937, 941]
    methods = ['orth_free', 'orth_current', 'similar_free', 'similar_current', 'orth_stale', 'orth_norm', 'orth_frozen']
    labels = ['Orth / free', 'Orth / current', 'Similar / free', 'Similar / current', 'Orth / stale', 'Orth / norm', 'Orth / frozen']
    colors = ['#c25f3d', '#176f83', '#d29b43', '#65894d', '#8c6899', '#9b7270', '#797979']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'go13'})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), layout='constrained')
    for geometry, color, label in [('orth', colors[1], 'Orthogonal'), ('similar', colors[3], 'Similarity control')]:
        curves = [load(f'go13_prepared_seed{s}')['curves'][geometry] for s in seeds]
        x = [r['feedback'] for r in curves[0]]
        y = np.array([[r['a']['accuracy'] * 100 for r in curve] for curve in curves])
        axes[0].plot(x, y.mean(0), marker='o', color=color, label=label)
        axes[0].fill_between(x, y.min(0), y.max(0), color=color, alpha=.12)
    axes[0].axhline(50, ls='--', color='#777777', lw=1)
    axes[0].set(title='Primary: history acquisition failed', xlabel='Observed rule feedback',
                ylabel='Held-out ko accuracy (%)', ylim=(48, 58))
    axes[0].legend(frameon=False, loc='upper left')
    axes[0].text(.03, .06, 'Lines: five-seed mean\nBands: observed seed range', transform=axes[0].transAxes, color='#555555')
    for method, label, color in zip(methods, labels, colors):
        rows = transfer['table'][method]['per_seed']
        x = np.array([r['old_after'] for r in rows]) * 100
        y = np.array([r['new_after'] for r in rows]) * 100
        axes[1].scatter(x, y, s=18, alpha=.3, color=color)
        axes[1].scatter(x.mean(), y.mean(), s=95, color=color, marker='X', label=label)
    axes[1].axhline(50, ls='--', color='#777777', lw=1)
    axes[1].set(title='Follow-up: retention vs acquisition', xlabel='Old suicide-rule accuracy (%)',
                ylabel='New ko-history accuracy (%)', xlim=(47, 102), ylim=(48.5, 59))
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    values = [100 * transfer['table'][m]['accepted_updates'] / 640 for m in methods]
    axes[2].barh(np.arange(len(methods)), values, color=colors, height=.68)
    axes[2].set_yticks(np.arange(len(methods)), labels)
    axes[2].invert_yaxis()
    axes[2].set(title='Follow-up: accepted parameter steps', xlabel='Accepted / 640 update windows (%)', xlim=(0, 116))
    for i, (method, value) in enumerate(zip(methods, values)):
        axes[2].text(value + 1, i, f"{transfer['table'][method]['accepted_updates']}/640", va='center', fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Go13: measurable geometric guarantees do not yet establish joint online learning', fontsize=14)
    directory = ROOT / 'analysis/figures'
    directory.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        path = directory / f'go13_summary.{extension}'
        fig.savefig(path, dpi=180, metadata={'Date': None} if extension == 'svg' else None)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
