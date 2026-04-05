"""Figure 3: All 5 schedule families visualized."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from figures.style import setup_style, save_figure, COLORS


def main():
    setup_style()

    from schedules import make_schedule

    N = 100
    families = [
        ('Monotonic', 'monotonic', {}),
        ('Single Reheat', 'single_reheat', {'t_reheat': 0.4, 'delta': 0.15}),
        ('Sawtooth', 'sawtooth', {'period': 25, 'sawtooth_delta': 0.08}),
        ('Damped Oscillation', 'damped_oscillation', {'A': 0.2, 'gamma': 2.5, 'freq': 4}),
        ('Adaptive (base)', 'monotonic', {}),
    ]

    colors = [
        COLORS['monotonic_ddim'],
        COLORS['single_reheat'],
        COLORS['sawtooth'],
        COLORS['damped_oscillation'],
        COLORS['adaptive'],
    ]

    fig, axes = plt.subplots(1, 5, figsize=(14, 2.5), sharey=True)

    for ax, (title, family, kwargs), color in zip(axes, families, colors):
        sched = make_schedule(family, N=N, **kwargs)
        t = np.linspace(0, 1, N)

        ax.plot(t, sched, color=color, linewidth=1.8)

        # Shade reheat regions (where alpha_bar decreases)
        diffs = np.diff(sched)
        for i in range(len(diffs)):
            if diffs[i] < -1e-4:
                ax.axvspan(t[i], t[i + 1], alpha=0.25, color='#E8734A', linewidth=0)

        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$t$ (sampling progress)')

        if ax == axes[0]:
            ax.set_ylabel(r'$\bar{\alpha}(t)$')

    # Annotate adaptive panel
    axes[4].annotate('Reheats triggered\nadaptively at runtime',
                     xy=(0.5, 0.5), xycoords='axes fraction',
                     ha='center', va='center', fontsize=7, style='italic',
                     color=COLORS['adaptive'])

    fig.tight_layout()
    save_figure(fig, 'fig3_schedule_families')


if __name__ == '__main__':
    main()
