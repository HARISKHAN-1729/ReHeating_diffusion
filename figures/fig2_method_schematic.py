"""Figure 2: Method schematic — monotonic vs reheating trajectories."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from figures.style import setup_style, save_figure, COLORS


def main():
    setup_style()

    from schedules import make_schedule

    N = 30
    mono = make_schedule('monotonic', N=N)
    reheat = make_schedule('single_reheat', N=N, t_reheat=0.4, delta=0.2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    steps = np.arange(N)

    # Monotonic
    ax.plot(steps, mono, color=COLORS['monotonic_ddim'], linewidth=2,
            label='Monotonic (DDIM)', zorder=3)

    # Reheating
    ax.plot(steps, reheat, color=COLORS['single_reheat'], linewidth=2,
            label='Reheating schedule', zorder=3)

    # Shade reheat regions
    diffs = np.diff(reheat)
    for i in range(len(diffs)):
        if diffs[i] < -1e-4:  # Alpha decreasing = reheat step
            ax.axvspan(i, i + 1, alpha=0.2, color='#E8734A', zorder=1)
            ax.annotate('', xy=(i + 0.5, reheat[i + 1]), xytext=(i + 0.5, reheat[i]),
                        arrowprops=dict(arrowstyle='->', color='#E8734A', lw=1.5))

    ax.set_xlabel('Sampling step index')
    ax.set_ylabel(r'$\bar{\alpha}$ (signal retention)')
    ax.set_title('Monotonic vs. Reheating Noise Schedule')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, N - 1)

    # Annotate
    ax.annotate('Reheat\n(add noise)', xy=(N * 0.38, 0.35), fontsize=8,
                color='#E8734A', ha='center', style='italic')

    fig.tight_layout()
    save_figure(fig, 'fig2_method_schematic')


if __name__ == '__main__':
    main()
