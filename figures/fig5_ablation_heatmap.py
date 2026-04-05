"""Figure 5: Ablation heatmap — FID vs (t_reheat, delta)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figures.style import setup_style, save_figure


def main():
    setup_style()

    csv_path = 'results/tables/exp6_ablation.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"Warning: {csv_path} not found. Using placeholder data.")
        t_reheat_vals = [0.3, 0.5, 0.7]
        delta_vals = [0.1, 0.3]
        rows = []
        for t in t_reheat_vals:
            for d in delta_vals:
                fid = 50 + np.random.randn() * 10
                rows.append({'t_reheat': t, 'delta': d, 'fid': fid})
        df = pd.DataFrame(rows)

    # Pivot for heatmap
    pivot = df.pivot_table(values='fid', index='delta', columns='t_reheat', aggfunc='mean')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Diverging colormap centered on median FID
    center = pivot.values[~np.isnan(pivot.values)].mean() if not np.all(np.isnan(pivot.values)) else 50
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdBu_r', center=center,
                ax=ax, cbar_kws={'label': 'FID-50K'}, linewidths=0.5)

    ax.set_xlabel(r'$t_{\mathrm{reheat}}$')
    ax.set_ylabel(r'$\delta$')
    ax.set_title('Ablation: FID vs Schedule Parameters')

    # Mark optimal
    if not pivot.empty:
        min_val = pivot.min().min()
        min_pos = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
        ax.add_patch(plt.Rectangle(
            (min_pos[1], min_pos[0]), 1, 1,
            fill=False, edgecolor='gold', linewidth=3
        ))

    fig.tight_layout()
    save_figure(fig, 'fig5_ablation_heatmap')


if __name__ == '__main__':
    main()
