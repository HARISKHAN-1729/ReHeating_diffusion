"""
Figure 1 -- Hero/Teaser (3-panel, full width).

Panel (a): 2D density contours with sampling trajectories.
Panel (b): Mode coverage vs NFE (from exp1_toy2d.csv).
Panel (c): FID vs NFE on CIFAR-10 (from exp2_cifar_ddpm.csv).
Shared legend at bottom.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

from figures.style import (
    setup_style, save_figure, COLORS, COLORS_LIGHT,
    METHOD_LABELS, METHOD_MARKERS, METHOD_ORDER,
)


# ---------------------------------------------------------------------------
# 8-Gaussian ring mixture utilities
# ---------------------------------------------------------------------------

def _ring_centers(n_modes=8, radius=4.0):
    """Return (n_modes, 2) array of Gaussian centers on a circle."""
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _mixture_density(xx, yy, centers, std=0.5):
    """Evaluate mixture density on a meshgrid."""
    density = np.zeros_like(xx)
    for c in centers:
        density += np.exp(-0.5 * ((xx - c[0])**2 + (yy - c[1])**2) / std**2)
    density /= (density.sum() + 1e-12)
    return density


def _make_placeholder_trajectories(centers, n_methods=5, n_traj=5, n_steps=25):
    """Generate placeholder trajectories that look plausible."""
    rng = np.random.RandomState(42)
    trajectories = {}
    method_keys = ['monotonic_ddim', 'monotonic_ddpm', 'single_reheat',
                   'damped_oscillation', 'adaptive']
    for mk in method_keys:
        trajs = []
        for _ in range(n_traj):
            # Start from random noise, end near a mode center
            target = centers[rng.randint(len(centers))]
            start = rng.randn(2) * 5
            t_arr = np.linspace(0, 1, n_steps)
            path = np.outer(1 - t_arr, start) + np.outer(t_arr, target)
            # Add perturbation -- more for reheating methods
            amp = 0.3 if mk.startswith('monotonic') else 0.9
            noise = rng.randn(n_steps, 2) * amp * (1 - t_arr[:, None])
            if mk in ('single_reheat', 'damped_oscillation', 'adaptive'):
                # Add oscillation to simulate reheating
                osc = amp * np.sin(np.linspace(0, 4 * np.pi, n_steps))
                noise[:, 0] += osc * (1 - t_arr)
            path += noise
            trajs.append(path)
        trajectories[mk] = trajs
    return trajectories


def _load_trajectories():
    """Load trajectory data or generate placeholders."""
    npz_path = os.path.join('results', 'tables', 'exp1_trajectories.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)
    print(f"[fig1_hero] {npz_path} not found, using placeholder trajectories.")
    centers = _ring_centers()
    return _make_placeholder_trajectories(centers)


def _load_mode_coverage():
    """Load mode coverage data from exp1_toy2d.csv or use placeholders."""
    csv_path = os.path.join('results', 'tables', 'exp1_toy2d.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    print(f"[fig1_hero] {csv_path} not found, using placeholder mode coverage data.")
    rng = np.random.RandomState(42)
    rows = []
    methods = ['monotonic_ddim', 'monotonic_ddpm', 'single_reheat',
               'damped_oscillation', 'adaptive']
    nfe_vals = [10, 25, 50]
    base_coverage = {
        'monotonic_ddim':     [5, 7, 8],
        'monotonic_ddpm':     [6, 7, 8],
        'single_reheat':      [7, 8, 8],
        'damped_oscillation': [6, 8, 8],
        'adaptive':           [7, 8, 8],
    }
    for m in methods:
        for i, nfe in enumerate(nfe_vals):
            rows.append({
                'method': m,
                'nfe': nfe,
                'modes_covered': base_coverage[m][i],
                'w2_distance': rng.uniform(0.2, 2.0) / (nfe / 10),
            })
    return pd.DataFrame(rows)


def _load_cifar_fid():
    """Load CIFAR-10 FID data from exp2_cifar_ddpm.csv or use placeholders."""
    csv_path = os.path.join('results', 'tables', 'exp2_cifar_ddpm.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Map config_key to method if needed
        if 'config_key' in df.columns and 'method' not in df.columns:
            df = df.rename(columns={'config_key': 'method'})
        elif 'config' in df.columns and 'method' not in df.columns:
            df = df.rename(columns={'config': 'method'})
        return df
    print(f"[fig1_hero] {csv_path} not found, using placeholder FID data.")
    rows = []
    methods = ['monotonic_ddim', 'monotonic_ddpm', 'single_reheat',
               'damped_oscillation', 'adaptive']
    nfe_vals = [10, 25, 50, 100]
    base_fid = {
        'monotonic_ddim':     [55.0, 18.0, 8.5, 5.2],
        'monotonic_ddpm':     [48.0, 15.0, 7.2, 4.8],
        'single_reheat':      [42.0, 13.5, 7.0, 4.6],
        'damped_oscillation': [45.0, 14.0, 7.5, 4.9],
        'adaptive':           [40.0, 12.0, 6.5, 4.3],
    }
    for m in methods:
        for i, nfe in enumerate(nfe_vals):
            rows.append({'method': m, 'nfe': nfe, 'fid': base_fid[m][i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def main():
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                             gridspec_kw={'width_ratios': [1.2, 1, 1]})

    # --- Panel (a): 2D trajectories on density contours ---
    ax = axes[0]
    centers = _ring_centers()
    lim = 7
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 200),
                          np.linspace(-lim, lim, 200))
    density = _mixture_density(xx, yy, centers, std=0.5)

    # Filled contours as soft background
    ax.contourf(xx, yy, density, levels=15, cmap='Greys', alpha=0.35)
    ax.contour(xx, yy, density, levels=8, colors='#999999', linewidths=0.3,
               alpha=0.5)

    trajectories = _load_trajectories()
    traj_methods = ['monotonic_ddim', 'single_reheat', 'damped_oscillation',
                    'adaptive']
    for mk in traj_methods:
        if mk not in trajectories:
            continue
        trajs = trajectories[mk]
        # Ensure we have a list of arrays
        if isinstance(trajs, np.ndarray) and trajs.ndim == 3:
            trajs = [trajs[j] for j in range(trajs.shape[0])]
        for tj in trajs[:5]:
            arr = np.array(tj)
            ax.plot(arr[:, 0], arr[:, 1], color=COLORS[mk],
                    linewidth=0.8, alpha=0.7, zorder=3)
            # Arrow at midpoint
            mid = len(arr) // 2
            if mid + 1 < len(arr):
                dx = arr[mid + 1, 0] - arr[mid, 0]
                dy = arr[mid + 1, 1] - arr[mid, 1]
                ax.annotate('', xy=(arr[mid + 1, 0], arr[mid + 1, 1]),
                            xytext=(arr[mid, 0], arr[mid, 1]),
                            arrowprops=dict(arrowstyle='->', color=COLORS[mk],
                                            lw=0.8),
                            zorder=4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('(a) Sampling Trajectories', fontsize=11)
    ax.set_aspect('equal')

    # --- Panel (b): Mode coverage vs NFE ---
    ax = axes[1]
    df_mode = _load_mode_coverage()
    methods_plot = ['monotonic_ddim', 'monotonic_ddpm', 'single_reheat',
                    'damped_oscillation', 'adaptive']
    nfe_vals = sorted(df_mode['nfe'].unique())

    x_pos = np.arange(len(nfe_vals))
    bar_width = 0.15
    offsets = np.arange(len(methods_plot)) - len(methods_plot) / 2 + 0.5

    for i, m in enumerate(methods_plot):
        subset = df_mode[df_mode['method'] == m].sort_values('nfe')
        vals = []
        for nfe in nfe_vals:
            row = subset[subset['nfe'] == nfe]
            vals.append(row['modes_covered'].values[0] if len(row) > 0 else 0)
        ax.bar(x_pos + offsets[i] * bar_width, vals, bar_width,
               color=COLORS.get(m, '#888888'), label=METHOD_LABELS.get(m, m),
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in nfe_vals])
    ax.set_xlabel('NFE')
    ax.set_ylabel('Modes Covered (out of 8)')
    ax.set_title('(b) Mode Coverage', fontsize=11)
    ax.set_ylim(0, 8.8)
    ax.axhline(8, color='#cccccc', linestyle='--', linewidth=0.6, zorder=0)

    # --- Panel (c): FID vs NFE on CIFAR-10 ---
    ax = axes[2]
    df_fid = _load_cifar_fid()
    methods_fid = [m for m in methods_plot if m in df_fid['method'].unique()]

    for m in methods_fid:
        subset = df_fid[df_fid['method'] == m].sort_values('nfe')
        ax.plot(subset['nfe'], subset['fid'],
                marker=METHOD_MARKERS.get(m, 'o'),
                color=COLORS.get(m, '#888888'),
                label=METHOD_LABELS.get(m, m),
                markersize=5, linewidth=1.8)

    ax.set_xscale('log')
    ax.set_xlabel('NFE')
    ax.set_ylabel('FID')
    ax.set_title('(c) CIFAR-10 FID', fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Shared legend at bottom ---
    handles = []
    for m in methods_plot:
        handles.append(Line2D([0], [0], color=COLORS.get(m, '#888888'),
                              marker=METHOD_MARKERS.get(m, 'o'),
                              label=METHOD_LABELS.get(m, m),
                              linewidth=1.8, markersize=5))
    fig.legend(handles=handles, loc='lower center', ncol=len(methods_plot),
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, 'fig1_hero')


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    main()
