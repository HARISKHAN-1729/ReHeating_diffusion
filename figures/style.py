"""
Shared aesthetic configuration for all figures.
Publication-quality style guide for NeurIPS submission.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

# === COLOR PALETTE ===
COLORS = {
    'monotonic_ddim':     '#4A7298',
    'monotonic_ddpm':     '#7BA7CC',
    'single_reheat':      '#E8734A',
    'sawtooth':           '#F2A553',
    'damped_oscillation': '#C94C6D',
    'adaptive':           '#6B4C9A',
    'reheat_stochastic':  '#2D9B7B',
}

COLORS_LIGHT = {k: v + '30' for k, v in COLORS.items()}

# === TYPOGRAPHY & LAYOUT ===
FONT_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
}

LAYOUT_CONFIG = {
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}


def setup_style():
    """Call at the top of every figure script."""
    plt.rcParams.update(FONT_CONFIG)
    plt.rcParams.update(LAYOUT_CONFIG)
    sns.set_palette(list(COLORS.values()))


def save_figure(fig, name, formats=('pdf', 'png')):
    """Save to results/figures/ in multiple formats."""
    os.makedirs('results/figures', exist_ok=True)
    for fmt in formats:
        fig.savefig(f'results/figures/{name}.{fmt}', format=fmt)
    plt.close(fig)
    print(f"Saved: results/figures/{name}.{{{','.join(formats)}}}")


METHOD_LABELS = {
    'monotonic_ddim':     'DDIM (monotonic)',
    'monotonic_ddpm':     'DDPM (\u03b7=1)',
    'single_reheat':      'Single Reheat',
    'sawtooth':           'Periodic Sawtooth',
    'damped_oscillation': 'Damped Oscillation',
    'adaptive':           'Adaptive Reheat',
    'reheat_stochastic':  'Reheat + Stochastic',
}

METHOD_MARKERS = {
    'monotonic_ddim':     'o',
    'monotonic_ddpm':     's',
    'single_reheat':      '^',
    'sawtooth':           'D',
    'damped_oscillation': 'v',
    'adaptive':           'P',
    'reheat_stochastic':  'X',
}

# Ordered list for consistent legend order
METHOD_ORDER = [
    'monotonic_ddim', 'monotonic_ddpm', 'single_reheat',
    'sawtooth', 'damped_oscillation', 'adaptive', 'reheat_stochastic',
]
