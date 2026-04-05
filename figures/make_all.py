"""Master script to regenerate all figures from saved data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    scripts = [
        ('Figure 1 — Hero', 'figures.fig1_hero'),
        ('Figure 2 — Method Schematic', 'figures.fig2_method_schematic'),
        ('Figure 3 — Schedule Families', 'figures.fig3_schedule_families'),
        ('Figure 4 — Cross-Model Bars', 'figures.fig4_cross_model_bars'),
        ('Figure 5 — Ablation Heatmap', 'figures.fig5_ablation_heatmap'),
        ('Figure 6 — Pareto Frontier', 'figures.fig6_pareto'),
        ('Figure 7 — Adaptive Analysis', 'figures.fig7_adaptive_analysis'),
    ]

    os.makedirs('results/figures', exist_ok=True)

    for name, module in scripts:
        print(f"\n{'='*60}")
        print(f"Generating: {name}")
        print(f"{'='*60}")
        try:
            mod = __import__(module, fromlist=['main'])
            mod.main()
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All figures generated. Check results/figures/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
