# INSTRUCTIONS.md — Complete Run Guide

> Clone on Colab, run everything, download results.

---

## 1. What to Push to GitHub

### Push ONLY code — no results, no checkpoints

```bash
cd /home/khan/Reheating_diffusion

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
venv/
.venv/

# Generated at runtime — do NOT push
checkpoints/
results/

# IDE / OS
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db
EOF

# Initialize and commit
git init
git add .gitignore
git add requirements.txt README.md CLAUDE.md PROGRESS.md INSTRUCTIONS.md
git add configs/
git add schedules/
git add samplers/
git add models/
git add experiments/
git add eval/
git add figures/
git add tests/

# Verify — should NOT show checkpoints/ or results/
git status

git commit -m "Initial commit: complete reheating diffusion codebase

- 5 schedule families (monotonic, single_reheat, sawtooth, damped_oscillation, adaptive)
- 3 sampler backends (DDIM, EDM2, Flow Matching)
- 4 model wrappers (toy MLP, DDPM, EDM2, Flow)
- 9 experiment scripts with --sanity/--full modes
- 7 publication-quality figure scripts
- 68 unit tests"

# Push to GitHub
gh repo create reheating-diffusion --public --source=. --remote=origin --push

# OR manually:
# git remote add origin https://github.com/YOUR_USERNAME/reheating-diffusion.git
# git branch -M main
# git push -u origin main
```

### What gets pushed vs excluded

| PUSH (code only) | EXCLUDE (generated at runtime) |
|---|---|
| `schedules/` `samplers/` `models/` | `checkpoints/` (~500 MB model weights) |
| `experiments/` `eval/` `figures/` | `results/` (CSVs, figures, tables) |
| `configs/` `tests/` | `__pycache__/` `venv/` |
| `requirements.txt` `*.md` | |

---

## 2. Colab Setup

Paste this into the **first cell** of a Colab notebook (use a **GPU runtime: A100 or T4**):

```python
# === CELL 1: Clone and install ===
!git clone https://github.com/YOUR_USERNAME/reheating-diffusion.git
%cd reheating-diffusion
!pip install -q -r requirements.txt

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## 3. Run Tests

```python
# === CELL 2: Verify codebase ===
!python -m pytest tests/ -v
```

Expected: `68 passed`.

---

## 4. Run All Experiments

### Option A: Sanity checks only (T4 / any GPU, ~30 min)

```python
# === CELL 3A: All sanity checks ===
!python experiments/exp1_toy2d.py --sanity
!python experiments/exp2_cifar_ddpm.py --sanity
!python experiments/exp3_cifar_edm2.py --sanity
!python experiments/exp4_imagenet_edm2.py --sanity
!python experiments/exp5_cifar_flow.py --sanity
!python experiments/exp6_ablation.py --sanity
!python experiments/exp7_pareto.py --sanity
!python experiments/exp8_adaptive.py --sanity
!python experiments/exp9_failures.py --sanity
```

### Option B: Full experiments (A100, ~12 hours)

```python
# === CELL 3B: All full experiments ===

# Exp 1 — 2D Toy (~30 min)
!python experiments/exp1_toy2d.py --full --n_samples 10000 --nfe 10,25,50

# Exp 2 — CIFAR-10 DDPM (~2 hours) *** MOST IMPORTANT ***
!python experiments/exp2_cifar_ddpm.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128

# Exp 3 — CIFAR-10 EDM2 (~2 hours, needs real checkpoint)
!python experiments/exp3_cifar_edm2.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128

# Exp 4 — ImageNet-64 (~3 hours, needs 8+ GB VRAM)
!python experiments/exp4_imagenet_edm2.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 64

# Exp 5 — Flow Matching (~3 hours: trains model then samples)
!python experiments/exp5_cifar_flow.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128

# Exp 6 — Ablation Grid (~3 hours, 42 configs)
!python experiments/exp6_ablation.py --full --n_samples 10000 --batch_size 128

# Exp 7 — Pareto Frontier (~2 hours)
!python experiments/exp7_pareto.py --full --n_samples 50000 --nfe 5,10,15,20,25,50,100,200 --batch_size 128

# Exp 8 — Adaptive Reheat (~2 hours)
!python experiments/exp8_adaptive.py --full --n_samples 50000 --nfe 25,50 --batch_size 128

# Exp 9 — Failure Characterization (~30 min)
!python experiments/exp9_failures.py --full --n_samples 1000 --batch_size 128
```

### Option C: Run just the critical ones first (A100, ~4 hours)

If you have limited Colab time, run these in priority order:

```python
# Priority 1: The make-or-break experiment
!python experiments/exp2_cifar_ddpm.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128

# Priority 2: The hero figure data
!python experiments/exp1_toy2d.py --full --n_samples 10000 --nfe 10,25,50

# Priority 3: Ablation (what hyperparams work best)
!python experiments/exp6_ablation.py --full --n_samples 10000 --batch_size 128

# Priority 4: Pareto (quality vs compute tradeoff)
!python experiments/exp7_pareto.py --full --n_samples 50000 --nfe 5,10,15,20,25,50,100,200 --batch_size 128
```

---

## 5. Generate Figures and Tables

```python
# === CELL 4: Figures (7 figures, PDF + PNG) ===
import os
os.environ['MPLBACKEND'] = 'Agg'
!python figures/make_all.py

# === CELL 5: LaTeX tables (6 tables) ===
!python results/generate_tables.py
```

---

## 6. Download Results from Colab

```python
# === CELL 6: Zip and download everything ===
!zip -r results.zip results/
from google.colab import files
files.download('results.zip')
```

Or download individual files:

```python
from google.colab import files

# Download specific CSVs
files.download('results/tables/exp2_cifar_ddpm.csv')

# Download specific figures
files.download('results/figures/fig1_hero.pdf')
files.download('results/figures/fig5_ablation_heatmap.pdf')

# Download all LaTeX tables
!zip -r tables.zip results/tables/*.tex
files.download('tables.zip')
```

---

## 7. What Gets Generated (Runtime Outputs)

After running all experiments + figures + tables:

```
results/
├── tables/
│   ├── exp1_toy2d.csv              # 2D toy: modes + W2 per method/NFE
│   ├── exp1_trajectories.npz       # Trajectory data for fig1
│   ├── exp2_cifar_ddpm.csv         # CIFAR DDPM: FID per config/NFE
│   ├── exp3_cifar_edm2.csv         # CIFAR EDM2: FID per config/NFE
│   ├── exp5_cifar_flow.csv         # Flow: FID per config/NFE
│   ├── exp6_ablation.csv           # Ablation: FID vs (t_reheat, delta)
│   ├── exp7_pareto.csv             # Pareto: FID + time vs NFE
│   ├── exp8_adaptive.csv           # Adaptive: FID vs threshold
│   ├── exp8_reheat_positions.npz   # Where reheats trigger
│   ├── exp9_failures.csv           # Failure: FID + stability at extreme delta
│   ├── table1_main_results.tex     # LaTeX: cross-model results
│   ├── table2_ablation.tex         # LaTeX: ablation grid
│   ├── table3_adaptive.tex         # LaTeX: adaptive sensitivity
│   ├── table4_failures.tex         # LaTeX: failure modes
│   ├── table_toy2d.tex             # LaTeX: 2D toy results
│   └── table_pareto.tex            # LaTeX: Pareto data
├── figures/
│   ├── fig1_hero.{pdf,png}         # 3-panel teaser
│   ├── fig2_method_schematic.{pdf,png}
│   ├── fig3_schedule_families.{pdf,png}
│   ├── fig4_cross_model_bars.{pdf,png}
│   ├── fig5_ablation_heatmap.{pdf,png}
│   ├── fig6_pareto_linear.{pdf,png}
│   ├── fig6_pareto_log.{pdf,png}
│   └── fig7_adaptive_analysis.{pdf,png}
checkpoints/
├── toy_mlp.pt                      # Trained 2D MLP (~845 KB)
├── flow_cifar10.pt                 # Trained flow model (~29 MB)
├── models--google--ddpm-cifar10-32/# Cached DDPM (~137 MB)
└── data/cifar-10-python.tar.gz     # CIFAR-10 dataset (~163 MB)
```

---

## Quick Reference — Copy-Paste for Colab

```python
# CELL 1: Setup
!git clone https://github.com/YOUR_USERNAME/reheating-diffusion.git
%cd reheating-diffusion
!pip install -q -r requirements.txt

# CELL 2: Tests
!python -m pytest tests/ -v

# CELL 3: Experiments (pick one option)
# --- SANITY (any GPU, 30 min) ---
!python experiments/exp1_toy2d.py --sanity && python experiments/exp2_cifar_ddpm.py --sanity && python experiments/exp3_cifar_edm2.py --sanity && python experiments/exp4_imagenet_edm2.py --sanity && python experiments/exp5_cifar_flow.py --sanity && python experiments/exp6_ablation.py --sanity && python experiments/exp7_pareto.py --sanity && python experiments/exp8_adaptive.py --sanity && python experiments/exp9_failures.py --sanity

# --- FULL (A100, ~12 hours) ---
!python experiments/exp1_toy2d.py --full --n_samples 10000 --nfe 10,25,50 && python experiments/exp2_cifar_ddpm.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128 && python experiments/exp3_cifar_edm2.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128 && python experiments/exp4_imagenet_edm2.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 64 && python experiments/exp5_cifar_flow.py --full --n_samples 50000 --nfe 10,25,50,100 --batch_size 128 && python experiments/exp6_ablation.py --full --n_samples 10000 --batch_size 128 && python experiments/exp7_pareto.py --full --n_samples 50000 --nfe 5,10,15,20,25,50,100,200 --batch_size 128 && python experiments/exp8_adaptive.py --full --n_samples 50000 --nfe 25,50 --batch_size 128 && python experiments/exp9_failures.py --full --n_samples 1000 --batch_size 128

# CELL 4: Figures + Tables
import os; os.environ['MPLBACKEND'] = 'Agg'
!python figures/make_all.py && python results/generate_tables.py

# CELL 5: Download
!zip -r results.zip results/
from google.colab import files; files.download('results.zip')
```
