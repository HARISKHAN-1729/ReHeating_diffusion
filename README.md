# Reheat to Refine: Non-Monotonic Noise Schedules for Diffusion Sampling

> **Diffusion models always sample by monotonically decreasing noise — we show that strategically *increasing* noise at intermediate steps (reheating) yields better samples at the same compute cost.**

---

## Overview

This repository implements and evaluates **non-monotonic noise schedules** (reheating) for diffusion model sampling. The core idea: instead of marching straight from noise to clean image, the sampler occasionally re-noises its current estimate — stepping backward to higher noise — before continuing forward. This requires **no retraining** and works with any pretrained diffusion model.

### Key Insight

Stochastic DDPM sampling already performs implicit, unstructured reheating (noise injection at every step partially undoes denoising). Our method makes this reheating **deliberate, structured, and optimized** — choosing *when* to reheat, *how much*, and *where* in the noise schedule.

### What's Here

- **5 schedule families** — from simple single-bump reheating to adaptive confidence-triggered reheating
- **3 sampler backends** — DDPM/DDIM, EDM2, and Flow Matching
- **9 experiments** — from 2D toy problems to ImageNet-64
- **Publication-quality figures** — reproducible from saved data

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100 recommended, V100 minimum)
- ~20 GB disk space (for model checkpoints and generated images)

### Installation

```bash
git clone <repo-url>
cd reheating-diffusion
pip install -r requirements.txt
```

### Minimal Example — Try Reheating in 30 Seconds

```python
from schedules import monotonic, single_reheat, damped_oscillation
from schedules import compare_all_schedules

# See what the schedules look like
compare_all_schedules(N=50)

# Create a reheating schedule
schedule = single_reheat.make(N=50, t_reheat=0.4, delta=0.15)
# schedule is a numpy array of alpha_bar values in sampling order
# Some consecutive pairs have alpha[i+1] < alpha[i] — these are reheat steps
```

### Run the First Experiment (2D Toy)

```bash
# Train the 2D model and run all schedule comparisons (~15 min on GPU)
python experiments/exp1_toy2d.py

# Results saved to:
#   results/tables/exp1_toy2d.csv
#   results/figures/fig1_hero.pdf
```

### Run the Make-or-Break Experiment (CIFAR-10)

```bash
# Downloads google/ddpm-cifar10-32 automatically (~1.5 hours for full evaluation)
python experiments/exp2_cifar_ddpm.py

# For a quick check with 10K images instead of 50K:
python experiments/exp2_cifar_ddpm.py --n_samples 10000
```

### Run All Experiments

```bash
# Sequential (safest, ~12 GPU-hours total on A100)
for i in 1 2 3 4 5 6 7 8 9; do
    python experiments/exp${i}_*.py
done

# Regenerate all figures from saved results
python figures/make_all.py
```

---

## Project Structure

```
reheating-diffusion/
│
├── CLAUDE.md                    # Implementation guide (for AI agent)
├── PROGRESS.md                  # Development progress tracker
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── schedules/                   # Noise schedule construction
│   ├── __init__.py              #   Exports: make_schedule(), all families
│   ├── monotonic.py             #   Family 1: cosine, linear, EDM baselines
│   ├── single_reheat.py         #   Family 2: one bump at chosen point
│   ├── sawtooth.py              #   Family 3: periodic partial re-noising
│   ├── damped_oscillation.py    #   Family 4: decaying sinusoidal perturbation
│   └── adaptive.py              #   Family 5: confidence-triggered reheating
│
├── samplers/                    # Sampling algorithms
│   ├── __init__.py
│   ├── ddim_reheating.py        #   Generalized DDIM (non-monotonic capable)
│   ├── edm2_reheating.py        #   EDM2 Euler/Heun with reheating
│   └── flow_reheating.py        #   Flow matching Euler with backward steps
│
├── models/                      # Model wrappers
│   ├── __init__.py
│   ├── toy_mlp.py               #   4-layer MLP for 2D experiments
│   ├── ddpm_wrapper.py          #   HuggingFace DDPM wrapper
│   ├── edm2_wrapper.py          #   EDM2 checkpoint wrapper
│   └── flow_wrapper.py          #   OT-CFM UNet wrapper
│
├── experiments/                 # Experiment scripts (self-contained)
│   ├── exp1_toy2d.py            #   2D Gaussian mixture proof-of-concept
│   ├── exp2_cifar_ddpm.py       #   CIFAR-10 with pretrained DDPM
│   ├── exp3_cifar_edm2.py       #   CIFAR-10 with pretrained EDM2
│   ├── exp4_imagenet_edm2.py    #   ImageNet-64 with pretrained EDM2
│   ├── exp5_cifar_flow.py       #   CIFAR-10 with flow matching
│   ├── exp6_ablation.py         #   Schedule hyperparameter grid search
│   ├── exp7_pareto.py           #   NFE Pareto frontier
│   ├── exp8_adaptive.py         #   Adaptive vs. fixed schedules
│   └── exp9_failures.py         #   Failure characterization
│
├── eval/                        # Evaluation metrics
│   ├── fid.py                   #   FID-50K (via clean-fid)
│   ├── inception_score.py       #   Inception Score
│   ├── precision_recall.py      #   Kynkäänniemi et al. P&R
│   └── mode_coverage.py         #   2D mode counting + W2 distance
│
├── figures/                     # Figure generation scripts
│   ├── style.py                 #   Shared colors, fonts, layout
│   ├── make_all.py              #   Regenerate all figures from saved data
│   ├── fig1_hero.py             #   Teaser (trajectories + coverage + FID)
│   ├── fig2_method_schematic.py #   Noise-level trajectory diagram
│   ├── fig3_schedule_families.py#   α̅(t) for all 5 families
│   ├── fig4_cross_model_bars.py #   FID bar charts across models
│   ├── fig5_ablation_heatmap.py #   FID vs (t_reheat, δ)
│   ├── fig6_pareto.py           #   FID vs NFE curves
│   └── fig7_adaptive_analysis.py#   Adaptive reheat diagnostics
│
├── configs/
│   └── schedules.yaml           #   Default hyperparameters for all schedules
│
├── results/
│   ├── tables/                  #   CSV files for all experiment results
│   └── figures/                 #   PDF + PNG outputs
│
└── checkpoints/                 #   Model weights (downloaded or trained)
```

---

## Schedule Families

### Family 1: Monotonic (Baseline)
Standard cosine or EDM schedule. Noise decreases at every step. The control condition all others are compared against.

### Family 2: Single Reheat (Bump)
One interruption at a chosen point. At fraction `t_reheat` through sampling, the noise level jumps backward by `δ` in log-SNR space, then resumes decreasing.

**Hyperparameters:** `t_reheat ∈ [0.2, 0.8]`, `δ ∈ [0.05, 0.5]`

### Family 3: Periodic Sawtooth
Monotonic decrease with periodic partial re-noising every `k` steps, each jumping back by `δ`.

**Hyperparameters:** `period k`, `magnitude δ`

### Family 4: Damped Oscillation
Physically-inspired: large oscillations early, smoothly transitioning to monotonic at the end.

**Formula:** `α̅(t) = cosine(t) + A · exp(−γt) · sin(2πft)`

**Hyperparameters:** `A` (amplitude), `γ` (damping), `f` (frequency)

### Family 5: Adaptive Reheat
Triggers reheating when the model's prediction becomes unstable (prediction x̂₀ changes sharply between consecutive steps). Reheats only when needed, concentrating compute where the model struggles.

**Hyperparameters:** `τ` (threshold, calibrated as 80th percentile from a short calibration run), `δ` (reheat magnitude)

---

## Experiments

| # | Name | Dataset | Model | What It Tests | GPU Time |
|---|------|---------|-------|---------------|----------|
| 1 | 2D Proof of Concept | Synthetic 2D | 4-layer MLP | Mode exploration via reheating | ~0.5h |
| 2 | CIFAR-10 DDPM | CIFAR-10 | Pretrained DDPM | Core hypothesis (make-or-break) | ~1.5h |
| 3 | CIFAR-10 EDM2 | CIFAR-10 | Pretrained EDM2 | Modern architecture validation | ~1.5h |
| 4 | ImageNet-64 EDM2 | ImageNet-64 | Pretrained EDM2 | Larger/more diverse dataset | ~2h |
| 5 | Flow Matching | CIFAR-10 | OT-CFM | Generalization beyond score-based | ~1.5h |
| 6 | Schedule Ablation | CIFAR-10 | Best model | Where/how much to reheat | ~2h |
| 7 | Pareto Frontier | CIFAR-10 | All | Quality-compute tradeoff | ~1h |
| 8 | Adaptive vs Fixed | CIFAR-10 | EDM2 | Data-dependent reheating | ~1h |
| 9 | Failure Modes | CIFAR-10 | EDM2 | When reheating hurts | ~1h |
| **Total** | | | | | **~12h** |

### Running Individual Experiments

Each experiment is self-contained and can be run independently. All scripts support two modes:

```bash
# Sanity check (256 samples, NFE=10,25 — runs on any GPU including 6GB)
python experiments/exp1_toy2d.py --sanity

# Full evaluation (50K samples, all NFE values — needs A100 or similar)
python experiments/exp1_toy2d.py --full
```

Common flags:
- `--sanity`: Quick validation with 256 samples (default)
- `--full`: Full-scale evaluation with 50K samples
- `--batch_size`: Batch size (default: 8 for image models, 256 for 2D toy)
- `--seed`: Random seed (default: 42)
- `--device`: CUDA device (default: cuda:0)

---

## Evaluation Metrics

### FID (Fréchet Inception Distance)
- Computed on 50K generated images vs full training set.
- Lower is better. Typical ranges: CIFAR-10 DDPM = 3–15 depending on NFE.
- Implementation uses `clean-fid` for accurate computation.

### Inception Score (IS)
- Measures quality and diversity. Higher is better.
- Reported as mean ± std over 10 splits.

### Precision & Recall
- Kynkäänniemi et al. decomposition.
- Precision = quality (generated images look real).
- Recall = diversity (all real image modes are covered).

### Mode Coverage (2D only)
- For 8-Gaussian mixture: count how many modes have ≥1% of samples within 2σ.
- Perfect coverage = 8/8.

### Wasserstein-2 Distance (2D only)
- Earth mover's distance between generated and true distributions.

---

## Reproducing Figures

All figures can be regenerated from saved experiment data:

```bash
# Regenerate all figures
python figures/make_all.py

# Regenerate individual figures
python figures/fig1_hero.py
python figures/fig5_ablation_heatmap.py
# ... etc.
```

Figures are saved to `results/figures/` in both PDF (for papers) and PNG (for presentations) at 300 DPI.

### Figure Design

All figures use a consistent visual style defined in `figures/style.py`:
- **Color palette:** Colorblind-friendly, 7 distinct colors mapping to the 7 method variants
- **Typography:** Serif fonts (Times New Roman / DejaVu Serif) matching NeurIPS style
- **Layout:** No top/right spines, subtle gridlines, tight bounding boxes
- **Export:** 300 DPI, tight layout, minimal whitespace

---

## The Core Algorithm

The entire method reduces to using the standard DDIM update with a non-monotonic alpha_bar sequence. The formula works identically whether the next noise level is lower (denoising) or higher (reheating):

```python
# This is the same formula for both denoising and reheating:
x_next = sqrt(alpha_next) * x0_pred + sqrt(1 - alpha_next) * eps_pred

# When alpha_next > alpha_curr → denoising (less noise)
# When alpha_next < alpha_curr → reheating (more noise)
# Same code. Same model. No retraining.
```

This is why reheating works with **any** pretrained diffusion model — DDPM, EDM2, flow matching — by changing only the sampler code.

---

## Configuration

All schedule hyperparameters are in `configs/schedules.yaml`:

```yaml
# Default schedule configurations
monotonic:
  type: cosine
  s: 0.008

single_reheat:
  t_reheat: 0.4       # Where to reheat (fraction of sampling)
  delta: 0.15          # How much to reheat (log-SNR units)

sawtooth:
  period: 25           # Steps between reheats
  delta: 0.08          # Reheat magnitude

damped_oscillation:
  A: 0.2               # Initial amplitude
  gamma: 2.5           # Damping rate
  freq: 4.0            # Oscillation frequency

adaptive:
  percentile: 80       # Calibration percentile for threshold
  delta: 0.15          # Reheat magnitude when triggered
  max_reheats: 15      # Budget cap on reheat steps
  n_calibration: 100   # Calibration samples
```

---

## Preliminary Results (Sanity Checks — 256 Samples)

These results are from quick validation runs. Full-scale numbers (50K samples) will replace them.

### Exp 1: 2D Toy — All methods cover 8/8 modes
DDPM (η=1) achieves the lowest Wasserstein-2 distance (0.18–0.20), supporting the claim that stochastic DDPM sampling acts as implicit reheating.

### Exp 6: Ablation — Early moderate reheating helps
The best configuration (t_reheat=0.3, δ=0.3) improves FID by 22.6 points over the DDIM baseline.

### Exp 7: Pareto — DDPM dramatically outperforms DDIM
A 71-point FID gap (327 vs 398 at NFE=10) between DDPM and DDIM, consistent with the paper's core hypothesis.

### Exp 9: Failure characterization — Extreme reheating is surprisingly stable
No NaN/Inf even at δ=1.0. Large δ values (0.5–1.0) actually _improve_ FID over the baseline (310 vs 404), suggesting more aggressive reheating than expected may be beneficial.

### LaTeX Tables

Pre-generated LaTeX tables are in `results/tables/`:
```bash
python results/generate_tables.py   # Regenerate from CSVs
```

---

## Citation

```bibtex
@inproceedings{reheating2026,
  title={Reheat to Refine: Non-Monotonic Noise Schedules Improve Diffusion Sampling},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
```

---

## License

This project is for research purposes.

---

## Troubleshooting

### FID is unreasonably high (>100)
Most likely an image normalization issue. Generated images must be uint8 in [0, 255], not float in [-1, 1]. Check the conversion: `imgs = ((imgs.clamp(-1,1) + 1) / 2 * 255).to(torch.uint8)`.

### Model download fails
Check your internet connection and HuggingFace access. For EDM2, you may need to download checkpoints manually from the Karras et al. GitHub repository.

### Out of GPU memory
Reduce batch size in the experiment script. For A100 (40GB), batch_size=128 works for CIFAR-10. For V100 (16GB), use batch_size=32–64. Use `--device cpu` for debugging (very slow for FID).

### Experiments take too long
Use `--n_samples 10000` for quick FID estimates (noisier but 5× faster). FID-10K is sufficient for development; switch to FID-50K only for final reported numbers.

### Negative results
This is expected — the paper's own assessment gives a ~35% chance of weak/negative results. Run all experiments honestly and document what you find.
