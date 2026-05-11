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



### Run the Make-or-Break Experiment (CIFAR-10)

```bash
# Downloads google/ddpm-cifar10-32 automatically (~1.5 hours for full evaluation)
python experiments/exp1_cifar_ddpm.py

# For a quick check with 10K images instead of 50K:
python experiments/exp2_cifar_ddpm.py --n_samples 10000
```

---

## Project Structure

```
reheating-diffusion/
│
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
│   ├── exp1_cifar_ddpm.py       #   CIFAR-10 with pretrained DDPM
│   ├── exp2_cifar_edm2.py       #   CIFAR-10 with pretrained EDM2│
│   ├── exp3_cifar_flow.py       #   CIFAR-10 with flow matching
│   ├── exp4_ablation.py         #   Schedule hyperparameter grid search
│   ├── exp5_pareto.py           #   NFE Pareto frontier
│        
│
├── eval/                        # Evaluation metrics
│   ├── fid.py                   #   FID-50K (via clean-fid)
│   ├── inception_score.py       #   Inception Score
│   ├── precision_recall.py      #   Kynkäänniemi et al. P&R
│   └── mode_coverage.py         #   2D mode counting + W2 distance
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
| 1 | CIFAR-10 DDPM | CIFAR-10 | Pretrained DDPM | Core hypothesis (make-or-break) 
| 2 | CIFAR-10 EDM2 | CIFAR-10 | Pretrained EDM2 | Modern architecture validation 
| 3 | Flow Matching | CIFAR-10 | OT-CFM | Generalization beyond score-based 
| 4 | Schedule Ablation | CIFAR-10 | Best model | Where/how much to reheat
| 5 | Pareto Frontier | CIFAR-10 | All | Quality-compute tradeoff 


### Running Individual Experiments

Each experiment is self-contained and can be run independently. All scripts support two modes:

```bash
# Sanity check (256 samples, NFE=10,25 — runs on any GPU including 6GB)
python experiments/exp1_toy2d.py --sanity

# Full evaluation (50K samples, all NFE values — needs A100 or similar)
python experiments/exp1_toy2d.py --full
```



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

## Citation

```bibtex
@inproceedings{reheating2026,
  title={Reheat to Refine: Non-Monotonic Noise Schedules Improve Diffusion Sampling},
  author={Muhammad Haris Khan},  
  year={2026}
}
```

---

## License

This project is for research purposes.

---

