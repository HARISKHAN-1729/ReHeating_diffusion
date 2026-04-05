# CLAUDE.md — Master Prompt for Reheating Diffusion Implementation

> **Read this entire file before writing a single line of code.**
> After every completed task, update `PROGRESS.md` with status, notes, and any blockers.
> If you are resuming after a session limit, read `PROGRESS.md` first to find where you left off.

---

## 0. PROJECT IDENTITY

**Project:** "Reheat to Refine: Non-Monotonic Noise Schedules for Diffusion Sampling"

**Goal:** Implement, run, and produce publication-quality results for all 9 experiments described in `reheating_diffusion_v2.md`, using the sampler code in `reheating_sampler_v2.py` as the foundation.

**Core claim:** Diffusion models sampled with strategically non-monotonic noise schedules (reheating) produce better samples than monotonic baselines at matched NFE — and DDPM's stochastic sampling is already implicit, unstructured reheating.

---

## 0.1 HARDWARE CONSTRAINTS (RTX 3060 — 6GB VRAM)

**This system is being developed on a laptop with an NVIDIA RTX 3060 (6GB VRAM).** All code must respect this. Full-scale experiments (50K samples) will be run later on a larger GPU.

### Hard Rules:
- **Max batch_size=8** for all image models (DDPM, EDM2, Flow). For 2D toy: batch_size=256 is fine.
- **Always use float16 inference:** `torch.cuda.amp.autocast(dtype=torch.float16)` around model forward passes.
- **Save images to disk incrementally** — never hold more than one batch in GPU memory.
- **Call `torch.cuda.empty_cache()`** between experiments.
- **Do NOT train Flow Matching from scratch** on this GPU (build the script, sanity-check with 500 training steps only).
- **Do NOT run ImageNet-64 experiments** on this GPU. Build the script but add a VRAM check that exits gracefully if < 8GB.

### Two Modes in Every Experiment Script:
```python
parser.add_argument('--sanity', action='store_true', default=True)   # 256 samples, NFE=10,25
parser.add_argument('--full', action='store_true', default=False)    # 50K samples, all NFE
parser.add_argument('--batch_size', type=int, default=8)
```
When `--sanity` (default): n_samples=256, nfe=10,25 only, save 16-image grid for visual check.
When `--full`: n_samples=50000, nfe=10,25,50,100, compute real FID-50K.

### In PROGRESS.md:
Mark sanity checks as **"SANITY PASSED"** (not "DONE"). Mark code completion as **"DONE"**.

---

## 1. FIRST ACTIONS (EVERY SESSION)

```
1. Read PROGRESS.md → find the current phase and next TODO.
2. If PROGRESS.md says "PHASE X — IN PROGRESS", resume from the last incomplete task.
3. If PROGRESS.md says "PHASE X — DONE", begin the next phase.
4. If PROGRESS.md does not exist, start from PHASE 1.
```

---

## 2. ENVIRONMENT SETUP (PHASE 1)

### 2.1 Directory Structure

Create exactly this layout:

```
reheating-diffusion/
├── CLAUDE.md                          # This file (copy from root)
├── PROGRESS.md                        # Progress tracker (copy from root)
├── README.md                          # Documentation (copy from root)
├── requirements.txt                   # Pinned dependencies
├── schedules/
│   ├── __init__.py
│   ├── monotonic.py                   # Family 1: cosine/linear/EDM baselines
│   ├── single_reheat.py               # Family 2: single bump schedule
│   ├── sawtooth.py                    # Family 3: periodic partial re-noising
│   ├── damped_oscillation.py          # Family 4: A·exp(-γt)·sin(2πft)
│   └── adaptive.py                    # Family 5: confidence-triggered reheating
├── samplers/
│   ├── __init__.py
│   ├── ddim_reheating.py              # Generalized DDIM (non-monotonic capable)
│   ├── edm2_reheating.py              # EDM2 Heun/Euler with reheating
│   └── flow_reheating.py              # Flow matching Euler with backward steps
├── models/
│   ├── __init__.py
│   ├── toy_mlp.py                     # 4-layer MLP for 2D experiments
│   ├── ddpm_wrapper.py                # HuggingFace DDPM wrapper
│   ├── edm2_wrapper.py                # EDM2 checkpoint wrapper
│   └── flow_wrapper.py                # OT-CFM UNet wrapper
├── experiments/
│   ├── __init__.py
│   ├── exp1_toy2d.py                  # 2D Gaussian mixture proof-of-concept
│   ├── exp2_cifar_ddpm.py             # CIFAR-10 with pretrained DDPM
│   ├── exp3_cifar_edm2.py             # CIFAR-10 with pretrained EDM2
│   ├── exp4_imagenet_edm2.py          # ImageNet-64 with pretrained EDM2
│   ├── exp5_cifar_flow.py             # CIFAR-10 with flow matching
│   ├── exp6_ablation.py               # Schedule hyperparameter grid
│   ├── exp7_pareto.py                 # NFE Pareto frontier
│   ├── exp8_adaptive.py               # Adaptive vs. fixed comparison
│   └── exp9_failures.py               # Failure characterization
├── eval/
│   ├── __init__.py
│   ├── fid.py                         # FID-50K (pytorch-fid or clean-fid)
│   ├── inception_score.py             # IS computation
│   ├── precision_recall.py            # Kynkäänniemi et al. P&R
│   └── mode_coverage.py               # 2D mode counting + W2 distance
├── figures/
│   ├── make_all.py                    # Master script to regenerate all figures
│   ├── fig1_hero.py                   # Teaser: trajectories + mode coverage + FID
│   ├── fig2_method_schematic.py       # Noise-level trajectory diagram
│   ├── fig3_schedule_families.py      # α̅(t) for all 5 families
│   ├── fig4_cross_model_bars.py       # FID bar charts across models
│   ├── fig5_ablation_heatmap.py       # FID vs (t_reheat, δ)
│   ├── fig6_pareto.py                 # FID vs NFE curves
│   ├── fig7_adaptive_analysis.py      # Adaptive reheat diagnostics
│   └── style.py                       # Shared aesthetic config
├── configs/
│   └── schedules.yaml                 # All schedule hyperparameters
├── results/
│   ├── tables/                        # CSV tables for every experiment
│   └── figures/                       # PDF/PNG outputs of all figures
└── checkpoints/                       # Downloaded/trained model weights
```

### 2.2 Dependencies

```
requirements.txt contents:

torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
PyYAML>=6.0
tqdm>=4.65.0
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
clean-fid>=0.1.35
pytorch-fid>=0.3.0
pot>=0.9.0
scikit-learn>=1.3.0
einops>=0.7.0
pillow>=10.0.0
```

Install with: `pip install -r requirements.txt`

### 2.3 Seeding & Reproducibility

Every experiment MUST:
- Set `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.cuda.manual_seed_all(42)`
- Set `torch.backends.cudnn.deterministic = True`
- Log the exact command, seed, and git hash (or file checksums) in `PROGRESS.md`
- Save raw metric values to CSV in `results/tables/`

---

## 3. IMPLEMENTATION PHASES

### PHASE 1: Scaffolding + Schedule Module (Priority: CRITICAL)

**Tasks:**
1. Create directory structure above.
2. Port schedule construction from `reheating_sampler_v2.py` into individual files under `schedules/`.
   - `monotonic.py` — cosine baseline, linear, EDM sigma schedules.
   - `single_reheat.py` — Family 2. Params: `t_reheat ∈ [0.2, 0.8]`, `delta ∈ [0.05, 0.5]`.
   - `sawtooth.py` — Family 3. Params: `period`, `sawtooth_delta`.
   - `damped_oscillation.py` — Family 4. Params: `A`, `gamma`, `freq`. Formula: `α̅(t) = cosine(t) + A·exp(-γt)·sin(2πft)`.
   - `adaptive.py` — Family 5. Calibration function + threshold logic.
   - Each file must have a `make(N, **kwargs) -> np.ndarray` function returning alpha_bars in sampling order.
   - Include `schedule_report()` and `compare_all_schedules()` from the v2 sampler.
3. Port `make_edm2_schedule()` and `make_flow_schedule()` converters.
4. Create `configs/schedules.yaml` with all default hyperparameters.
5. Write unit tests: verify schedule shapes, boundary values (first = high noise, last ≈ 0), non-monotonicity counts.

**Validation gate:** `python -m pytest tests/test_schedules.py` passes. `compare_all_schedules(50)` prints correct table.

---

### PHASE 2: Samplers (Priority: CRITICAL)

**Tasks:**
1. Port `ddim_sample()` into `samplers/ddim_reheating.py`.
   - Must handle non-monotonic alpha_bar sequences.
   - Must support `eta=0` (DDIM) and `eta=1` (DDPM-like stochastic).
   - Must handle timestep conversion: `t_int = ((1 - alpha_bar) * num_train_timesteps).long()`
   - `return_trajectory` option for visualization.
2. Port `edm2_sample()` into `samplers/edm2_reheating.py`.
   - Uses sigma-parameterization: `d = (x - D(x,σ)) / σ`, `x_next = x + (σ_next - σ_cur) * d`.
   - Reheat = σ_next > σ_cur. Same formula, just adds noise.
   - Support both Euler and Heun (2nd-order) stepping.
3. Port `flow_sample()` into `samplers/flow_reheating.py`.
   - Euler step: `x = x + dt * v_θ(x, t)`. Backward step = negative dt.
   - Time schedule from `make_flow_schedule()`.
4. Port `adaptive_sample()` and `calibrate_threshold()` into `samplers/` or keep in `schedules/adaptive.py`.
   - AdaptiveStats dataclass for diagnostics.
   - NFE budget enforcement: `max_nfe = len(base_schedule) + max_reheats`.
5. Write integration tests with a dummy model (random noise predictor) to verify:
   - Output shape correctness
   - NFE counting is accurate
   - Trajectory recording works
   - No NaN/Inf in outputs

**Validation gate:** All samplers produce valid tensor outputs with a dummy model.

---

### PHASE 3: Model Wrappers (Priority: CRITICAL)

**Tasks:**
1. **`models/toy_mlp.py`** — 4-layer MLP for 2D:
   - Input: `(x ∈ R², t ∈ R)` → Output: `ε ∈ R²` (noise prediction)
   - Architecture: `[x; sinusoidal_embed(t)] → 256 → 256 → 256 → 256 → 2`
   - Sinusoidal embedding: 64-dim, frequencies from `[1, 10000]`.
   - ~50K parameters.
   - Training function: standard DDPM loss on 8-Gaussian ring mixture.
   - 50K training steps, lr=1e-3, Adam, batch_size=512.
   - Training should take ~10 minutes on GPU.

2. **`models/ddpm_wrapper.py`** — HuggingFace DDPM:
   - Load `google/ddpm-cifar10-32` via `diffusers.DDPMPipeline`.
   - Expose a clean interface: `model(x_t, timestep_int) → eps_pred`.
   - Handle the UNet's expected input format.
   - Cache the model on first load.

3. **`models/edm2_wrapper.py`** — EDM2:
   - Download pretrained EDM2-S for CIFAR-10 from Karras et al. official repo.
   - If official checkpoint is unavailable, fall back to the `edm2-cifar10-32x32-cond-vp.pkl` or equivalent.
   - Expose: `denoiser(x, sigma) → denoised_x` (not noise, but clean image prediction).
   - Handle EDM2's preconditioning: `D(x;σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ))`.
   - **IMPORTANT:** If you cannot find/download EDM2 checkpoints, document this in PROGRESS.md and implement a training script using the EDM2 framework instead. Do NOT silently skip.

4. **`models/flow_wrapper.py`** — OT-CFM:
   - Option A: find a pretrained OT-CFM checkpoint for CIFAR-10.
   - Option B (more likely): train a small U-Net with OT-CFM objective.
     - Architecture: small UNet (~10M params), sinusoidal time embedding.
     - Training: OT-CFM loss with linear interpolation `x_t = (1-t)·ε + t·x_0`, velocity target `v = x_0 - ε`.
     - ~2 hours on A100, ~4 hours on V100.
   - Expose: `velocity_model(x, t) → v`.

5. **`models/ddpm_wrapper.py` for ImageNet-64** — separate wrapper or config flag for `edm2_wrapper.py` to load ImageNet-64 weights.

**Validation gate:** Each model loads, accepts correct input shapes, returns correct output shapes. Dummy forward pass completes without error.

---

### PHASE 4: Evaluation Metrics (Priority: CRITICAL)

**Tasks:**
1. **`eval/fid.py`**:
   - Use `clean-fid` or `pytorch-fid` for FID-50K.
   - Function: `compute_fid(generated_images_dir, dataset_name, batch_size=64) → float`.
   - Pre-compute and cache reference statistics for CIFAR-10 and ImageNet-64.
   - For faster iteration during development, support FID-10K with a flag.

2. **`eval/inception_score.py`**:
   - Standard IS computation using Inception v3.
   - Function: `compute_is(images, splits=10) → (mean, std)`.

3. **`eval/precision_recall.py`**:
   - Kynkäänniemi et al. improved precision and recall.
   - Function: `compute_precision_recall(real_features, gen_features, k=3) → (precision, recall)`.

4. **`eval/mode_coverage.py`** (for 2D experiments):
   - Count modes covered: a mode is "covered" if ≥ 1% of samples fall within 2σ of its center.
   - Wasserstein-2 distance to ground truth (use `pot` library).
   - Function: `count_modes(samples, centers, stds, threshold=0.01) → int`.
   - Function: `wasserstein2(samples, reference) → float`.

**Validation gate:** FID returns a reasonable number on a small test set. Mode coverage correctly counts 8/8 on a trivial distribution.

---

### PHASE 5: Experiment 1 — 2D Proof of Concept (Priority: HIGH)

**This is the hero figure. It must be visually stunning.**

**Tasks:**
1. Train the toy MLP on 8-Gaussian ring mixture (50K steps, ~10 min).
2. Sample 10K points with each schedule family at NFE ∈ {10, 25, 50}:
   - Monotonic (DDIM, η=0)
   - Monotonic (DDPM, η=1)
   - Single reheat (t_reheat=0.4, δ=0.15)
   - Damped oscillation (A=0.2, γ=2.5, f=4)
   - Adaptive (τ = 80th percentile from calibration)
3. Measure: modes covered (out of 8), W2 distance to true distribution.
4. Record individual sampling trajectories (5-10 trajectories per method) for visualization.
5. Save all results to `results/tables/exp1_toy2d.csv`.

**Figure 1 (Hero):** Three panels:
- (a) Sampling trajectories overlaid on true density contours. Show 5 trajectories per method. Monotonic trajectories lock in early, reheating trajectories oscillate.
- (b) Mode coverage (out of 8) vs NFE for each method. Bar or line chart.
- (c) W2 distance vs NFE.

**Validation gate:** At least one reheating schedule covers more modes than monotonic at NFE=10. Results saved to CSV.

---

### PHASE 6: Experiment 2 — CIFAR-10 DDPM (Priority: CRITICAL — Make-or-Break)

**If this doesn't show improvement, document the negative result honestly.**

**Tasks:**
1. Load `google/ddpm-cifar10-32`.
2. Generate 50K images per configuration (or 10K for initial quick check):

| Config | Schedule | η | NFE values |
|--------|----------|---|------------|
| DDIM baseline | Monotonic cosine | 0.0 | 10, 25, 50, 100 |
| DDPM baseline | Monotonic cosine | 1.0 | 10, 25, 50, 100 |
| Single reheat | t=0.4, δ=0.15 | 0.0 | 10, 25, 50, 100 |
| Damped oscillation | A=0.2, γ=2.5, f=4 | 0.0 | 10, 25, 50, 100 |
| Adaptive reheat | τ=80th pct | 0.0 | 10, 25, 50, 100 |
| Reheat + stochastic | Best fixed, η=0.5 | 0.5 | 10, 25, 50, 100 |

3. Compute FID-50K and IS for each.
4. Save to `results/tables/exp2_cifar_ddpm.csv`.
5. Save generated images in batch-organized directories for FID computation.

**CRITICAL:** Run DDIM and DDPM baselines FIRST. Then run single reheat. If single reheat does not beat at least one baseline at any NFE, log this in PROGRESS.md and continue with other experiments — the negative result is still scientifically valuable.

**Validation gate:** FID numbers are in a reasonable range (CIFAR-10 DDPM FID is typically 3-15 depending on NFE). Results saved.

---

### PHASE 7: Experiments 3-5 — Cross-Model Validation (Priority: HIGH)

**Tasks:**
1. **Exp 3 (CIFAR-10 EDM2):** Adapt reheating to σ-parameterization. Run same schedule grid as Exp 2 but with EDM2's Euler/Heun sampler. Compute FID.
2. **Exp 4 (ImageNet-64 EDM2):** Run only the best 2-3 schedules from Exp 2-3 on ImageNet-64. FID-50K.
3. **Exp 5 (Flow Matching):** Train or load OT-CFM on CIFAR-10. Run schedules with backward Euler steps. FID.
4. Save all results to `results/tables/exp3_*.csv`, `exp4_*.csv`, `exp5_*.csv`.

**If EDM2 checkpoints are unavailable:** Train EDM2-S from scratch on CIFAR-10 (~12h on A100), or use a substitute modern architecture. Document what you used.

**If flow matching training is needed:** Train for ~2h, validate that the monotonic baseline FID is reasonable before testing reheating.

---

### PHASE 8: Experiments 6-9 — Ablations and Analysis (Priority: HIGH)

**Tasks:**

**Exp 6 (Ablation Grid):**
- Model: best-performing from Exp 2-5.
- Grid: `t_reheat ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}` × `δ ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}`.
- 42 configs × 10K images each.
- Produce heatmap data → `results/tables/exp6_ablation.csv`.

**Exp 7 (Pareto Frontier):**
- For each model: generate 50K images at NFE ∈ {5, 10, 15, 20, 25, 50, 100, 200}.
- Methods: DDIM, DDPM, best reheat (det), best reheat (stoch).
- Measure FID and wall-clock time.
- Save → `results/tables/exp7_pareto.csv`.

**Exp 8 (Adaptive vs Fixed):**
- Model: EDM2 on CIFAR-10.
- Calibration: 100 monotonic samples → compute τ at 80th percentile.
- Generate 50K images with adaptive schedule.
- Log WHERE reheats trigger across 1000 samples.
- Sensitivity: vary τ from 50th to 95th percentile.
- Save → `results/tables/exp8_adaptive.csv`.

**Exp 9 (Failure Characterization):**
- Aggressive reheating: δ = 0.5, δ = 1.0, multiple large reheats.
- 1K images per config. Manual inspection notes.
- Decompose into precision (quality) vs recall (diversity).
- Save → `results/tables/exp9_failures.csv`.

---

### PHASE 9: Figure Generation (Priority: CRITICAL)

**All figures MUST be publication-quality. Follow the style guide below.**

#### FIGURE STYLE GUIDE

```python
# figures/style.py — import this in every figure script

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# === COLOR PALETTE ===
# A cohesive, colorblind-friendly palette with strong visual identity.
COLORS = {
    'monotonic_ddim':     '#4A7298',    # Steel blue — baseline, understated
    'monotonic_ddpm':     '#7BA7CC',    # Light steel — stochastic baseline
    'single_reheat':      '#E8734A',    # Burnt orange — primary method
    'sawtooth':           '#F2A553',    # Amber — secondary method
    'damped_oscillation': '#C94C6D',    # Crimson rose — oscillation
    'adaptive':           '#6B4C9A',    # Deep violet — adaptive (premium)
    'reheat_stochastic':  '#2D9B7B',    # Teal — combined method
}

# Matching lighter shades for fills/backgrounds
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
    import os
    os.makedirs('results/figures', exist_ok=True)
    for fmt in formats:
        fig.savefig(f'results/figures/{name}.{fmt}', format=fmt)
    print(f"Saved: results/figures/{name}.{{pdf,png}}")

METHOD_LABELS = {
    'monotonic_ddim':     'DDIM (monotonic)',
    'monotonic_ddpm':     'DDPM (η=1)',
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
```

#### FIGURE SPECIFICATIONS

**Figure 1 — Hero/Teaser (3-panel, full width):**
- Size: `(14, 4)` inches (NeurIPS full-width).
- Panel (a): 2D density contours (filled, soft gradient) with 5 sampling trajectories per method overlaid. Use thin lines (0.8pt) with arrow markers showing direction. Background: light gray density contours of the true 8-Gaussian mixture.
- Panel (b): Mode coverage vs NFE. Grouped bar chart or line plot with markers.
- Panel (c): FID vs NFE on CIFAR-10 (preview from Exp 2). Line plot with markers, log-scale x-axis.
- Shared legend at bottom, horizontal, outside axes.

**Figure 2 — Method Schematic:**
- Size: `(7, 3.5)` (single column).
- Horizontal axis: sampling step index. Vertical axis: noise level σ (or α̅).
- Show monotonic schedule as smooth decreasing curve.
- Show reheating schedule with red-shaded "reheat" arrows (increasing σ) and blue "denoise" arrows (decreasing σ).
- Annotate the DDIM update equation showing it works both directions.
- Clean, minimal, diagram-style (not a data plot).

**Figure 3 — Schedule Families (5 subplots):**
- Size: `(14, 2.5)`.
- 5 small subplots in a row, one per family.
- Each shows α̅(t) as a line. Red shading where α̅ decreases (reheat regions).
- Consistent y-axis [0,1] and x-axis [0,1] across all panels.
- Panel titles: "Monotonic", "Single Reheat", "Sawtooth", "Damped Oscillation", "Adaptive (example)".

**Figure 4 — Cross-Model Results (grouped bar chart):**
- Size: `(10, 4)`.
- Groups: DDPM/CIFAR-10, EDM2/CIFAR-10, Flow/CIFAR-10, EDM2/ImageNet-64.
- Bars within each group: DDIM, DDPM, Best Reheating.
- FID on y-axis (lower is better — annotate this).
- Error bars if multiple seeds.
- Clean, professional bar chart. No 3D effects.

**Figure 5 — Ablation Heatmap:**
- Size: `(6, 5)`.
- Axes: t_reheat (x), δ (y).
- Color: FID value. Use a diverging colormap centered on the DDIM baseline FID.
- Blues = better than baseline, reds = worse than baseline.
- Annotate the optimal cell.
- Colorbar with label "FID-50K".

**Figure 6 — Pareto Frontier:**
- Size: `(7, 5)`.
- X-axis: NFE (log scale). Y-axis: FID.
- One curve per method with markers.
- Shaded area between best reheating and best monotonic to highlight the gap.
- Both linear and log-scale NFE variants (save as fig6a and fig6b).

**Figure 7 — Adaptive Analysis (3-panel):**
- Size: `(14, 4)`.
- (a): Histogram of reheat trigger positions across 1K samples. X-axis: sampling step. Should show concentration in critical zone.
- (b): FID vs τ (threshold). Line plot with confidence band.
- (c): Side-by-side generated samples: "reheat triggered" vs "no reheat" for the same seed.

---

### PHASE 10: Tables and Final Results (Priority: HIGH)

Generate these tables in both CSV and LaTeX format:

**Table 1 — Main Results:**
| Model | Dataset | NFE | DDIM FID | DDPM FID | Best Reheat FID | Schedule | Improvement |

**Table 2 — Ablation Grid:**
| t_reheat | δ | FID | ΔFIDvsBaseline |

**Table 3 — Adaptive Sensitivity:**
| τ percentile | FID | Avg reheats/sample |

**Table 4 — Failure Modes:**
| δ | FID | Precision | Recall | Artifacts |

Save all to `results/tables/` as both `.csv` and `.tex`.

---

## 4. CRITICAL IMPLEMENTATION DETAILS

### 4.1 The Core Algorithm (Non-Negotiable Correctness)

The DDIM update that works for BOTH denoising AND reheating:

```python
# Given alpha_bar at current step (a_c) and next step (a_n):
# When a_n > a_c: standard denoising (less noise)
# When a_n < a_c: reheating (more noise) — same formula, no special case needed

eps_pred = model(x, timestep)
x0_pred = (x - sqrt(1 - a_c) * eps_pred) / sqrt(a_c)
x0_pred = x0_pred.clamp(-1, 1)

# Deterministic step (works both directions):
x_next = sqrt(a_n) * x0_pred + sqrt(1 - a_n) * eps_pred

# Stochastic step (only during denoising):
if eta > 0 and a_n > a_c:
    var = (1 - a_n)/(1 - a_c) * (1 - a_c/a_n)
    sigma = eta * sqrt(var)
    direction = sqrt(1 - a_n - sigma²) * eps_pred
    x_next = sqrt(a_n) * x0_pred + direction + sigma * randn_like(x)
```

### 4.2 Timestep Conversion (Common Bug Source)

For HuggingFace DDPM models:
```python
# alpha_bar → integer timestep (0 = clean, 999 = noisy)
timestep = int((1 - alpha_bar) * num_train_timesteps)
timestep = clamp(timestep, 0, num_train_timesteps - 1)
```

Do NOT use `t = int(alpha_bar * 999)` — this inverts the mapping.

### 4.3 Schedule Ordering Convention

- `make_schedule()` returns alpha_bars in **sampling order**: index 0 = high noise (low α̅), index N-1 = low noise (high α̅ ≈ 1).
- For non-monotonic schedules, some consecutive entries have α̅[i+1] < α̅[i] — these are reheat steps.
- The sampler iterates from index 0 to N-1.

### 4.4 FID Computation Pitfalls

- Images MUST be in [0, 255] uint8 format, NOT [-1, 1] float.
- Convert: `imgs = ((imgs.clamp(-1,1) + 1) / 2 * 255).to(torch.uint8)`
- Use the SAME reference statistics for all comparisons.
- FID-50K means 50,000 generated images compared against the full training set statistics.
- For fast iteration, FID-10K is acceptable but note it in results.

### 4.5 Memory Management

- Generate images in batches (64 or 128).
- Save batches to disk incrementally, don't hold 50K images in RAM.
- Clear CUDA cache between experiments: `torch.cuda.empty_cache()`

---

## 5. WHAT TO DO WHEN THINGS GO WRONG

### Model Download Fails
1. Log the error in PROGRESS.md.
2. Try alternative sources (HuggingFace, official GitHub releases, torch.hub).
3. If no pretrained model is available, implement training from scratch and document the training hyperparameters used.
4. Never silently substitute a different model.

### FID is Unreasonably High (>100 on CIFAR-10)
1. Check image normalization (the #1 cause).
2. Check that the model is receiving the correct timestep format.
3. Check that the schedule is in the correct order.
4. Generate a small batch and visually inspect — are they recognizable images?
5. Run the monotonic baseline first — if baseline FID is bad, the problem is in the model wrapper, not the schedule.

### Negative Results
- This is expected with ~35% probability (per the paper's own assessment).
- **Do NOT fabricate positive results.**
- Log the negative result honestly in PROGRESS.md.
- Continue with other experiments — negative results on one model don't invalidate others.
- If ALL results are negative, produce the "Why Monotonic Sampling Works" analysis instead.

### Out of GPU Memory
- Reduce batch size.
- Use `torch.cuda.amp.autocast()` for float16 inference.
- For FID computation, generate and save images in chunks.

---

## 6. PROGRESS REPORTING

After EVERY completed task, update `PROGRESS.md` with:

```markdown
### [TASK NAME] — [STATUS: DONE / IN PROGRESS / BLOCKED / FAILED]
- **Date:** YYYY-MM-DD
- **What was done:** Brief description
- **Key results:** Any metrics or outputs
- **Blockers:** Any issues encountered
- **Next step:** What to do next
```

After completing each experiment, add a results summary:
```markdown
### Experiment X Results Summary
| Config | NFE | FID | IS | Notes |
|--------|-----|-----|----|-------|
| ...    | ... | ... | .. | ...   |
```

---

## 7. QUALITY CHECKLIST (Before Marking Any Phase DONE)

- [ ] All functions have docstrings
- [ ] All experiments save results to CSV
- [ ] All figures save to PDF and PNG at 300 DPI
- [ ] Seeds are set for reproducibility
- [ ] No hardcoded absolute paths (use relative paths from project root)
- [ ] PROGRESS.md is updated with results
- [ ] No silent failures (all errors are logged or raised)

---

## 8. RESUME PROTOCOL

When starting a new session:

```
1. cat PROGRESS.md
2. Find the last "IN PROGRESS" or "DONE" entry
3. Continue from the next undone task
4. Do NOT re-run completed experiments unless results are missing
5. If results/ files exist, verify them before re-running
```

---

## 9. FINAL DELIVERABLES CHECKLIST

When ALL phases are complete, verify:

- [ ] `results/tables/` contains CSV files for all 9 experiments
- [ ] `results/figures/` contains PDF+PNG for all 7 main figures
- [ ] `PROGRESS.md` has complete status for all phases
- [ ] `README.md` has full documentation
- [ ] All code runs with `python experiments/expN_*.py` from project root
- [ ] `figures/make_all.py` regenerates all figures from saved data
- [ ] LaTeX tables are generated for the paper
