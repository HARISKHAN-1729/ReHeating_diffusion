# Reheat to Refine: Non-Monotonic Noise Schedules for Diffusion Sampling

## A Complete Research Synthesis — Revised

---

## 1. PAPER METADATA

**Title (final):** *Reheat to Refine: Non-Monotonic Noise Schedules Improve Diffusion Sampling*

**Venue:** NeurIPS 2026

**Paper type:** 9-page empirical paper with lightweight theory. Primarily a "rethinking assumptions" contribution backed by thorough experiments across model families.

**One-sentence pitch:** Diffusion models always sample by monotonically decreasing noise — we show that strategically *increasing* noise at intermediate steps (reheating) yields better samples at the same compute cost, and that stochastic DDPM sampling already does this accidentally.

---

## 2. THE CORE INSIGHT

### 2.1 The Assumption Nobody Questions

Every diffusion sampler — DDPM, DDIM, EDM, Flow Matching, Consistency Models — traverses noise levels monotonically from high to low during generation. The signal-to-noise ratio strictly increases at every step. No paper states this as an assumption. It is treated as axiomatic.

Formally: given a sampling trajectory visiting noise levels σ₁ > σ₂ > ... > σ_N = 0, the ordering is always strictly decreasing. The cumulative signal retention α̅ is always monotonically increasing.

### 2.2 The Simple Idea

**What if sampling visited noise levels non-monotonically?**

Instead of marching straight from noise to clean image, the sampler occasionally *re-noises* its current estimate — steps backward to higher noise — before continuing forward. Like a sculptor who steps back, squints, smears a section that looked wrong, and then re-sculpts it.

### 2.3 The Killer Observation: Stochastic DDPM Already Does This

Here is the argument the field has missed. Compare DDIM (η=0) and DDPM (η=1) sampling:

- **DDIM** predicts x̂₀, then deterministically moves to the next noise level. Monotonic. No noise re-injected.
- **DDPM** predicts x̂₀, moves to the next noise level, then *adds fresh random noise*. The added noise partially *undoes* the denoising step — it pushes the sample back toward higher noise.

DDPM sampling is already mild, unstructured reheating. And it often outperforms DDIM at low NFE (10–50 steps). This is well-documented but never explained through this lens.

Our proposal: **make the reheating deliberate, structured, and optimized.** Instead of adding random noise at every step (DDPM) or never (DDIM), strategically choose *when* to reheat, *how much* to reheat, and *where in the noise schedule* to reheat. This turns an accidental property of stochastic sampling into a first-class design axis.

### 2.4 Three Mechanisms (Intuitions, Not Proofs)

We offer three intuitions for why reheating should help. We are explicit: these are motivating hypotheses to be tested empirically, not formal guarantees.

**Mechanism 1: Error correction at the critical zone.** Denoising errors at mid-range noise levels (log-SNR ≈ 0) determine global structure — pose, layout, object identity. Monotonic sampling commits to these decisions in one pass. Reheating gives the model a second pass through the critical zone, starting from a partially-corrected state. The re-noised intermediate retains coarse structure (low frequencies survive moderate noise) while resetting fine-grained commitments.

**Mechanism 2: Effective compute reallocation.** Not all noise levels are equally important. The noise schedule literature (Karras EDM, Hang et al. ICCV 2025) shows that mid-range noise levels dominate perceptual quality. Monotonic schedules spread NFE uniformly across the entire SNR range, including regions where the model barely changes the image. Reheating concentrates effective compute at the critical zone — spending more of the NFE budget where it matters most.

**Mechanism 3: Mode exploration.** At high noise, the model's prediction x̂₀ is an average over plausible images — a blurry "mean face" rather than a specific face. As noise decreases, the model commits to a specific mode. Re-noising at intermediate levels allows the trajectory to partially de-commit, potentially settling into a different (better) mode on the second pass. This is the intuition behind simulated annealing — periodically raising temperature helps escape local optima — applied as an analogy, not a formal equivalence.

**What these intuitions are NOT:** We do not claim reheating produces independent error estimates (the re-noised sample is constructed from the model's own prediction, so errors are correlated). We do not claim this satisfies detailed balance (diffusion sampling was never MCMC). We do not claim formal convergence guarantees. These are empirical hypotheses. The experiments test them.

---

## 3. MATHEMATICAL FRAMEWORK

### 3.1 Why Non-Monotonic Sampling Is Already Valid (No Retraining)

The DDIM update rule operates on arbitrary pairs of noise levels:

```
x_{σ'} = √α̅_{σ'} · x̂₀(x_σ, σ) + √(1 − α̅_{σ'}) · ε_θ(x_σ, σ)
```

where x̂₀ is the predicted clean image. This equation is valid for *any* σ and σ', regardless of ordering:

- When α̅_{σ'} > α̅_σ (less noise): standard denoising step.
- When α̅_{σ'} < α̅_σ (more noise): the step *re-noises* the predicted clean image.

The denoiser ε_θ was trained on all noise levels via random t-sampling. It can answer queries at any σ. **No retraining is needed.** Any pretrained diffusion model — DDPM, EDM, flow matching — can be sampled with a non-monotonic schedule by changing only the sampler code.

### 3.2 Formal Definitions

**Definition 1 (Reheating schedule).** A sequence of noise levels S = (σ₁, σ₂, ..., σ_N) where σ₁ = σ_max, σ_N = 0, and ∃ i such that σ_{i+1} > σ_i (noise increases at step i — a "reheat" step).

**Definition 2 (Reheating ratio).** At step i, r_i = σ_{i+1}/σ_i. Steps with r_i > 1 are reheats; r_i < 1 are standard denoises.

**Definition 3 (Reheating overhead).** Let L = Σᵢ |log-SNR(σ_{i+1}) − log-SNR(σᵢ)| be the path length, and Δ = |log-SNR(σ_N) − log-SNR(σ₁)| be the net progress. The reheating overhead is O = L/Δ − 1. For monotonic schedules, O = 0.

### 3.3 Schedule Families

We propose five schedule families, ordered from simplest to most adaptive.

**Family 1: Monotonic Baseline**
Standard cosine or EDM schedule. O = 0. The control condition.

**Family 2: Single Reheat (Bump)**
One interruption at a chosen point. At fraction t_reheat through sampling, jump backward by δ in log-SNR space, then resume monotonic decrease.

Hyperparameters: t_reheat ∈ [0.2, 0.8], δ ∈ [0.05, 0.5].

This is the minimal non-monotonic schedule — answers "does any reheating help at all?"

**Family 3: Periodic Sawtooth**
Monotonic decrease with periodic partial re-noising. Every k steps, jump back by fixed fraction δ.

Hyperparameters: period k, magnitude δ.

Answers "does repeated reheating compound benefits?"

**Family 4: Damped Oscillation**
Physically-inspired schedule: large oscillations early (where global structure errors matter most), smoothly transitioning to monotonic at the end (where fine details are set):

```
α̅(t) = cosine(t) + A · exp(−γt) · sin(2πft)
```

Hyperparameters: amplitude A, damping γ, frequency f.

Answers "should reheating decay over time?"

**Family 5: Adaptive Reheat (First-Class Citizen)**

The most principled variant. Instead of fixed schedules, trigger reheating when the model signals uncertainty.

**Confidence metric:** At each step, compute the change in predicted x̂₀ between consecutive steps:

```
Δ_i = ‖x̂₀(x_{σ_i}, σ_i) − x̂₀(x_{σ_{i-1}}, σ_{i-1})‖₂ / √(d)
```

normalized by dimensionality d. Large Δ means the model's prediction is unstable — it's changing its mind about what the clean image looks like. This signals the trajectory is near a mode boundary or the model made a mistake it's now trying to correct.

**Reheat trigger:** When Δ_i exceeds a threshold τ (calibrated per-model as the 80th percentile of Δ over a small calibration run), trigger a reheat: jump back by δ_adaptive in log-SNR space.

**Why this is better than fixed schedules:** It reheats *only when needed* — spending budget on correction precisely when the model is struggling, rather than on a fixed cadence that may waste steps where the model is confident.

**Noise-level normalization:** Raw prediction norms are meaningless across noise levels (high norm at high noise is expected). The Δ metric avoids this by measuring *change* between consecutive predictions, which is naturally comparable across noise levels.

**Calibration:** Run 100 monotonic samples, record Δ_i at each step, compute the threshold τ as the 80th percentile. This takes ~2 minutes and is done once per model.

### 3.4 Connection to Stochastic Sampling (The Unifying View)

Stochastic DDPM sampling (η > 0) adds noise at every step:

```
x_{t-1} = μ_θ(x_t, t) + σ_t · z,  z ~ N(0, I)
```

The σ_t · z term partially re-noises the sample. At η = 1 (full DDPM), this is equivalent to predicting x̂₀ and then constructing x_{t-1} by mixing x̂₀ with fresh noise at the appropriate ratio.

Our reheating schedules generalize this:
- **DDIM (η=0):** No reheating. Fully deterministic.
- **DDPM (η=1):** Uniform, unstructured reheating at every step.
- **Reheating schedules:** Strategic, variable-magnitude reheating at chosen steps.

This positions our work as *interpolating between and extending beyond DDIM and DDPM*, rather than proposing something alien. DDPM's advantage over DDIM at low NFE *is* the reheating effect. We optimize it.

---

## 4. RELATED WORK

### 4.1 Directly Related

**RePaint (Lugmayr et al., CVPR 2022).** Uses forward-backward jumps during sampling for inpainting. The known image regions provide a strong conditioning anchor that survives renoising. In unconditional generation, no such anchor exists. We do *not* claim to generalize RePaint — the mechanisms differ. But RePaint demonstrates that re-noising during sampling is not catastrophic, which supports our hypothesis at a basic level.

**SDEdit (Meng et al., 2021).** Adds noise to a guide image, then denoises. A single re-noising event for guided editing. Our schedules apply multiple structured re-noisings during unconditional generation.

**Stochastic DDPM sampling.** The most important related work, and the one we build on directly. DDPM's noise injection during sampling is implicit, uniform reheating. We make it explicit and structured.

### 4.2 Noise Schedule Literature

**VDM (Kingma et al., NeurIPS 2021):** Proved continuous-time ELBO is invariant to scalar noise schedules (only endpoints matter). Critically, this applies to *training*, not sampling — sampling schedule design has real impact on sample quality.

**MuLAN (Sahoo et al., NeurIPS 2024):** Broke VDM's invariance with per-pixel multivariate schedules. Orthogonal to our work (they modify training; we modify sampling).

**EDM (Karras et al., NeurIPS 2022):** Principled framework for noise schedule design. All schedules are monotonic.

**"Align Your Steps" (Sabour et al., 2024):** Optimizes sampling schedules post-hoc. Monotonic only.

**"Improved Noise Schedule" (Hang et al., ICCV 2025):** Shows mid-range noise levels (log-SNR ≈ 0) are most important for training. Our reheating concentrates sampling compute at the same critical zone.

### 4.3 Non-Gaussian and Non-Standard Forward Processes

**Cold Diffusion (Bansal et al., NeurIPS 2023), Soft Diffusion (Daras et al., TMLR 2023), Blurring Diffusion (Hoogeboom & Salimans, ICLR 2023).** All explore non-standard corruption but with monotonic schedules.

**DiracDiffusion (2024):** Explicitly notes "the requirement for monotonicity does not necessarily arise in image generation" but does not pursue the idea.

### 4.4 Timestep-Free Diffusion

**Blind DDMs (Kadkhodaie et al., Feb 2026), Equilibrium Matching (Wang & Du, Oct 2025).** Show diffusion can work without timestep conditioning. Orthogonal — we keep timestep conditioning but change the order of timesteps visited.

### 4.5 What Makes This Paper Different From All of Them

No prior work proposes or tests non-monotonic noise schedules for unconditional diffusion sampling as a principled design axis. No prior work frames stochastic DDPM sampling as implicit reheating or proposes to optimize the reheating pattern.

---

## 5. EXPERIMENTAL PLAN

### 5.1 Design Principles

Three principles govern the experimental design:

1. **Run the make-or-break experiment first.** CIFAR-10 FID with a pretrained model. If this doesn't show improvement, the paper doesn't exist.

2. **Test across model families.** DDPM, EDM2, and flow matching. A finding that only works on DDPM is unpublishable at NeurIPS 2026.

3. **Fair comparisons only.** All comparisons at matched NFE *and* matched wall-clock time (since reheating costs the same per step, these collapse to the same comparison, but we report both).

### 5.2 Overview Table

| # | Experiment | Dataset | Model | Measures | GPU-hrs |
|---|-----------|---------|-------|----------|---------|
| 1 | 2D Proof of Concept | Synthetic 2D | Small MLP | Mode coverage, trajectories | 0.5 |
| 2 | CIFAR-10 (DDPM) | CIFAR-10 | Pretrained DDPM | FID, IS at varied NFE | 1.5 |
| 3 | CIFAR-10 (EDM2) | CIFAR-10 | Pretrained EDM2 | FID, IS at varied NFE | 1.5 |
| 4 | ImageNet-64 (EDM2) | ImageNet-64 | Pretrained EDM2 | FID at varied NFE | 2 |
| 5 | Flow Matching | CIFAR-10 | Pretrained OT-CFM | FID at varied NFE | 1.5 |
| 6 | Schedule Ablation | CIFAR-10 | EDM2 | FID vs. (t_reheat, δ) | 2 |
| 7 | NFE Pareto + Wall-Clock | CIFAR-10 | All models | FID vs. NFE and FID vs. seconds | 1 |
| 8 | Adaptive vs. Fixed | CIFAR-10 | EDM2 | FID, reheat frequency analysis | 1 |
| 9 | Failure Characterization | CIFAR-10 | EDM2 | When reheating hurts, and why | 1 |
| **Total** | | | | | **~12 hrs** |

All experiments fit on a single A100 in one weekend.

### 5.3 Experiment 1: 2D Proof of Concept (The Hero Figure)

**Goal:** Visually demonstrate mode exploration via reheating.

**Setup:**
- Data: Mixture of 8 Gaussians in a ring (standard mode collapse benchmark)
- Model: 4-layer MLP, sinusoidal time embedding, ~50K params
- Train: Standard DDPM objective, 50K steps (~10 minutes)
- Sample: 10K points each with monotonic, single reheat, damped oscillation, adaptive

**What to measure and plot:**
- Number of modes covered (out of 8) vs. NFE budget
- Wasserstein-2 distance to ground truth
- **Key figure:** Sampling trajectories overlaid on true density contours. Monotonic trajectories lock into a mode early. Reheating trajectories oscillate and explore before committing.

**Why this works for certain:** Mode coverage in 2D is a known failure mode of low-NFE monotonic sampling. Reheating gives the trajectory more chances to jump between modes. The question is only *how much* it helps — not *whether*.

**What this experiment does NOT prove:** Better 2D mode coverage does not guarantee better FID on images. We state this explicitly and use it only as a visual proof-of-concept, not as evidence for the image results.

### 5.4 Experiment 2: CIFAR-10 with Pretrained DDPM

**Goal:** Test the core hypothesis on the simplest standard benchmark.

**Setup:**
- Model: `google/ddpm-cifar10-32` (HuggingFace). ε-prediction, 1000 training timesteps.
- Generate: 50K images per configuration.
- Compute: FID-50K (pytorch-fid), IS.

**Configurations (all at matched NFE):**

| Config | Schedule | NFE |
|--------|----------|-----|
| DDIM baseline | Monotonic cosine, η=0 | 10, 25, 50, 100 |
| DDPM baseline | Monotonic cosine, η=1 | 10, 25, 50, 100 |
| Single reheat | Bump at t=0.4, δ=0.15, η=0 | 10, 25, 50, 100 |
| Damped oscillation | A=0.2, γ=2.5, f=4, η=0 | 10, 25, 50, 100 |
| Adaptive reheat | τ=80th percentile, η=0 | 10, 25, 50, 100 |
| Reheat + stochastic | Best fixed schedule, η=0.5 | 10, 25, 50, 100 |

**Critical baselines:**
- DDIM (no reheating, no stochasticity)
- DDPM (unstructured reheating via stochasticity)
- Reheating (structured reheating, no stochasticity)
- Reheating + stochastic (structured + unstructured combined)

The DDPM baseline is essential. If DDPM already beats DDIM at low NFE (documented in literature), and our reheating schedule beats DDPM, that demonstrates that *structured* reheating outperforms *random* reheating — the core claim.

### 5.5 Experiment 3: CIFAR-10 with Pretrained EDM2

**Goal:** Validate on a modern, high-performing architecture.

**Setup:**
- Model: EDM2-S (small) pretrained on CIFAR-10. Available from Karras et al.'s official repository.
- EDM2 uses a different parameterization (σ-based, not α̅-based) and Heun's 2nd-order sampler.
- Adapt reheating schedules to EDM2's σ(t) parameterization.

**Why EDM2 matters:** EDM2 represents the current state of the art for unconditional image generation. If reheating doesn't help EDM2, the finding is limited to legacy architectures and loses NeurIPS relevance. If it does help, the impact is immediate — EDM2 is widely used.

**Adaptation for EDM2:** EDM2 parameterizes noise as σ(t) with the Heun sampler. Reheating in this framework means the σ sequence includes segments where σ increases. The Heun sampler's error-correction step already does a forward-backward evaluation — our reheating adds a *noise-level* forward-backward on top.

### 5.6 Experiment 4: ImageNet-64 with Pretrained EDM2

**Goal:** Test on a larger, more diverse dataset.

**Setup:**
- Model: EDM2 pretrained on ImageNet 64×64.
- Pretrained checkpoint from Karras et al. official release.
- Generate 50K images, compute FID.
- Test only the best 2-3 schedules from Exp 2-3 (not the full grid).

**Why this matters:** CIFAR-10 is 32×32 with 10 classes. ImageNet-64 has 1000 classes and much more diversity. If reheating helps mode coverage, the benefit should be *larger* on ImageNet-64 (more modes to explore).

### 5.7 Experiment 5: Flow Matching Model

**Goal:** Test whether reheating generalizes beyond score-based diffusion to flow matching.

**Setup:**
- Model: Optimal-transport conditional flow matching (OT-CFM) trained on CIFAR-10.
  - Train a small U-Net with OT-CFM objective (~2 hours on A100).
  - Or use a pretrained checkpoint if available.
- Flow matching uses a velocity field v_θ(x, t) and an ODE from t=0 (noise) to t=1 (data).
- Reheating = non-monotonic time traversal in the ODE integration. Instead of t: 0→1 monotonically, visit t-values that sometimes decrease.

**Why this matters:** Flow matching is rapidly replacing score-based diffusion in practice (Stable Diffusion 3, Flux). If reheating works for flow matching, the impact doubles.

**Adaptation for flow matching:** The Euler step is:
```
x_{t+Δt} = x_t + Δt · v_θ(x_t, t)
```
A reheat step uses negative Δt — stepping backward along the flow. The velocity field v_θ is defined for all t ∈ [0,1], so backward steps are valid.

### 5.8 Experiment 6: Schedule Shape Ablation (The Key Contribution)

**Goal:** Map the (reheat_timing, reheat_magnitude) landscape systematically.

**Setup:**
- Model: Best-performing model from Exp 2-5 (likely EDM2 on CIFAR-10).
- For single reheat: grid over t_reheat ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} × δ ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}
- Total: 42 configurations × 10K images each ≈ 2 GPU-hours.

**Output:** A heatmap of FID vs. (t_reheat, δ). This is the paper's central empirical contribution — it tells practitioners *where* to reheat and *how much*.

**Hypothesis:** Optimal reheat sits at t_reheat ≈ 0.3-0.5 (the critical zone where log-SNR ≈ 0 and global structure is decided) with moderate δ ≈ 0.1-0.2. Too early (high noise) wastes the reheat on already-noisy states. Too late (low noise) disrupts fine details. Too strong disrupts coherent structure. Too weak has no effect.

### 5.9 Experiment 7: NFE Pareto Frontier + Wall-Clock Time

**Goal:** Show that reheating shifts the quality-compute tradeoff curve.

**Setup:**
- For each model (DDPM, EDM2, flow matching):
  - Generate 50K images at NFE ∈ {5, 10, 15, 20, 25, 50, 100, 200}
  - With: monotonic DDIM, monotonic DDPM, best reheating (deterministic), best reheating (stochastic)
- Measure FID and wall-clock time for each.

**Key plots:**
1. FID vs. NFE (quality vs. compute budget)
2. FID vs. wall-clock seconds (quality vs. real time)

Since each NFE costs the same regardless of noise direction, these plots will be identical up to a constant factor. We report both because reviewers respond differently to each framing.

**Expected result:** Reheating shifts the Pareto frontier, with the largest gains at low NFE (5-50 steps). At high NFE (200+), monotonic schedules have enough steps to self-correct, and the advantage diminishes.

**The NFE-fairness argument, made precise:** A critic might argue that with 100 NFE and 20 reheat steps, you get "only 80 productive steps." This assumes all denoising steps have equal marginal value — which is false. Steps in the low-noise regime barely change the image. Reheating trades low-value steps in the easy regime for high-value revisits to the critical zone. The Pareto plot tests this directly: if reheating at 80 effective forward steps + 20 reheat steps beats monotonic at 100 forward steps, the trade was worth it.

### 5.10 Experiment 8: Adaptive vs. Fixed Schedules

**Goal:** Test whether data-dependent reheating outperforms fixed schedules.

**Setup:**
- Model: EDM2 on CIFAR-10.
- Calibration run: 100 monotonic samples, record Δ_i (prediction change) at each step, compute threshold τ = 80th percentile.
- Generate 50K images with adaptive schedule.
- Compare FID to: best fixed schedule, DDIM, DDPM.

**Additional analysis:**
- Log *where* the adaptive schedule actually reheats across 1000 samples.
- Plot histogram of reheat step positions — does the adaptive schedule independently discover that the critical zone is the right place to reheat?
- If yes, this is strong evidence for Mechanism 2 (compute reallocation to critical zone).

**Sensitivity analysis:** Vary τ from 50th to 95th percentile. Plot FID vs. τ to check robustness.

### 5.11 Experiment 9: Failure Characterization

**Goal:** Understand *when and why* reheating hurts. This may be more interesting than the success cases.

**Setup:**
- Run aggressive reheating schedules that are expected to fail (δ = 0.5, δ = 1.0, multiple large reheats).
- For each, generate 1K images and manually inspect.
- Compute FID, but also decompose into precision (quality) and recall (diversity) using the Kynkäänniemi et al. metric.

**Questions to answer:**
1. Does aggressive reheating hurt precision more than it helps recall? (Expected: yes.)
2. At what δ does the quality-diversity tradeoff flip? (Critical parameter.)
3. Do failed reheat samples show specific artifacts — blurriness, inconsistent structure, mode mixing?
4. Does combining reheating with stochastic sampling (η > 0) help or hurt in the failure regime?

**Why this experiment matters for NeurIPS:** Reviewers increasingly value papers that characterize *when methods fail*, not just when they succeed. A thorough failure analysis strengthens the paper even if gains are moderate.

---

## 6. IMPLEMENTATION

### 6.1 Core Algorithm

```python
def sample_with_reheating(model, alpha_bars_sampling, shape, eta=0.0):
    """
    Generalized DDIM sampler supporting non-monotonic schedules.
    alpha_bars_sampling: sequence of alpha_bar values, ordered for sampling
                         (first = low alpha / high noise, last = high alpha / low noise).
    Works identically to DDIM when schedule is monotonic.
    ~10 lines different from standard DDIM.
    """
    x = torch.randn(shape)
    
    for i in range(len(alpha_bars_sampling) - 1):
        a_curr = alpha_bars_sampling[i]
        a_next = alpha_bars_sampling[i + 1]
        
        # Standard: predict noise and clean image
        eps_pred = model(x, noise_level=a_curr)
        x0_pred = (x - sqrt(1 - a_curr) * eps_pred) / sqrt(a_curr)
        x0_pred = x0_pred.clamp(-1, 1)
        
        # === THE ONLY DIFFERENCE FROM STANDARD DDIM ===
        # This update works regardless of whether a_next > a_curr (denoise)
        # or a_next < a_curr (reheat). The math is identical.
        if eta > 0 and a_next > a_curr:  # Stochastic only when denoising
            sigma = eta * sqrt((1-a_next)/(1-a_curr) * (1 - a_curr/a_next))
            direction = sqrt(1 - a_next - sigma**2) * eps_pred
            x = sqrt(a_next) * x0_pred + direction + sigma * torch.randn_like(x)
        else:
            x = sqrt(a_next) * x0_pred + sqrt(1 - a_next) * eps_pred
    
    return x0_pred
```

### 6.2 Adaptive Reheat Implementation

```python
def sample_adaptive(model, base_schedule, shape, tau, delta=0.15, eta=0.0):
    """
    Adaptive reheating: trigger reheat when prediction changes sharply.
    tau: threshold for prediction change (calibrated per-model).
    """
    x = torch.randn(shape)
    prev_x0_pred = None
    schedule = list(base_schedule)  # Mutable copy
    i = 0
    nfe = 0
    max_nfe = len(base_schedule) - 1  # Budget cap
    
    while i < len(schedule) - 1 and nfe < max_nfe:
        a_curr = schedule[i]
        
        eps_pred = model(x, noise_level=a_curr)
        x0_pred = (x - sqrt(1 - a_curr) * eps_pred) / sqrt(a_curr)
        x0_pred = x0_pred.clamp(-1, 1)
        nfe += 1
        
        # Check if prediction changed sharply
        if prev_x0_pred is not None:
            delta_pred = (x0_pred - prev_x0_pred).pow(2).mean().sqrt()
            if delta_pred > tau and nfe < max_nfe - 5:  # Keep budget for final steps
                # Trigger reheat: jump backward in log-SNR
                a_reheat = a_curr * (1 - delta)  # Lower alpha = more noise
                a_reheat = max(a_reheat, 0.001)
                # Insert reheat into schedule
                schedule.insert(i + 1, a_reheat)
        
        prev_x0_pred = x0_pred.detach()
        
        # Standard DDIM step to next noise level
        a_next = schedule[i + 1]
        x = sqrt(a_next) * x0_pred + sqrt(1 - a_next) * eps_pred
        i += 1
    
    return x0_pred
```

### 6.3 Adaptation for EDM2

EDM2 parameterizes noise as σ and uses preconditioning: D(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ)). The sampling step with Heun's method uses σ directly. A reheating step in EDM2:

```python
# Standard EDM2 step: move from sigma[i] to sigma[i+1] where sigma[i] > sigma[i+1]
# Reheating step: sigma[i+1] > sigma[i] — just let it happen naturally
# The denoiser D(x; sigma) is valid for any sigma
x_next = x_curr + (sigma_next - sigma_curr) * d_cur  # d_cur = (x - D(x;sigma)) / sigma
# When sigma_next > sigma_curr, this adds noise. No code change needed.
```

### 6.4 Adaptation for Flow Matching

Flow matching uses velocity v_θ(x, t) with t ∈ [0, 1]:

```python
# Standard: t goes 0 → 1 (noise → data), dt > 0
# Reheating: sometimes dt < 0, stepping backward in time
x = x + dt * v_theta(x, t)  # dt can be negative!
```

The velocity field is continuous and defined everywhere in [0,1]. Backward steps are valid Euler steps of the same ODE.

---

## 7. THEORY (LIGHTWEIGHT AND HONEST)

### 7.1 What We Claim (And What We Don't)

We make three theoretical observations. We are explicit about their limitations.

### 7.2 Observation 1: Reheating as Compute Reallocation

**Claim.** A reheating schedule with N total NFE visits certain noise levels twice — once during the initial descent and once after reheating. The total number of model evaluations in the critical zone (|log-SNR| < C for some constant C) is higher for reheating than for a monotonic schedule with the same N.

**Proof.** Direct counting. A monotonic schedule with N steps passes through each log-SNR level at most once. A schedule with k reheat steps through the critical zone passes through it k+1 times. ∎

**Limitation.** More evaluations in a zone doesn't automatically mean better quality — it depends on whether the model's predictions improve with the additional context from the reheated state. This is an empirical question.

### 7.3 Observation 2: Reheating Reduces Prediction Variance

**Claim.** Let x₀* be the true clean image. For a fixed random seed, let x̂₀^{mono} be the prediction after monotonic sampling, and x̂₀^{reheat} be the prediction after a single reheat at the critical zone. Under the assumption that the denoiser's error at the second visit is strictly positively correlated but not perfectly correlated with the first visit's error:

```
Var[x̂₀^{reheat}] < Var[x̂₀^{mono}]
```

if and only if the correlation ρ between the two errors satisfies ρ < 1.

**Proof sketch.** The variance of the average of two estimators with correlation ρ is (1 + ρ)/2 times the individual variance. For ρ < 1, this is less than 1. ∎

**What we do NOT claim:**
- We do not claim ρ is small. It may be close to 1 for mild reheating.
- We do not claim the variance reduction translates directly to FID improvement.
- The "averaging" is implicit (through the DDIM update) not explicit.

This observation motivates the hypothesis; the experiments test it.

### 7.4 Observation 3: Analogy to Simulated Annealing (Intuition Only)

The noise level σ in diffusion sampling plays a role analogous to temperature in statistical physics: high σ smooths the energy landscape, low σ sharpens it. Oscillating σ during sampling is analogous to oscillating temperature in simulated annealing, which helps escape local optima.

**This is an analogy, not a formal equivalence.** We do not claim:
- Detailed balance (diffusion sampling is not MCMC)
- Convergence to the Boltzmann distribution
- That SA theory directly applies

We use the analogy purely to motivate the design space. The experiments justify the method, not the analogy.

---

## 8. KEY FIGURES (7 MAIN FIGURES)

1. **Figure 1 (Teaser/Hero).** Three panels: (a) 2D sampling trajectories — monotonic vs. reheating, overlaid on true density. (b) Mode coverage vs. NFE for both methods. (c) FID vs. NFE on CIFAR-10 showing reheating Pareto frontier.

2. **Figure 2 (Method Schematic).** The reheating process shown as a trajectory through noise-level space. Annotate: "denoise" arrows (decreasing σ), "reheat" arrows (increasing σ), and the DDIM update equation showing both directions use the same formula.

3. **Figure 3 (Schedule Families).** Five subplots showing α̅(t) for all schedule families. Red shading highlights reheat regions. Include the adaptive schedule's realized trajectory for one sample.

4. **Figure 4 (Cross-Model Results).** Bar charts: FID at 50 NFE for DDIM, DDPM, best reheating, across DDPM/EDM2/Flow Matching/ImageNet-64. This is the key evidence for generality.

5. **Figure 5 (Ablation Heatmap).** FID as function of (t_reheat, δ) for single-reheat schedule. Clear optimal region highlighted.

6. **Figure 6 (NFE Pareto Frontier).** FID vs. NFE curves for all methods. Both linear and log-scale NFE axes. Reheating curve sits below monotonic curves (lower FID = better).

7. **Figure 7 (Adaptive Analysis).** (a) Histogram of where adaptive reheating triggers across 1K samples. (b) FID vs. threshold τ showing robustness. (c) Per-sample comparison: samples where adaptive reheat triggered vs. didn't — visual examples.

---

## 9. ANTICIPATED REVIEWER OBJECTIONS (REVISED)

### Objection 1: "This is just wasting NFE. More monotonic steps would be better."

**Response (in paper):** This assumes all denoising steps have equal marginal value. They don't — steps in the low-noise regime barely change the image (documented in EDM, Hang et al. 2025). Reheating trades low-value steps for high-value revisits to the critical zone. Figure 6 tests this directly: if reheating at N steps beats monotonic at N steps, the trade was worth it.

### Objection 2: "Isn't DDPM already doing this?"

**Response:** Yes — and this is our key insight, not an objection. DDPM sampling injects noise at every step, uniformly and randomly. Our contribution is to make this reheating *strategic*: choosing when and how much to reheat. The improvement of reheating over DDPM (shown in Table 1) demonstrates that structure matters. DDPM wastes reheat budget on easy steps; we concentrate it on hard ones.

### Objection 3: "The theory doesn't prove anything."

**Response:** We agree — and we are explicit about this in Section 7.1. The theory provides three motivating observations, clearly marked as intuitions. The paper's contribution is empirical: the finding that non-monotonic schedules improve generation quality is established through controlled experiments across three model families, not through theory. We believe this is the appropriate level of theory for a "rethinking assumptions" paper.

### Objection 4: "Why not test on text-to-image / larger scale?"

**Response:** Single-A100 compute constraint. We test on EDM2 (current SOTA architecture) and flow matching (current SOTA paradigm) on datasets up to ImageNet-64. Extension to latent diffusion / text-to-image is immediate — our method changes only the sampler, not the model — and we list this as future work.

### Objection 5: "RePaint already did renoising during sampling."

**Response:** RePaint re-noises for inpainting conditioning, where the known image regions anchor the generation. In unconditional generation, no such anchor exists. Our method works through a different mechanism (compute reallocation and error correction vs. known-unknown harmonization), and we test on unconditional generation where RePaint's approach does not apply. We do not claim to generalize RePaint.

---

## 10. PAPER STRUCTURE (9 PAGES + APPENDIX)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.25 | Assumption → insight (DDPM already reheats) → method → results |
| 1. Introduction | 1.25 | Monotonicity assumption → DDPM observation → reheating as design axis → contributions |
| 2. Background | 0.75 | DDPM/DDIM/EDM/FM equations, noise schedules (brief) |
| 3. Method | 1.5 | Non-monotonic schedules: definition, 5 families, DDPM connection, adaptive design |
| 4. Theory | 0.75 | Three observations with explicit caveats |
| 5. Experiments | 3.0 | Exp 1–9 with figures and tables |
| 6. Related Work | 0.75 | RePaint, SDEdit, noise schedule literature, stochastic sampling |
| 7. Discussion | 0.5 | Failure characterization, limitations, future work |
| 8. Conclusion | 0.25 | |
| **Total** | **9.0** | |
| Appendix A | 2 | Full ablation tables, FID at all NFE/schedule combos |
| Appendix B | 1 | Additional qualitative samples |
| Appendix C | 1 | Adaptive schedule calibration details |
| Appendix D | 1 | Flow matching and EDM2 adaptation details |

---

## 11. CONTRIBUTIONS LIST (FOR ABSTRACT/INTRODUCTION)

1. We identify a universal unexamined assumption in diffusion sampling — monotonic noise traversal — and show it is unnecessary.

2. We observe that stochastic DDPM sampling already performs implicit, unstructured reheating, and that this partially explains DDPM's advantage over DDIM at low NFE.

3. We propose five families of non-monotonic (reheating) schedules, including an adaptive variant that triggers reheating based on model prediction stability.

4. We demonstrate FID improvements across three model families (DDPM, EDM2, flow matching) and two datasets (CIFAR-10, ImageNet-64) at matched NFE, with no retraining.

5. We characterize when reheating helps and when it hurts, providing practical guidelines for schedule design.

---

## 12. HONEST NEURIPS ASSESSMENT (REVISED)

### What makes this a NeurIPS paper:

- **Novelty:** Genuine — confirmed by literature search across 2022–2026. No prior work tests non-monotonic sampling for unconditional generation as a principled method.
- **Simplicity:** ~10 lines of code change to any DDIM sampler. No retraining.
- **The DDPM insight:** Framing DDPM's stochastic advantage as implicit reheating is original and connects disparate observations in the literature.
- **Cross-model validation:** DDPM + EDM2 + flow matching. Not limited to one architecture.
- **Practical value:** Zero-cost improvement for any pretrained model.

### What could kill it:

- **No FID improvement on images.** If the ablation heatmap is flat or negative everywhere, the paper doesn't exist. This is a ~40% risk based on the RePaint/DDPM evidence supporting the hypothesis, tempered by the legitimate concern that unconditional generation lacks the conditioning anchor that helps RePaint.
- **Only works on DDPM, not EDM2/FM.** This would limit the scope to legacy architectures and signal that the effect is an artifact of DDPM's specific noise schedule.
- **Gains < 2% and inconsistent.** Puts us in "interesting negative result" territory, which NeurIPS doesn't reward in the main track. Could pivot to ICML workshop or TMLR.

### Probability estimates:

- Strong result (≥5% FID improvement, works across models): ~35%
- Moderate result (2-5% improvement, at least 2 models): ~30%
- Weak/negative result: ~35%

These estimates are honest. The 35% strong-result probability is justified by: DDPM's documented advantage over DDIM at low NFE (evidence that noise reinjection helps), RePaint's success (evidence that renoising isn't catastrophic), and the theoretical intuitions above. The 35% negative probability is justified by: no conditioning anchor in unconditional generation, potential for disrupting coherent structure at moderate reheating, and the possibility that DDPM's advantage comes from other aspects of stochastic sampling (exploration of nearby modes via noise) rather than the reheating effect specifically.

### The critical path:

**Week 1 deliverable:** FID numbers for CIFAR-10 with pretrained DDPM at 50 NFE, comparing DDIM vs. DDPM vs. single reheat vs. damped oscillation. If no schedule beats DDPM, reconsider. If any schedule beats DDPM, proceed with full experimental plan.

---

## 13. TIMELINE (4 WEEKS)

| Week | Tasks | Decision Gate |
|------|-------|---------------|
| 1 | Implement reheating sampler. Run Exp 1 (2D) and Exp 2 (CIFAR-10/DDPM). | **GATE: Does any reheating schedule beat DDPM at 50 NFE? If no → pivot.** |
| 2 | Run Exp 3 (EDM2), Exp 5 (flow matching), Exp 6 (ablation grid). | **GATE: Does it work on ≥2 model families? If no → scope down.** |
| 3 | Run Exp 4 (ImageNet-64), Exp 7 (Pareto), Exp 8 (adaptive), Exp 9 (failures). Create figures. | |
| 4 | Write paper. Internal review. Polish. Supplementary. | |

### Pivot strategies:

- **If reheating never beats DDPM:** Write "Why Monotonic Sampling Works: An Empirical Investigation" showing that DDPM's stochastic reheating is already near-optimal. Publishable at a workshop.
- **If works on DDPM only:** Write a focused DDPM paper: "Structured Stochastic Sampling for DDPMs." Target TMLR.
- **If works on DDPM + EDM2 + FM:** Full NeurIPS submission as planned.

---

## 14. CODE STRUCTURE

```
reheating-diffusion/
├── schedules/
│   ├── monotonic.py              # Cosine, linear, EDM sigma schedules
│   ├── single_reheat.py          # Family 2
│   ├── sawtooth.py               # Family 3
│   ├── damped_oscillation.py     # Family 4
│   └── adaptive.py               # Family 5 (calibration + sampling)
├── samplers/
│   ├── ddim_reheating.py         # Generalized DDIM (works non-monotonic)
│   ├── edm2_reheating.py         # EDM2 Heun sampler with reheating
│   └── flow_reheating.py         # Flow matching Euler with reheating
├── experiments/
│   ├── exp1_toy2d.py             # 2D Gaussian mixture
│   ├── exp2_cifar_ddpm.py        # CIFAR-10 with pretrained DDPM
│   ├── exp3_cifar_edm2.py        # CIFAR-10 with pretrained EDM2
│   ├── exp4_imagenet_edm2.py     # ImageNet-64 with pretrained EDM2
│   ├── exp5_cifar_flow.py        # CIFAR-10 with flow matching
│   ├── exp6_ablation.py          # Schedule hyperparameter grid
│   ├── exp7_pareto.py            # NFE Pareto frontier
│   ├── exp8_adaptive.py          # Adaptive vs. fixed comparison
│   └── exp9_failures.py          # Failure characterization
├── eval/
│   ├── fid.py                    # FID-50K computation
│   ├── precision_recall.py       # Kynkäänniemi et al. P&R
│   └── mode_coverage.py          # 2D mode counting
├── figures/
│   └── make_all.py               # Reproduce all paper figures
└── configs/
    └── schedules.yaml             # All schedule hyperparameters
```

---

## 15. FUTURE WORK (FOR DISCUSSION SECTION)

1. **Text-to-image.** Test with Stable Diffusion / Flux. Hypothesis: reheating helps prompt adherence by giving the model multiple passes to satisfy text conditioning.

2. **Learned reheating schedules.** Differentiable schedule optimization — learn the optimal α̅ trajectory end-to-end by backpropagating through the sampler (requires differentiable FID proxy).

3. **Connection to diffusion distillation.** Consistency models and progressive distillation compress many steps into few. Does reheating help in the low-step regime of distilled models?

4. **Theoretical convergence.** Rigorous analysis of non-monotonic probability flow ODEs. Under what conditions does a non-monotonic path converge to the same terminal distribution as a monotonic one?

5. **Reheating for guided generation.** Classifier-free guidance + reheating — does reheating help balance diversity and adherence at high guidance scales?
