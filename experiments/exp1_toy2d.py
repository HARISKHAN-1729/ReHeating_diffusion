"""
Experiment 1: 2D Toy Proof of Concept.

Train (or load) a toy MLP on 8-Gaussian ring mixture, then sample with
each schedule family at multiple NFE values. Measures mode coverage
and Wasserstein-2 distance. Records trajectories for visualisation.

Usage:
    python -m experiments.exp1_toy2d                     # sanity (default)
    python -m experiments.exp1_toy2d --full               # full run
    python -m experiments.exp1_toy2d --device cpu          # CPU fallback
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve project root so imports work when invoked from any directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.toy_mlp import train_toy_mlp, load_toy_mlp, get_8_gaussian_centers, make_8_gaussians
from schedules import make_schedule
from schedules.adaptive import calibrate_threshold
from samplers.ddim_reheating import ddim_sample, adaptive_sample
from eval.mode_coverage import count_modes, wasserstein2


# ── helpers ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_or_train_model(device: str, checkpoint_path: str) -> torch.nn.Module:
    """Load a pretrained toy MLP or train one from scratch."""
    if os.path.isfile(checkpoint_path):
        print(f"Loading pretrained toy MLP from {checkpoint_path}")
        return load_toy_mlp(path=checkpoint_path, device=device)
    print(f"No checkpoint at {checkpoint_path} — training from scratch …")
    return train_toy_mlp(
        n_steps=50000,
        batch_size=512,
        lr=1e-3,
        device=device,
        save_path=checkpoint_path,
        num_train_timesteps=1000,
    )


# ── main experiment ────────────────────────────────────────────────────────

def run_experiment(args):
    """Run the full Experiment 1 loop."""
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = "cpu"

    nfe_list = [int(n) for n in args.nfe.split(",")]
    n_samples = args.n_samples
    batch_size = args.batch_size

    # ── Paths ──────────────────────────────────────────────────────────
    tables_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "toy_mlp.pt")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────
    model = get_or_train_model(device, ckpt_path)
    model.eval()

    # ── Ground-truth reference samples ─────────────────────────────────
    centers, stds = get_8_gaussian_centers(std=0.05)
    ref_samples = make_8_gaussians(n_samples=10000)

    # ── Schedule / method configs ──────────────────────────────────────
    methods = {
        "monotonic_ddim": {
            "family": "monotonic",
            "eta": 0.0,
            "schedule_kwargs": {},
        },
        "monotonic_ddpm": {
            "family": "monotonic",
            "eta": 1.0,
            "schedule_kwargs": {},
        },
        "single_reheat": {
            "family": "single_reheat",
            "eta": 0.0,
            "schedule_kwargs": {"t_reheat": 0.4, "delta": 0.15},
        },
        "damped_oscillation": {
            "family": "damped_oscillation",
            "eta": 0.0,
            "schedule_kwargs": {"A": 0.2, "gamma": 2.5, "freq": 4},
        },
        "adaptive": {
            "family": "monotonic",  # base schedule; adaptive sampler modifies it
            "eta": 0.0,
            "schedule_kwargs": {},
        },
    }

    # ── Results accumulators ───────────────────────────────────────────
    rows = []
    all_trajectories = {}  # key: (method, nfe) -> np.ndarray

    n_traj = 5  # number of trajectories to record per method / NFE

    # ── Adaptive threshold calibration (done once) ─────────────────────
    print("\n=== Calibrating adaptive threshold ===")
    calib_schedule = make_schedule("monotonic", N=25)
    # calibrate_threshold uses fp16+cuda autocast internally; for CPU we
    # patch by running the sampler ourselves below instead.
    if device == "cpu":
        # Run a lightweight manual calibration on CPU (no autocast)
        tau = _calibrate_cpu(model, calib_schedule, n_calibration=50, device=device)
    else:
        tau = calibrate_threshold(
            model=model,
            shape=(1, 2),
            alpha_bars=calib_schedule,
            n_calibration=50,
            percentile=80.0,
            device=device,
            num_train_timesteps=1000,
        )
    print(f"Adaptive tau = {tau:.6f}")

    # ── Main loop ──────────────────────────────────────────────────────
    for method_name, cfg in methods.items():
        for nfe in nfe_list:
            print(f"\n--- {method_name}  NFE={nfe} ---")
            t0 = time.time()

            schedule = make_schedule(cfg["family"], N=nfe, **cfg["schedule_kwargs"])

            # --- sample with trajectory recording (small batch) ----------
            traj_shape = (n_traj, 2)
            set_seed(args.seed)  # reset seed for fair trajectory comparison

            if method_name == "adaptive":
                traj_samples = _sample_adaptive_with_traj(
                    model=model,
                    base_schedule=schedule,
                    shape=traj_shape,
                    tau=tau,
                    delta=0.15,
                    device=device,
                    eta=cfg["eta"],
                )
            else:
                _, traj_list = ddim_sample(
                    model=model,
                    alpha_bars=schedule,
                    shape=traj_shape,
                    device=device,
                    eta=cfg["eta"],
                    num_train_timesteps=1000,
                    return_trajectory=True,
                    use_fp16=False,
                )
                traj_samples = torch.stack(traj_list, dim=0).numpy()  # (steps, n_traj, 2)

            all_trajectories[(method_name, nfe)] = traj_samples

            # --- sample full batch for metrics ---------------------------
            set_seed(args.seed)
            collected = []
            remaining = n_samples
            while remaining > 0:
                bs = min(batch_size, remaining)
                sample_shape = (bs, 2)

                if method_name == "adaptive":
                    out = adaptive_sample(
                        model=model,
                        base_alpha_bars=schedule,
                        shape=sample_shape,
                        tau=tau,
                        delta=0.15,
                        device=device,
                        eta=cfg["eta"],
                        num_train_timesteps=1000,
                        use_fp16=False,
                    )
                else:
                    out = ddim_sample(
                        model=model,
                        alpha_bars=schedule,
                        shape=sample_shape,
                        device=device,
                        eta=cfg["eta"],
                        num_train_timesteps=1000,
                        return_trajectory=False,
                        use_fp16=False,
                    )

                collected.append(out.cpu().numpy())
                remaining -= bs

            samples = np.concatenate(collected, axis=0)[:n_samples]

            # --- metrics -------------------------------------------------
            modes = count_modes(samples, centers, stds)
            w2 = wasserstein2(samples, ref_samples)
            elapsed = time.time() - t0

            rows.append({
                "method": method_name,
                "nfe": nfe,
                "modes_covered": modes,
                "w2_distance": w2,
                "n_samples": n_samples,
                "time_s": round(elapsed, 2),
            })

            print(f"  modes={modes}/8   W2={w2:.4f}   time={elapsed:.1f}s")

    # ── Save results ───────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "exp1_toy2d.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save trajectories
    traj_path = os.path.join(tables_dir, "exp1_trajectories.npz")
    traj_dict = {}
    for (m, n), arr in all_trajectories.items():
        key = f"{m}_nfe{n}"
        traj_dict[key] = arr
    np.savez(traj_path, **traj_dict)
    print(f"Trajectories saved to {traj_path}")

    # ── Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — 2D TOY RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'NFE':>4} {'Modes':>6} {'W2':>10} {'Time(s)':>8}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['method']:<25} {r['nfe']:>4} {r['modes_covered']:>6}/8 "
              f"{r['w2_distance']:>10.4f} {r['time_s']:>8.1f}")
    print("=" * 70)

    # ── Cleanup ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


# ── CPU-safe calibration (avoids torch.cuda.amp on CPU) ───────────────────

@torch.no_grad()
def _calibrate_cpu(
    model,
    alpha_bars: np.ndarray,
    n_calibration: int = 50,
    device: str = "cpu",
    num_train_timesteps: int = 1000,
    percentile: float = 80.0,
) -> float:
    """Lightweight calibration that works on CPU (no autocast)."""
    ab = torch.tensor(alpha_bars, dtype=torch.float32, device=device)
    all_deltas = []

    for _ in range(n_calibration):
        x = torch.randn(1, 2, device=device)
        prev_x0 = None

        for i in range(len(ab) - 1):
            a_c = ab[i]
            ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
            ts = ts.unsqueeze(0)

            eps = model(x, ts)
            x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)

            if prev_x0 is not None:
                delta = (x0 - prev_x0).pow(2).mean().sqrt().item()
                all_deltas.append(delta)

            prev_x0 = x0.clone()

            a_n = ab[i + 1]
            eps_recon = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps_recon

    tau = float(np.percentile(all_deltas, percentile))
    print(f"Calibrated tau = {tau:.4f} (percentile {percentile}, "
          f"from {len(all_deltas)} measurements, CPU)")
    return tau


# ── Adaptive sampling with trajectory recording ──────────────────────────

@torch.no_grad()
def _sample_adaptive_with_traj(
    model,
    base_schedule: np.ndarray,
    shape: tuple,
    tau: float,
    delta: float = 0.15,
    device: str = "cuda",
    eta: float = 0.0,
    num_train_timesteps: int = 1000,
    max_reheats: int = 15,
) -> np.ndarray:
    """
    Run adaptive_sample while also recording the full trajectory.

    Returns:
        np.ndarray of shape (n_steps, batch, 2).
    """
    schedule = list(base_schedule)
    device_t = torch.device(device)
    x = torch.randn(shape, device=device_t)
    prev_x0 = None
    nfe = 0
    n_reheats = 0
    max_nfe = len(base_schedule) + max_reheats
    trajectory = [x.cpu().clone().numpy()]

    i = 0
    while i < len(schedule) - 1 and nfe < max_nfe:
        a_c = torch.tensor(schedule[i], dtype=torch.float32, device=device_t)
        ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        ts = ts.expand(shape[0])

        eps = model(x, ts)
        x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)
        eps = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()
        nfe += 1

        if prev_x0 is not None:
            delta_pred = (x0 - prev_x0).pow(2).mean().sqrt().item()
            if (delta_pred > tau
                    and n_reheats < max_reheats
                    and i < len(schedule) - 5):
                a_reheat = float(a_c.item()) * (1 - delta)
                a_reheat = max(a_reheat, 0.001)
                schedule.insert(i + 1, a_reheat)
                n_reheats += 1

        prev_x0 = x0.clone()

        a_n = torch.tensor(schedule[i + 1], dtype=torch.float32, device=device_t)
        is_denoising = a_n > a_c
        if eta > 0 and is_denoising:
            var = (1 - a_n) / (1 - a_c) * (1 - a_c / a_n)
            sigma = eta * var.clamp(min=0).sqrt()
            direction = (1 - a_n - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_n.sqrt() * x0 + direction + sigma * torch.randn_like(x)
        else:
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps

        trajectory.append(x.cpu().clone().numpy())
        i += 1

    return np.array(trajectory)  # (steps, batch, 2)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 1: 2D Toy Proof of Concept — reheating diffusion"
    )
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of 2D samples to generate per config.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for sampling.")
    parser.add_argument("--nfe", type=str, default="10,25,50",
                        help="Comma-separated list of NFE values.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--sanity", action="store_true", default=True,
                        help="Sanity mode: 256 samples, NFE=10,25 only.")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Full mode: n_samples samples at all NFE values.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Sanity overrides (--full takes precedence)
    if args.full:
        args.sanity = False
    if args.sanity:
        args.n_samples = 256
        args.nfe = "10,25"
        print("[SANITY MODE] n_samples=256, nfe=10,25")
    else:
        print(f"[FULL MODE] n_samples={args.n_samples}, nfe={args.nfe}")

    run_experiment(args)


if __name__ == "__main__":
    main()
