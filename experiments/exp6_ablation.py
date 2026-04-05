"""
Experiment 6: Schedule hyperparameter ablation grid.

Sweeps over (t_reheat, delta) for the single-reheat schedule
using a pretrained DDPM on CIFAR-10. Measures FID for each
configuration to identify the optimal reheat parameters.

Usage:
    python experiments/exp6_ablation.py --sanity          # 6 configs, 256 samples
    python experiments/exp6_ablation.py --full             # 42 configs, 50K samples
"""

import argparse
import os
import sys
import time
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from schedules import make_schedule, schedule_report
from samplers.ddim_reheating import ddim_sample
from models.ddpm_wrapper import load_ddpm
from eval.fid import save_images_for_fid, compute_fid


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------
def generate_images(
    model,
    alpha_bars: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: str,
    image_dir: str,
) -> float:
    """
    Generate n_samples images in batches, save to disk, return wall-clock seconds.
    """
    os.makedirs(image_dir, exist_ok=True)
    shape_single = (3, 32, 32)  # CIFAR-10
    generated = 0
    t0 = time.time()

    pbar = tqdm(total=n_samples, desc="Generating", leave=False)
    while generated < n_samples:
        bs = min(batch_size, n_samples - generated)
        shape = (bs, *shape_single)

        images = ddim_sample(
            model=model,
            alpha_bars=alpha_bars,
            shape=shape,
            device=device,
            eta=0.0,
            clip_denoised=True,
            num_train_timesteps=1000,
            use_fp16=True,
        )

        save_images_for_fid(images, image_dir, start_idx=generated)
        generated += bs
        pbar.update(bs)

        # Free GPU memory between batches
        del images
        torch.cuda.empty_cache()

    pbar.close()
    return time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp 6: Schedule hyperparameter ablation")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="Number of images per config")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation (keep <=8 for 6GB VRAM)")
    parser.add_argument("--nfe", type=str, default="25",
                        help="Fixed NFE for ablation (default 25)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sanity", action="store_true", default=True,
                        help="Sanity mode: small grid, 256 samples")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Full mode: full 42-config grid, 50K samples")
    args = parser.parse_args()

    # Full mode overrides sanity defaults
    if args.full:
        args.sanity = False
        if args.n_samples == 256:
            args.n_samples = 50000

    set_seed(args.seed)
    nfe = int(args.nfe)

    # ----- Define ablation grid -----
    if args.full:
        t_reheat_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        delta_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    else:
        t_reheat_values = [0.3, 0.5, 0.7]
        delta_values = [0.1, 0.3]

    total_configs = len(t_reheat_values) * len(delta_values)
    print(f"=== Experiment 6: Ablation Grid ===")
    print(f"  Mode:       {'FULL' if args.full else 'SANITY'}")
    print(f"  NFE:        {nfe}")
    print(f"  Grid:       {len(t_reheat_values)} t_reheat x {len(delta_values)} delta = {total_configs} configs")
    print(f"  Samples:    {args.n_samples} per config")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device}")
    print(f"  Seed:       {args.seed}")
    print()

    # ----- Load model -----
    model = load_ddpm(device=args.device)

    # ----- Also run DDIM baseline for comparison -----
    print("--- Generating DDIM baseline (monotonic, eta=0) ---")
    set_seed(args.seed)
    baseline_dir = os.path.join(PROJECT_ROOT, "results", "tmp_fid", "exp6_baseline")
    baseline_alpha = make_schedule("monotonic", N=nfe)
    baseline_report = schedule_report(baseline_alpha, name="Baseline (monotonic)")
    baseline_time = generate_images(
        model, baseline_alpha, args.n_samples, args.batch_size, args.device, baseline_dir
    )
    print(f"  Generation time: {baseline_time:.1f}s")
    try:
        baseline_fid = compute_fid(baseline_dir, "cifar10", batch_size=64, device=args.device)
    except Exception as e:
        print(f"  FID computation failed: {e}")
        baseline_fid = float("nan")
    print(f"  Baseline FID: {baseline_fid:.2f}")
    shutil.rmtree(baseline_dir, ignore_errors=True)
    torch.cuda.empty_cache()

    # ----- Run ablation grid -----
    results = []
    config_idx = 0
    for t_reheat in t_reheat_values:
        for delta in delta_values:
            config_idx += 1
            config_name = f"t{t_reheat:.1f}_d{delta:.2f}"
            print(f"\n--- Config {config_idx}/{total_configs}: "
                  f"t_reheat={t_reheat}, delta={delta} ---")

            # Reset seed for each config so results are comparable
            set_seed(args.seed)

            # Build schedule
            alpha_bars = make_schedule(
                "single_reheat", N=nfe, t_reheat=t_reheat, delta=delta
            )
            stats = schedule_report(alpha_bars, name=config_name)

            # Generate images
            img_dir = os.path.join(
                PROJECT_ROOT, "results", "tmp_fid", f"exp6_{config_name}"
            )
            wall_time = generate_images(
                model, alpha_bars, args.n_samples, args.batch_size,
                args.device, img_dir
            )
            print(f"  Generation time: {wall_time:.1f}s")

            # Compute FID
            try:
                fid_score = compute_fid(
                    img_dir, "cifar10", batch_size=64, device=args.device
                )
            except Exception as e:
                print(f"  FID computation failed: {e}")
                fid_score = float("nan")

            delta_fid = fid_score - baseline_fid if not (
                np.isnan(fid_score) or np.isnan(baseline_fid)
            ) else float("nan")

            print(f"  FID: {fid_score:.2f}  (delta vs baseline: {delta_fid:+.2f})")

            results.append({
                "t_reheat": t_reheat,
                "delta": delta,
                "fid": fid_score,
                "fid_delta_vs_baseline": delta_fid,
                "n_samples": args.n_samples,
                "nfe": nfe,
                "reheat_steps": stats["reheat_steps"],
                "wall_time_sec": wall_time,
            })

            # Cleanup
            shutil.rmtree(img_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    # ----- Save results -----
    results_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exp6_ablation.csv")

    df = pd.DataFrame(results)
    # Add baseline row
    baseline_row = pd.DataFrame([{
        "t_reheat": 0.0,
        "delta": 0.0,
        "fid": baseline_fid,
        "fid_delta_vs_baseline": 0.0,
        "n_samples": args.n_samples,
        "nfe": nfe,
        "reheat_steps": 0,
        "wall_time_sec": baseline_time,
    }])
    df = pd.concat([baseline_row, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ----- Summary -----
    print("\n=== Ablation Results Summary ===")
    print(f"{'t_reheat':>10} {'delta':>8} {'FID':>8} {'dFID':>8}")
    print("-" * 40)
    print(f"{'baseline':>10} {'---':>8} {baseline_fid:>8.2f} {'0.00':>8}")
    for r in results:
        print(f"{r['t_reheat']:>10.1f} {r['delta']:>8.2f} "
              f"{r['fid']:>8.2f} {r['fid_delta_vs_baseline']:>+8.2f}")

    # Best config
    valid = [r for r in results if not np.isnan(r["fid"])]
    if valid:
        best = min(valid, key=lambda r: r["fid"])
        print(f"\nBest config: t_reheat={best['t_reheat']}, delta={best['delta']}, "
              f"FID={best['fid']:.2f}")

    # ----- Sanity check -----
    passed = True
    if not results:
        print("\nSANITY CHECK FAILED: No results generated.")
        passed = False
    elif all(np.isnan(r["fid"]) for r in results):
        print("\nSANITY CHECK FAILED: All FID values are NaN.")
        passed = False
    else:
        print(f"\nSANITY CHECK PASSED: {len(results)} configs evaluated, "
              f"{sum(1 for r in results if not np.isnan(r['fid']))} valid FID scores.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
