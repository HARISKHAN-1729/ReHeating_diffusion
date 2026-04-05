"""
Experiment 8: Adaptive vs fixed schedule comparison.
Calibrate threshold tau at various percentiles, measure FID and reheat positions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Exp 8: Adaptive vs Fixed")
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--nfe', type=str, default='25')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity', action='store_true', default=True)
    parser.add_argument('--full', action='store_true', default=False)
    args = parser.parse_args()

    if args.full:
        args.sanity = False
        args.n_samples = args.n_samples if args.n_samples != 256 else 50000

    nfe = int(args.nfe.split(',')[0])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    from models.ddpm_wrapper import load_ddpm
    from schedules import make_schedule
    from schedules.adaptive import calibrate_threshold
    from samplers.ddim_reheating import ddim_sample, adaptive_sample
    from eval.fid import save_images_for_fid, compute_fid

    device = args.device
    model = load_ddpm(device=device)

    base_alpha_bars = make_schedule('monotonic', N=nfe)

    # Percentiles to test
    percentiles = [60, 80, 95] if args.sanity else [50, 60, 70, 80, 90, 95]
    n_cal = 20 if args.sanity else 100

    # Calibrate at each percentile
    print("Calibrating thresholds...")
    thresholds = {}
    for pct in percentiles:
        tau = calibrate_threshold(
            model, shape=(1, 3, 32, 32), alpha_bars=base_alpha_bars,
            n_calibration=n_cal, percentile=pct, device=device,
        )
        thresholds[pct] = tau

    results = []
    all_reheat_positions = {}
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Run DDIM baseline first
    print(f"\n--- DDIM baseline | NFE={nfe} ---")
    out_dir = f"results/tmp_fid/exp8_ddim_baseline"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for batch_start in tqdm(range(0, args.n_samples, args.batch_size), desc="DDIM baseline"):
        bs = min(args.batch_size, args.n_samples - batch_start)
        torch.manual_seed(args.seed + batch_start)
        images = ddim_sample(
            model, base_alpha_bars, (bs, 3, 32, 32),
            device=device, eta=0.0, use_fp16=True,
        )
        save_images_for_fid(images.float().clamp(-1, 1), out_dir, start_idx=batch_start)
        del images
        torch.cuda.empty_cache()

    try:
        baseline_fid = compute_fid(out_dir, "cifar10", device=device)
    except Exception:
        baseline_fid = float('nan')
    baseline_time = time.time() - t0

    results.append({
        'percentile': 'baseline', 'tau': 0, 'fid': baseline_fid,
        'avg_reheats': 0, 'wall_time_sec': baseline_time,
    })
    print(f"  Baseline FID: {baseline_fid:.2f}")

    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)

    # Run adaptive at each threshold
    for pct in percentiles:
        tau = thresholds[pct]
        print(f"\n--- Adaptive (pct={pct}, tau={tau:.4f}) | NFE={nfe} ---")
        out_dir = f"results/tmp_fid/exp8_adaptive_pct{pct}"
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        total_reheats = 0
        n_stats_samples = 0

        for batch_start in tqdm(range(0, args.n_samples, args.batch_size), desc=f"Adaptive pct={pct}"):
            bs = min(args.batch_size, args.n_samples - batch_start)
            torch.manual_seed(args.seed + batch_start)

            images, stats = adaptive_sample(
                model, base_alpha_bars, (bs, 3, 32, 32),
                tau=tau, delta=0.15, device=device, eta=0.0,
                max_reheats=15, return_stats=True, use_fp16=True,
            )
            save_images_for_fid(images.float().clamp(-1, 1), out_dir, start_idx=batch_start)
            total_reheats += stats.total_reheats
            n_stats_samples += 1

            if pct not in all_reheat_positions:
                all_reheat_positions[pct] = []
            all_reheat_positions[pct].extend(stats.reheat_positions)

            del images
            torch.cuda.empty_cache()

        wall_time = time.time() - t0
        avg_reheats = total_reheats / max(n_stats_samples, 1)

        try:
            fid = compute_fid(out_dir, "cifar10", device=device)
        except Exception:
            fid = float('nan')

        results.append({
            'percentile': pct, 'tau': tau, 'fid': fid,
            'avg_reheats': avg_reheats, 'wall_time_sec': wall_time,
        })
        print(f"  FID: {fid:.2f}, Avg reheats: {avg_reheats:.1f}")

        shutil.rmtree(out_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/tables/exp8_adaptive.csv', index=False)
    print(f"\nResults saved to results/tables/exp8_adaptive.csv")
    print(df.to_string(index=False))

    # Save reheat positions
    np.savez('results/tables/exp8_reheat_positions.npz',
             **{f'pct{k}': np.array(v) for k, v in all_reheat_positions.items()})

    if args.sanity:
        print("\nSANITY CHECK PASSED")


if __name__ == '__main__':
    main()
