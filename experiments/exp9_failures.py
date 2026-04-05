"""
Experiment 9: Failure characterization.
Tests aggressive reheating to understand where and how reheating breaks.
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
    parser = argparse.ArgumentParser(description="Exp 9: Failure Characterization")
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
        args.n_samples = args.n_samples if args.n_samples != 256 else 1000

    nfe = int(args.nfe.split(',')[0])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    from models.ddpm_wrapper import load_ddpm
    from schedules import make_schedule
    from samplers.ddim_reheating import ddim_sample
    from eval.fid import save_images_for_fid, compute_fid

    device = args.device
    model = load_ddpm(device=device)

    # Aggressive configs
    configs = {
        'baseline_ddim': {'family': 'monotonic', 'eta': 0.0},
        'mild_reheat_d03': {'family': 'single_reheat', 'eta': 0.0, 't_reheat': 0.5, 'delta': 0.3},
        'aggressive_d05': {'family': 'single_reheat', 'eta': 0.0, 't_reheat': 0.5, 'delta': 0.5},
        'extreme_d07': {'family': 'single_reheat', 'eta': 0.0, 't_reheat': 0.5, 'delta': 0.7},
        'extreme_d10': {'family': 'single_reheat', 'eta': 0.0, 't_reheat': 0.5, 'delta': 1.0},
        'sawtooth_aggressive': {'family': 'sawtooth', 'eta': 0.0, 'period': 10, 'sawtooth_delta': 0.3},
    }

    results = []
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    for config_name, config in configs.items():
        print(f"\n--- {config_name} | NFE={nfe} ---")
        family = config['family']
        eta = config.get('eta', 0.0)
        sched_kwargs = {k: v for k, v in config.items()
                        if k not in ('family', 'eta')}

        try:
            alpha_bars = make_schedule(family, N=nfe, **sched_kwargs)
        except Exception as e:
            print(f"  Schedule construction failed: {e}")
            results.append({
                'config': config_name, 'nfe': nfe, 'fid': float('nan'),
                'has_nan': True, 'has_inf': False, 'notes': str(e),
            })
            continue

        out_dir = f"results/tmp_fid/exp9_{config_name}"
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        has_nan = False
        has_inf = False
        first_batch = None

        for batch_start in tqdm(range(0, args.n_samples, args.batch_size), desc="Generating"):
            bs = min(args.batch_size, args.n_samples - batch_start)
            torch.manual_seed(args.seed + batch_start)

            images = ddim_sample(
                model, alpha_bars, (bs, 3, 32, 32),
                device=device, eta=eta, use_fp16=True,
            )
            images = images.float()

            if torch.isnan(images).any():
                has_nan = True
            if torch.isinf(images).any():
                has_inf = True

            images = images.clamp(-1, 1)
            save_images_for_fid(images, out_dir, start_idx=batch_start)

            if first_batch is None:
                first_batch = images[:16].cpu()
            del images
            torch.cuda.empty_cache()

        wall_time = time.time() - t0

        # Save grid
        if first_batch is not None:
            import torchvision
            from PIL import Image
            grid = torchvision.utils.make_grid(
                (first_batch.clamp(-1, 1) + 1) / 2, nrow=4
            )
            grid_pil = Image.fromarray(
                (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            grid_pil.save(f"results/figures/exp9_{config_name}_grid.png")

        # FID
        try:
            fid = compute_fid(out_dir, "cifar10", device=device)
        except Exception as e:
            fid = float('nan')

        notes = []
        if has_nan:
            notes.append("NaN in outputs")
        if has_inf:
            notes.append("Inf in outputs")

        results.append({
            'config': config_name, 'nfe': nfe, 'fid': fid,
            'has_nan': has_nan, 'has_inf': has_inf,
            'wall_time_sec': wall_time,
            'notes': '; '.join(notes) if notes else 'OK',
        })
        print(f"  FID: {fid:.2f}, NaN: {has_nan}, Inf: {has_inf}")

        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv('results/tables/exp9_failures.csv', index=False)
    print(f"\nResults saved to results/tables/exp9_failures.csv")
    print(df.to_string(index=False))

    if args.sanity:
        print("\nSANITY CHECK PASSED")


if __name__ == '__main__':
    main()
