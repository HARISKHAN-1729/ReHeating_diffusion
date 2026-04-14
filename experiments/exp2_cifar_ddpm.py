"""
Experiment 2: CIFAR-10 DDPM with reheating schedules.

Compares monotonic (DDIM/DDPM) baselines against reheating schedules
on google/ddpm-cifar10-32 at various NFE budgets.

Configs:
  1. DDIM baseline (monotonic, eta=0)
  2. DDPM baseline (monotonic, eta=1)
  3. Single Reheat (t_reheat=0.4, delta=0.15, eta=0)
  4. Damped Oscillation (A=0.2, gamma=2.5, freq=4, eta=0)
  5. Adaptive Reheat (tau calibrated at 80th percentile, eta=0)
  6. Reheat + Stochastic (single_reheat, eta=0.5)

Usage:
  python experiments/exp2_cifar_ddpm.py --sanity          # 256 samples, NFE=10,25
  python experiments/exp2_cifar_ddpm.py --full             # 50K samples, NFE=10,25,50,100
  python experiments/exp2_cifar_ddpm.py --n_samples 1024 --nfe 10,25,50
"""

import argparse
import os
import sys
import time
import csv

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from schedules import make_schedule, count_reheat_steps, calibrate_threshold
from samplers.ddim_reheating import ddim_sample, adaptive_sample
from models.ddpm_wrapper import load_ddpm
from eval.fid import save_images_for_fid, compute_fid


# ============================================================================
# Experiment configuration
# ============================================================================

NUM_TRAIN_TIMESTEPS = 1000
IMAGE_SHAPE_SUFFIX = (3, 32, 32)


def get_configs():
    """Return list of experiment configs."""
    configs = [
        {
            'name': 'DDIM (monotonic)',
            'key': 'ddim_monotonic',
            'family': 'monotonic',
            'schedule_kwargs': {},
            'eta': 0.0,
            'use_adaptive': False,
        },
        {
            'name': 'DDPM (eta=1)',
            'key': 'ddpm_eta1',
            'family': 'monotonic',
            'schedule_kwargs': {},
            'eta': 1.0,
            'use_adaptive': False,
        },
        {
            'name': 'Single Reheat',
            'key': 'single_reheat',
            'family': 'single_reheat',
            'schedule_kwargs': {'t_reheat': 0.4, 'delta': 0.15},
            'eta': 0.0,
            'use_adaptive': False,
        },
        {
            'name': 'Damped Oscillation',
            'key': 'damped_oscillation',
            'family': 'damped_oscillation',
            'schedule_kwargs': {'A': 0.2, 'gamma': 2.5, 'freq': 4.0},
            'eta': 0.0,
            'use_adaptive': False,
        },
        {
            'name': 'Adaptive Reheat',
            'key': 'adaptive_reheat',
            'family': 'monotonic',
            'schedule_kwargs': {},
            'eta': 0.0,
            'use_adaptive': True,
        },
        {
            'name': 'Reheat + Stochastic',
            'key': 'reheat_stochastic',
            'family': 'single_reheat',
            'schedule_kwargs': {'t_reheat': 0.4, 'delta': 0.15},
            'eta': 0.5,
            'use_adaptive': False,
        },
    ]
    return configs


# ============================================================================
# Seed setup
# ============================================================================

def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Image grid saving
# ============================================================================

def save_grid(images: torch.Tensor, path: str, nrow: int = 4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = (images.clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(images[:16], nrow=nrow, padding=2, normalize=False)
    grid_np = (grid.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(path)
    print(f"  Saved grid: {path}")


# ============================================================================
# Generation loop
# ============================================================================

def generate_images(
    model,
    timesteps: np.ndarray,
    alphas_cumprod: torch.Tensor,
    n_samples: int,
    batch_size: int,
    device: str,
    eta: float,
    output_dir: str,
    use_adaptive: bool = False,
    tau: float = 0.0,
    delta_jump: int = 50,
):
    """
    Generate n_samples images in batches and save to disk.

    Args:
        model: DDPM model wrapper
        timesteps: integer timestep sequence from make_schedule()
        alphas_cumprod: model's own alphas_cumprod tensor
        n_samples: total images to generate
        batch_size: images per batch
        device: compute device
        eta: stochasticity (0=DDIM, 1=DDPM)
        output_dir: save directory
        use_adaptive: use adaptive sampler
        tau: adaptive threshold
        delta_jump: adaptive reheat jump size (integer timesteps)

    Returns:
        grid_images: first 16 images for visualization
        elapsed: generation time in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    grid_images = None
    idx = 0
    start_time = time.time()

    for batch_start in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - batch_start)
        shape = (current_batch_size, *IMAGE_SHAPE_SUFFIX)

        if use_adaptive:
            samples = adaptive_sample(
                model=model,
                base_timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                shape=shape,
                tau=tau,
                delta_jump=delta_jump,
                device=device,
                eta=eta,
                max_reheats=15,
                use_fp16=True,
            )
        else:
            samples = ddim_sample(
                model=model,
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                shape=shape,
                device=device,
                eta=eta,
                clip_denoised=True,
                use_fp16=True,
            )

        save_images_for_fid(samples, output_dir, start_idx=idx)
        idx += current_batch_size

        if grid_images is None:
            grid_images = samples[:16].cpu().clone()
        elif grid_images.shape[0] < 16:
            needed = 16 - grid_images.shape[0]
            grid_images = torch.cat([grid_images, samples[:needed].cpu().clone()], dim=0)

        del samples
        torch.cuda.empty_cache()

        n_done = min(batch_start + current_batch_size, n_samples)
        if n_done % max(batch_size * 4, 1) == 0 or n_done == n_samples:
            elapsed_so_far = time.time() - start_time
            rate = n_done / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"    Generated {n_done}/{n_samples} images "
                  f"({elapsed_so_far:.1f}s, {rate:.1f} img/s)")

    elapsed = time.time() - start_time
    return grid_images, elapsed


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(args):
    set_seeds(args.seed)
    nfe_values = [int(x.strip()) for x in args.nfe.split(',')]

    results_dir = os.path.join(PROJECT_ROOT, 'results')
    tables_dir = os.path.join(results_dir, 'tables')
    figures_dir = os.path.join(results_dir, 'figures')
    images_base_dir = os.path.join(results_dir, 'exp2_images')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("Experiment 2: CIFAR-10 DDPM — Reheating Schedules")
    print("=" * 70)
    print(f"  n_samples:  {args.n_samples}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  NFE values: {nfe_values}")
    print(f"  device:     {args.device}")
    print(f"  seed:       {args.seed}")
    mode_str = "FULL" if args.full else "SANITY"
    print(f"  mode:       {mode_str}")
    print()

    # Load model — now also gives us alphas_cumprod
    model = load_ddpm(device=args.device)
    alphas_cumprod = model.alphas_cumprod
    print()

    # Calibrate adaptive threshold
    n_calibration = 100 if args.full else 20
    print(f"Calibrating adaptive threshold (n_calibration={n_calibration})...")
    tau = calibrate_threshold(
        model=model,
        alphas_cumprod=alphas_cumprod,
        shape=(1, *IMAGE_SHAPE_SUFFIX),
        N=25,
        n_calibration=n_calibration,
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        percentile=80.0,
        device=args.device,
    )
    print()

    configs = get_configs()
    all_results = []

    for cfg in configs:
        for nfe in nfe_values:
            config_key = f"{cfg['key']}_nfe{nfe}"
            print("-" * 60)
            print(f"Config: {cfg['name']}  |  NFE: {nfe}")
            print("-" * 60)

            set_seeds(args.seed)

            # Build timestep schedule (integers, not alpha_bars)
            timesteps = make_schedule(
                cfg['family'], N=nfe,
                num_train_timesteps=NUM_TRAIN_TIMESTEPS,
                **cfg['schedule_kwargs'],
            )
            n_reheat_steps = count_reheat_steps(timesteps)
            print(f"  Schedule: {cfg['family']}, {len(timesteps)} steps, "
                  f"{n_reheat_steps} reheat")
            print(f"  Timesteps: [{timesteps[0]}...{timesteps[-1]}]")

            output_dir = os.path.join(images_base_dir, config_key)

            grid_images, elapsed = generate_images(
                model=model,
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                device=args.device,
                eta=cfg['eta'],
                output_dir=output_dir,
                use_adaptive=cfg['use_adaptive'],
                tau=tau,
                delta_jump=50,
            )

            print(f"  Generation time: {elapsed:.1f}s")

            grid_path = os.path.join(figures_dir, f"exp2_{config_key}_grid.png")
            save_grid(grid_images, grid_path)

            print("  Computing FID...")
            try:
                fid_score = compute_fid(
                    generated_dir=output_dir,
                    dataset_name="cifar10",
                    batch_size=64,
                    device=args.device,
                )
                print(f"  FID: {fid_score:.2f}")
            except Exception as e:
                print(f"  FID computation failed: {e}")
                fid_score = float('nan')

            result = {
                'config': cfg['name'],
                'config_key': cfg['key'],
                'family': cfg['family'],
                'eta': cfg['eta'],
                'nfe': nfe,
                'n_samples': args.n_samples,
                'fid': fid_score,
                'reheat_steps': n_reheat_steps,
                'gen_time_s': elapsed,
                'use_adaptive': cfg['use_adaptive'],
            }
            all_results.append(result)
            print(f"  Result: FID={fid_score:.2f}, time={elapsed:.1f}s")
            print()
            torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(tables_dir, 'exp2_cifar_ddpm.csv')
    fieldnames = [
        'config', 'config_key', 'family', 'eta', 'nfe',
        'n_samples', 'fid', 'reheat_steps', 'gen_time_s', 'use_adaptive',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to: {csv_path}")

    # Print summary
    print()
    print("=" * 80)
    print("EXPERIMENT 2 RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Config':<25} {'NFE':>4} {'FID':>8} {'Reheat':>7} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        fid_str = f"{r['fid']:.2f}" if not np.isnan(r['fid']) else "N/A"
        print(f"{r['config']:<25} {r['nfe']:>4} {fid_str:>8} "
              f"{r['reheat_steps']:>7} {r['gen_time_s']:>8.1f}")

    # Assessment
    print()
    if args.full:
        baselines = [r for r in all_results if r['config_key'] in ('ddim_monotonic', 'ddpm_eta1')]
        reheating = [r for r in all_results if r['config_key'] not in ('ddim_monotonic', 'ddpm_eta1')]

        for nfe in nfe_values:
            baseline_fids = [
                r['fid'] for r in baselines
                if r['nfe'] == nfe and not np.isnan(r['fid'])
            ]
            reheat_fids = [
                (r['config'], r['fid']) for r in reheating
                if r['nfe'] == nfe and not np.isnan(r['fid'])
            ]
            if baseline_fids and reheat_fids:
                best_baseline = min(baseline_fids)
                best_reheat_name, best_reheat_fid = min(reheat_fids, key=lambda x: x[1])
                improvement = best_baseline - best_reheat_fid
                print(f"  NFE={nfe}: best baseline FID={best_baseline:.2f}, "
                      f"best reheat FID={best_reheat_fid:.2f} ({best_reheat_name}), "
                      f"improvement={improvement:+.2f}")

        print()
        print(">>> FULL EXPERIMENT COMPLETE <<<")
    else:
        has_valid_fid = any(not np.isnan(r['fid']) for r in all_results)
        has_all_generated = all(r['gen_time_s'] > 0 for r in all_results)

        print("SANITY CHECK:")
        print(f"  All configs generated: {has_all_generated}")
        print(f"  Valid FID scores:      {has_valid_fid}")

        # Check DDIM FID is in expected range
        ddim_results = [r for r in all_results
                        if r['config_key'] == 'ddim_monotonic' and not np.isnan(r['fid'])]
        if ddim_results:
            for r in ddim_results:
                expected_ok = r['fid'] < 100 if r['nfe'] >= 25 else r['fid'] < 200
                status = "OK" if expected_ok else "SUSPICIOUS (too high)"
                print(f"  DDIM NFE={r['nfe']}: FID={r['fid']:.2f} — {status}")

        # Check FID decreases with more NFE (basic sanity)
        ddim_by_nfe = sorted(ddim_results, key=lambda r: r['nfe'])
        if len(ddim_by_nfe) >= 2:
            fid_decreasing = ddim_by_nfe[-1]['fid'] < ddim_by_nfe[0]['fid']
            print(f"  DDIM FID decreases with more NFE: {fid_decreasing}")
            if not fid_decreasing:
                print("  ⚠ WARNING: FID should decrease with more steps. "
                      "Check schedule/model alignment.")

        if has_all_generated:
            print()
            print(">>> SANITY CHECK PASSED <<<")
        else:
            print()
            print(">>> SANITY CHECK FAILED <<<")

    return all_results


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 2: CIFAR-10 DDPM with reheating schedules"
    )
    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--nfe', type=str, default='10,25')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity', action='store_true', default=True)
    parser.add_argument('--full', action='store_true', default=False)

    args = parser.parse_args()
    if args.full:
        args.sanity = False
        args.n_samples = 50000
        args.nfe = '10,25,50,100'

    return args


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
