"""
Experiment 3: CIFAR-10 with pretrained EDM2.

Tests non-monotonic sigma schedules with EDM2's sigma-parameterized sampler.
Supports Euler and Heun (2nd-order) stepping.

Configs:
  - Monotonic Euler (baseline)
  - Monotonic Heun (2nd-order baseline)
  - Single Reheat (Euler)
  - Damped Oscillation (Euler)

Usage:
  python experiments/exp3_cifar_edm2.py --sanity          # 256 samples, NFE=10,25
  python experiments/exp3_cifar_edm2.py --full             # 50K samples, all NFE
  python experiments/exp3_cifar_edm2.py --batch_size 4     # Override batch size
"""

import argparse
import os
import sys
import time
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedules import make_edm2_schedule
from samplers.edm2_reheating import edm2_sample
from models.edm2_wrapper import load_edm2
from eval.fid import save_images_for_fid, compute_fid


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_configs(nfe_list: list) -> list:
    """
    Build experiment configurations.

    Each config is a dict with:
      - name: human-readable identifier
      - family: schedule family name
      - order: 1 (Euler) or 2 (Heun)
      - schedule_kwargs: extra args for make_edm2_schedule
    """
    configs = []

    for nfe in nfe_list:
        # Monotonic Euler baseline
        configs.append({
            'name': f'Monotonic_Euler_NFE{nfe}',
            'family': 'monotonic',
            'nfe': nfe,
            'order': 1,
            'schedule_kwargs': {},
        })

        # Monotonic Heun baseline (2nd order, uses ~2x NFE internally)
        configs.append({
            'name': f'Monotonic_Heun_NFE{nfe}',
            'family': 'monotonic',
            'nfe': nfe,
            'order': 2,
            'schedule_kwargs': {},
        })

        # Single Reheat (Euler)
        configs.append({
            'name': f'SingleReheat_Euler_NFE{nfe}',
            'family': 'single_reheat',
            'nfe': nfe,
            'order': 1,
            'schedule_kwargs': {'t_reheat': 0.4, 'delta': 0.15},
        })

        # Damped Oscillation (Euler)
        configs.append({
            'name': f'DampedOsc_Euler_NFE{nfe}',
            'family': 'damped_oscillation',
            'nfe': nfe,
            'order': 1,
            'schedule_kwargs': {'A': 0.2, 'gamma': 2.5, 'freq': 4.0},
        })

    return configs


def generate_samples(
    denoiser,
    config: dict,
    n_samples: int,
    batch_size: int,
    device: str,
    output_dir: str,
) -> dict:
    """
    Generate samples for a single configuration.

    Generates in batches, saves images to disk incrementally.
    Returns timing and sample count info.
    """
    family = config['family']
    nfe = config['nfe']
    order = config['order']
    schedule_kwargs = config['schedule_kwargs']

    # Build sigma schedule
    sigmas = make_edm2_schedule(family=family, N=nfe, **schedule_kwargs)

    shape = (batch_size, 3, 32, 32)
    os.makedirs(output_dir, exist_ok=True)

    total_generated = 0
    start_time = time.time()

    n_batches = (n_samples + batch_size - 1) // batch_size
    pbar = tqdm(range(n_batches), desc=config['name'], leave=False)

    for batch_idx in pbar:
        current_bs = min(batch_size, n_samples - total_generated)
        if current_bs <= 0:
            break

        current_shape = (current_bs, 3, 32, 32)

        # Generate batch
        samples = edm2_sample(
            denoiser=denoiser,
            sigmas=sigmas,
            shape=current_shape,
            device=device,
            order=order,
            use_fp16=True,
        )

        # Save images to disk
        save_images_for_fid(samples, output_dir, start_idx=total_generated)
        total_generated += current_bs

        # Free GPU memory
        del samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    return {
        'n_samples': total_generated,
        'time_seconds': elapsed,
        'time_per_sample': elapsed / max(total_generated, 1),
    }


def run_experiment(args):
    """Run the full Experiment 3."""
    set_seed(args.seed)

    # Parse NFE values
    nfe_list = [int(x.strip()) for x in args.nfe.split(',')]

    # Determine sample count
    if args.full:
        n_samples = args.n_samples if args.n_samples != 256 else 50000
        print(f"FULL MODE: {n_samples} samples, NFE = {nfe_list}")
    else:
        n_samples = args.n_samples
        print(f"SANITY MODE: {n_samples} samples, NFE = {nfe_list}")

    print(f"Device: {args.device}, Batch size: {args.batch_size}, Seed: {args.seed}")
    print()

    # Load EDM2 model
    print("Loading EDM2 model for CIFAR-10...")
    denoiser = load_edm2(dataset="cifar10", device=args.device)
    is_stub = getattr(denoiser, '_is_stub', False)
    if is_stub:
        print("WARNING: Using EDM2 stub model. Results are for pipeline verification only.")
    print()

    # Build configs
    configs = build_configs(nfe_list)
    print(f"Total configurations to run: {len(configs)}")
    print()

    # Results storage
    results = []
    results_dir = PROJECT_ROOT / 'results'
    tables_dir = results_dir / 'tables'
    images_base_dir = results_dir / 'generated' / 'exp3_cifar_edm2'
    os.makedirs(tables_dir, exist_ok=True)

    # Run each configuration
    for cfg_idx, config in enumerate(configs):
        print(f"[{cfg_idx + 1}/{len(configs)}] Running: {config['name']}")
        set_seed(args.seed)

        output_dir = str(images_base_dir / config['name'])
        gen_info = generate_samples(
            denoiser=denoiser,
            config=config,
            n_samples=n_samples,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=output_dir,
        )

        # Compute FID if we have enough samples
        fid_score = float('nan')
        if args.full and n_samples >= 10000:
            try:
                print(f"  Computing FID...")
                fid_score = compute_fid(
                    output_dir,
                    dataset_name="cifar10",
                    batch_size=64,
                    device=args.device,
                )
                print(f"  FID: {fid_score:.2f}")
            except Exception as e:
                print(f"  FID computation failed: {e}")
                fid_score = float('nan')
        elif not args.full:
            print(f"  Skipping FID in sanity mode (need --full for FID-50K)")

        result = {
            'config': config['name'],
            'family': config['family'],
            'nfe': config['nfe'],
            'order': config['order'],
            'n_samples': gen_info['n_samples'],
            'time_seconds': round(gen_info['time_seconds'], 2),
            'time_per_sample': round(gen_info['time_per_sample'], 4),
            'fid': round(fid_score, 4) if not np.isnan(fid_score) else 'NaN',
            'is_stub': is_stub,
        }
        results.append(result)

        print(f"  Generated {gen_info['n_samples']} samples in {gen_info['time_seconds']:.1f}s "
              f"({gen_info['time_per_sample']:.3f}s/sample)")
        print()

        # Clear GPU memory between configs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results to CSV
    csv_path = tables_dir / 'exp3_cifar_edm2.csv'
    fieldnames = ['config', 'family', 'nfe', 'order', 'n_samples',
                  'time_seconds', 'time_per_sample', 'fid', 'is_stub']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

    # Print summary table
    print()
    print("=" * 90)
    print("EXPERIMENT 3 — CIFAR-10 EDM2 RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<35} {'NFE':>5} {'Order':>6} {'FID':>10} {'Time(s)':>10}")
    print("-" * 90)
    for r in results:
        fid_str = f"{r['fid']:.2f}" if isinstance(r['fid'], float) else r['fid']
        print(f"{r['config']:<35} {r['nfe']:>5} {r['order']:>6} {fid_str:>10} {r['time_seconds']:>10.1f}")
    print("=" * 90)

    # Sanity check validation
    if not args.full:
        all_generated = all(r['n_samples'] == n_samples for r in results)
        no_nans_in_time = all(r['time_seconds'] > 0 for r in results)
        if all_generated and no_nans_in_time:
            print("\nSANITY CHECK PASSED: All configs generated expected samples without errors.")
        else:
            print("\nSANITY CHECK FAILED: Some configs did not complete successfully.")
            if not all_generated:
                print("  - Not all configs generated the expected number of samples")
            if not no_nans_in_time:
                print("  - Some configs had zero or negative timing")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: CIFAR-10 with EDM2 reheating schedules"
    )
    parser.add_argument('--n_samples', type=int, default=256,
                        help='Number of samples to generate per config')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation (max 8 for 6GB VRAM)')
    parser.add_argument('--nfe', type=str, default='10,25',
                        help='Comma-separated NFE values')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sanity', action='store_true', default=True,
                        help='Sanity check mode: 256 samples, NFE=10,25 (default)')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Full mode: 50K samples, NFE=10,25,50,100, compute FID')

    args = parser.parse_args()

    # Full mode overrides
    if args.full:
        args.sanity = False
        if args.nfe == '10,25':
            args.nfe = '10,25,50,100'
        if args.n_samples == 256:
            args.n_samples = 50000

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    run_experiment(args)


if __name__ == '__main__':
    main()
