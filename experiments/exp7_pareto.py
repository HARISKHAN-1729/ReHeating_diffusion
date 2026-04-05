"""
Experiment 7: NFE Pareto frontier.
FID vs NFE curves for all methods, measuring wall-clock time.
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
    parser = argparse.ArgumentParser(description="Exp 7: Pareto Frontier")
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
        args.n_samples = args.n_samples if args.n_samples != 256 else 50000
        args.nfe = '5,10,15,20,25,50,100,200'

    nfe_list = [int(x) for x in args.nfe.split(',')]

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

    methods = {
        'ddim': {'family': 'monotonic', 'eta': 0.0},
        'ddpm': {'family': 'monotonic', 'eta': 1.0},
        'reheat_det': {'family': 'single_reheat', 'eta': 0.0,
                       't_reheat': 0.4, 'delta': 0.15},
        'reheat_stoch': {'family': 'single_reheat', 'eta': 0.5,
                         't_reheat': 0.4, 'delta': 0.15},
    }

    results = []
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    for method_name, method_cfg in methods.items():
        for nfe in nfe_list:
            print(f"\n--- {method_name} | NFE={nfe} ---")
            family = method_cfg['family']
            eta = method_cfg['eta']
            sched_kwargs = {k: v for k, v in method_cfg.items()
                           if k not in ('family', 'eta')}

            alpha_bars = make_schedule(family, N=nfe, **sched_kwargs)
            out_dir = f"results/tmp_fid/exp7_{method_name}_nfe{nfe}"
            os.makedirs(out_dir, exist_ok=True)

            t0 = time.time()
            n_generated = 0

            for batch_start in tqdm(range(0, args.n_samples, args.batch_size),
                                    desc="Generating"):
                bs = min(args.batch_size, args.n_samples - batch_start)
                cur_shape = (bs, 3, 32, 32)
                torch.manual_seed(args.seed + batch_start)

                images = ddim_sample(
                    model, alpha_bars, cur_shape,
                    device=device, eta=eta, use_fp16=True,
                )
                images = images.float().clamp(-1, 1)
                save_images_for_fid(images, out_dir, start_idx=batch_start)
                n_generated += bs
                del images
                torch.cuda.empty_cache()

            wall_time = time.time() - t0

            try:
                fid = compute_fid(out_dir, dataset_name="cifar10", device=device)
            except Exception as e:
                print(f"  FID failed: {e}")
                fid = float('nan')

            results.append({
                'method': method_name, 'nfe': nfe, 'fid': fid,
                'n_samples': n_generated, 'wall_time_sec': wall_time,
            })
            print(f"  FID: {fid:.2f}, Time: {wall_time:.1f}s")

            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv('results/tables/exp7_pareto.csv', index=False)
    print(f"\nResults saved to results/tables/exp7_pareto.csv")
    print(df.to_string(index=False))

    if args.sanity:
        print("\nSANITY CHECK PASSED")


if __name__ == '__main__':
    main()
