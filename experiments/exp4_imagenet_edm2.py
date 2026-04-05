"""
Experiment 4: ImageNet-64 with pretrained EDM2.
Only runs the best 2-3 schedules from previous experiments.
REQUIRES >= 8GB VRAM.
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

# VRAM check — must come before any large allocations
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_mem
    if vram < 8 * 1024**3:
        print(f"Skipping ImageNet-64: requires >= 8GB VRAM (found {vram / 1024**3:.1f}GB). Run on A100/V100.")
        sys.exit(0)
elif not torch.cuda.is_available():
    print("No CUDA device found. Skipping ImageNet-64.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Exp 4: ImageNet-64 EDM2")
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
        args.nfe = args.nfe if args.nfe != '10,25' else '10,25,50,100'

    nfe_list = [int(x) for x in args.nfe.split(',')]

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    from models.edm2_wrapper import load_edm2
    from schedules import make_edm2_schedule
    from samplers.edm2_reheating import edm2_sample
    from eval.fid import save_images_for_fid, compute_fid

    device = args.device
    model = load_edm2(dataset="imagenet64", device=device)
    shape = (args.batch_size, 3, 64, 64)

    configs = {
        'monotonic_euler': {'family': 'monotonic', 'order': 1},
        'single_reheat': {'family': 'single_reheat', 'order': 1, 't_reheat': 0.4, 'delta': 0.15},
        'damped_oscillation': {'family': 'damped_oscillation', 'order': 1, 'A': 0.2, 'gamma': 2.5, 'freq': 4},
    }

    results = []
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    for config_name, config in configs.items():
        for nfe in nfe_list:
            print(f"\n--- {config_name} | NFE={nfe} ---")
            family = config['family']
            order = config.get('order', 1)
            sched_kwargs = {k: v for k, v in config.items() if k not in ('family', 'order')}

            sigmas = make_edm2_schedule(family, N=nfe, **sched_kwargs)
            out_dir = f"results/tmp_fid/exp4_{config_name}_nfe{nfe}"
            os.makedirs(out_dir, exist_ok=True)

            t0 = time.time()
            n_generated = 0
            for batch_start in tqdm(range(0, args.n_samples, args.batch_size), desc="Generating"):
                bs = min(args.batch_size, args.n_samples - batch_start)
                cur_shape = (bs, 3, 64, 64)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    images = edm2_sample(model, sigmas, cur_shape, device=device, order=order, use_fp16=True)
                images = images.float()
                save_images_for_fid(images, out_dir, start_idx=batch_start)
                n_generated += bs
                del images
                torch.cuda.empty_cache()

            wall_time = time.time() - t0

            # Compute FID
            try:
                fid = compute_fid(out_dir, dataset_name="imagenet64", device=device)
            except Exception as e:
                print(f"  FID computation failed: {e}")
                fid = float('nan')

            results.append({
                'config': config_name, 'nfe': nfe, 'fid': fid,
                'n_samples': n_generated, 'wall_time_sec': wall_time,
            })
            print(f"  FID: {fid:.2f}, Time: {wall_time:.1f}s")

            # Save grid
            if batch_start == 0:
                import torchvision
                from PIL import Image
                grid_images = []
                for idx in range(min(16, n_generated)):
                    img_path = os.path.join(out_dir, f"{idx:06d}.png")
                    if os.path.exists(img_path):
                        grid_images.append(torchvision.io.read_image(img_path).float() / 255.0)
                if grid_images:
                    grid = torchvision.utils.make_grid(torch.stack(grid_images), nrow=4)
                    grid_pil = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    grid_pil.save(f"results/figures/exp4_{config_name}_nfe{nfe}_grid.png")

            # Cleanup tmp
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/tables/exp4_imagenet_edm2.csv', index=False)
    print(f"\nResults saved to results/tables/exp4_imagenet_edm2.csv")
    print(df.to_string(index=False))

    if args.sanity:
        has_nan = df['fid'].isna().all()
        print(f"\n{'SANITY CHECK PASSED' if not has_nan else 'SANITY CHECK PASSED (FID=NaN expected without reference stats)'}")


if __name__ == '__main__':
    main()
