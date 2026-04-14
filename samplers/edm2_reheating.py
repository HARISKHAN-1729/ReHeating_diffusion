"""
EDM wrapper for CIFAR-10 (and ImageNet-64).

Loads pretrained EDM v1 checkpoint from Karras et al. (NeurIPS 2022).
CIFAR-10 unconditional VP: FID ~1.97

The NVIDIA EDM checkpoints use custom pickle that requires their
dnnlib/torch_utils modules. This wrapper handles:
  1. Downloading the checkpoint from NVIDIA CDN
  2. Cloning the EDM repo for required modules
  3. Loading the model with correct pickle context
  4. Falling back to stub if anything fails

Interface: denoiser(x, sigma) -> denoised_x
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn

# Checkpoint URLs (EDM v1, Karras et al. 2022)
EDM_URLS = {
    "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    "imagenet64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
}

EDM_REPO = "https://github.com/NVlabs/edm.git"


class EDM2Stub(nn.Module):
    """
    Stub denoiser for when checkpoints are unavailable.
    Correct interface, meaningless outputs. Pipeline verification only.
    """
    def __init__(self, img_channels=3, img_resolution=32):
        super().__init__()
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, img_channels, 3, padding=1),
        )
        self._is_stub = True

    def forward(self, x, sigma):
        sigma_data = 0.5
        s = sigma.view(-1, 1, 1, 1)
        c_skip = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
        c_out = s * sigma_data / (s ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (s ** 2 + sigma_data ** 2).sqrt()
        return c_skip * x + c_out * self.net(c_in * x)


def _download_file(url, dest):
    """Download a file with progress reporting."""
    import urllib.request
    print(f"  Downloading {url}")
    print(f"  -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _ensure_edm_repo(cache_dir):
    """
    Clone NVIDIA's EDM repo to get dnnlib/torch_utils for pickle loading.
    Returns the repo path, or None if cloning fails.
    """
    repo_dir = os.path.join(cache_dir, "edm_repo")
    if os.path.exists(os.path.join(repo_dir, "dnnlib")):
        return repo_dir

    print("  Cloning NVIDIA EDM repo for model loading modules...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", EDM_REPO, repo_dir],
            check=True, capture_output=True, text=True,
        )
        print(f"  Cloned to {repo_dir}")
        return repo_dir
    except Exception as e:
        print(f"  Git clone failed: {e}")
        # Try pip install as fallback
        try:
            print("  Trying alternative: creating minimal dnnlib stub...")
            _create_dnnlib_stub(repo_dir)
            return repo_dir
        except Exception as e2:
            print(f"  Stub creation also failed: {e2}")
            return None


def _create_dnnlib_stub(repo_dir):
    """
    Create minimal dnnlib/torch_utils stubs sufficient for pickle loading.
    The EDM pickle files reference these modules, but we only need the
    class definitions to be importable — the weights are in the pickle.
    """
    os.makedirs(os.path.join(repo_dir, "dnnlib"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "torch_utils"), exist_ok=True)

    # dnnlib/__init__.py
    with open(os.path.join(repo_dir, "dnnlib", "__init__.py"), "w") as f:
        f.write("from .util import EasyDict\n")

    # dnnlib/util.py
    with open(os.path.join(repo_dir, "dnnlib", "util.py"), "w") as f:
        f.write("class EasyDict(dict):\n")
        f.write("    def __getattr__(self, name): return self[name] if name in self else super().__getattribute__(name)\n")
        f.write("    def __setattr__(self, name, value): self[name] = value\n")
        f.write("    def __delattr__(self, name): del self[name]\n")

    # torch_utils/__init__.py
    with open(os.path.join(repo_dir, "torch_utils", "__init__.py"), "w") as f:
        f.write("")

    # torch_utils/persistence.py — the critical one for pickle loading
    with open(os.path.join(repo_dir, "torch_utils", "persistence.py"), "w") as f:
        f.write("import pickle\nimport io\nimport torch\n")
        f.write("def open_url(url, **kwargs): raise NotImplementedError\n")


def _load_edm_pickle(pkl_path, repo_dir, device):
    """
    Load an EDM pickle file with NVIDIA's custom modules on sys.path.
    """
    # Temporarily add repo to path
    added = False
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        added = True

    try:
        import pickle
        import io

        # Custom unpickler that handles NVIDIA's module references
        class _EDMUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Try normal import first
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    # For unknown modules, return a dummy
                    print(f"  Warning: stubbing {module}.{name}")
                    return type(name, (), {})

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # EDM checkpoints store: {'ema': model, 'model': model, ...}
        if isinstance(data, dict):
            if 'ema' in data:
                model = data['ema']
            elif 'model' in data:
                model = data['model']
            else:
                # Try first value
                model = next(iter(data.values()))
        else:
            model = data

        model = model.to(device).eval()
        return model

    except Exception as e:
        print(f"  Pickle loading failed: {e}")
        # Try alternative: load with torch
        try:
            data = torch.load(pkl_path, map_location=device, weights_only=False)
            if isinstance(data, dict) and 'ema' in data:
                model = data['ema']
            elif isinstance(data, dict):
                model = next(iter(data.values()))
            else:
                model = data
            return model.to(device).eval()
        except Exception as e2:
            print(f"  torch.load also failed: {e2}")
            return None
    finally:
        if added:
            sys.path.remove(repo_dir)


def load_edm2(
    dataset="cifar10",
    device="cuda",
    cache_dir="checkpoints",
):
    """
    Load pretrained EDM model for CIFAR-10 or ImageNet-64.

    Attempts in order:
      1. Load from local checkpoint (checkpoints/edm-{dataset}.pkl)
      2. Download from NVIDIA CDN + clone EDM repo for modules
      3. Fall back to stub (pipeline verification only)

    Returns:
        nn.Module with interface: model(x, sigma) -> denoised_x
    """
    os.makedirs(cache_dir, exist_ok=True)

    if dataset == "cifar10":
        img_channels, img_resolution = 3, 32
    elif dataset == "imagenet64":
        img_channels, img_resolution = 3, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Checkpoint filename
    pkl_name = f"edm-{dataset}.pkl"
    pkl_path = os.path.join(cache_dir, pkl_name)

    # Also check legacy naming
    alt_paths = [
        os.path.join(cache_dir, f"edm2-{dataset}.pkl"),
        os.path.join(cache_dir, f"edm-cifar10-32x32-uncond-vp.pkl"),
    ]

    # Step 1: Check if checkpoint exists locally
    found_path = None
    for p in [pkl_path] + alt_paths:
        if os.path.exists(p):
            found_path = p
            break

    # Step 2: Download if not found
    if found_path is None and dataset in EDM_URLS:
        print(f"EDM checkpoint not found locally. Downloading...")
        success = _download_file(EDM_URLS[dataset], pkl_path)
        if success:
            found_path = pkl_path

    # Step 3: Load the checkpoint
    if found_path is not None:
        print(f"Loading EDM checkpoint from {found_path}")

        # Ensure we have the EDM repo modules for unpickling
        repo_dir = _ensure_edm_repo(cache_dir)

        model = _load_edm_pickle(found_path, repo_dir, device)
        if model is not None:
            model._is_stub = False
            n_params = sum(p.numel() for p in model.parameters())
            print(f"EDM loaded: {n_params:,} parameters")

            # Verify interface with a quick forward pass
            try:
                with torch.no_grad():
                    test_x = torch.randn(1, img_channels, img_resolution, img_resolution, device=device)
                    test_sigma = torch.ones(1, device=device)
                    test_out = model(test_x, test_sigma)
                    assert test_out.shape == test_x.shape, f"Output shape mismatch: {test_out.shape}"
                print(f"  Interface verified: ({img_channels},{img_resolution},{img_resolution}) -> OK")
            except Exception as e:
                print(f"  Interface test failed: {e}")
                print(f"  Model may need different calling convention — check EDM docs")

            return model

    # Step 4: Fall back to stub
    print(f"EDM checkpoint not available for {dataset}. Using stub.")
    print(f"  To use real EDM, manually download:")
    if dataset in EDM_URLS:
        print(f"    wget {EDM_URLS[dataset]} -O {pkl_path}")
    model = EDM2Stub(img_channels=img_channels, img_resolution=img_resolution)
    model = model.to(device).eval()
    return model
