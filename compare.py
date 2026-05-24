#!/usr/bin/env python3
"""
ExrCompare: Compute PSNR, SSIM, and LPIPS between a set of reference images
and their corresponding target images. Metrics are computed on raw HDR linear
values (RGB only).

Usage:
    python compare.py [--config config.yaml] [--output results.csv]
"""

import argparse
import contextlib
import os
import sys
import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Must be set before cv2 is imported to enable EXR support.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tabulate import tabulate

# Suppress torchvision deprecation warnings originating from lpips internals.
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image as float32 RGB [H, W, 3] in its native HDR range.

    Supports EXR and any format OpenCV can read with ANYDEPTH.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # Ensure float32
    img = img.astype(np.float32)

    if img.ndim == 2:
        # Grayscale → replicate to 3 channels
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        # Drop alpha
        img = img[:, :, :3]
    elif img.shape[2] != 3:
        raise ValueError(f"Unexpected channel count {img.shape[2]} in: {path}")

    # OpenCV loads BGR → convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



# ---------------------------------------------------------------------------
# Metric helpers — PyTorch implementations matching the 3DGS reference
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    return g / g.sum()


@torch.no_grad()
def _psnr_pt(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR matching 3DGS image_utils.py. Input: [0,1] NCHW float32."""
    mse = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    if mse.item() == 0.0:
        return float("inf")
    return (20.0 * torch.log10(1.0 / torch.sqrt(mse))).item()


@torch.no_grad()
def _ssim_pt(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """SSIM matching 3DGS loss_utils.py conv2d implementation. Input: [0,1] NCHW float32."""
    ch = img1.shape[1]
    k = _gaussian_kernel(window_size).to(img1.device)
    win = k.unsqueeze(1).mm(k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    win = win.expand(ch, 1, window_size, window_size).type_as(img1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, win, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, win, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    s1 = F.conv2d(img1 * img1, win, padding=pad, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, win, padding=pad, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, win, padding=pad, groups=ch) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(ref, tgt, lpips_model, device, lpips_lock=None):
    """Return dict with PSNR, SSIM, LPIPS for a (ref, tgt) pair.

    All three metrics use the same normalised [0,1] tensor, so GPU acceleration
    applies to PSNR and SSIM as well. Matches the 3DGS evaluation convention:
    - PSNR: 20*log10(1/sqrt(MSE)) on [0,1] input  (image_utils.py)
    - SSIM: 11x11 Gaussian conv2d, C1=0.01^2, C2=0.03^2  (loss_utils.py)
    - LPIPS: net='vgg', [0,1] input, no normalize flag  (metrics.py)

    lpips_lock: pass a shared threading.Lock when running inside a thread pool
    to serialise VGG forward passes and cap peak memory usage.
    """
    data_range = float(ref.max())
    if data_range == 0.0:
        data_range = 1.0

    # Build shared normalised tensors once — reused for all three metrics.
    scale = data_range
    ref_t = torch.from_numpy(
        np.clip(ref / scale, 0.0, 1.0).astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(
        np.clip(tgt / scale, 0.0, 1.0).astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    # PSNR and SSIM — GPU-accelerated when device is CUDA, no lock needed.
    psnr_val = _psnr_pt(ref_t, tgt_t)
    ssim_val = _ssim_pt(ref_t, tgt_t)

    # LPIPS — serialised via lock to cap peak activation memory.
    _ctx = lpips_lock if lpips_lock is not None else contextlib.nullcontext()
    with _ctx:
        with torch.no_grad():
            lpips_val = lpips_model(ref_t, tgt_t).item()

    del ref_t, tgt_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS between images.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--output", default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker threads. 0 = auto (half of CPU count). Use 1 to disable."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    comparisons = config.get("comparisons", [])
    if not comparisons:
        print("[ERROR] No comparisons defined in config.", file=sys.stderr)
        sys.exit(1)

    # Workers: capped at 2. LPIPS inference is serialised (lpips_lock),
    # so extra workers only multiply memory by loading many images at once.
    # 2 workers is sufficient to overlap image I/O with computation.
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers > 0 else 2
    torch_threads = max(1, cpu_count // 2)
    torch.set_num_threads(torch_threads)

    # Initialise LPIPS model (AlexNet, as it's the fastest and most standard)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Workers: {workers}  torch threads/worker: {torch_threads}")
    print("[INFO] Loading LPIPS model (vgg)...")
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()
    lpips_lock = threading.Lock()  # serialise VGG inference: one pass at a time

    # Build flat list of (ref_path, tgt_path) tasks — images are loaded on demand
    # inside the worker to avoid keeping all refs in memory simultaneously.
    tasks: list[tuple] = []
    for entry in comparisons:
        ref_path = entry.get("ref", "")
        targets = entry.get("targets", [])
        if not ref_path or not targets:
            print(f"[WARN] Skipping incomplete entry: {entry}")
            continue
        for tgt_path in targets:
            tasks.append((ref_path, tgt_path))

    def _process_pair(task: tuple) -> dict | None:
        ref_path, tgt_path = task
        try:
            ref_img = load_image(ref_path)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            return None
        try:
            tgt_img = load_image(tgt_path)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            return None
        if ref_img.shape != tgt_img.shape:
            print(
                f"[WARN] Shape mismatch: ref {ref_img.shape} vs tgt {tgt_img.shape} "
                f"({tgt_path}) — skipping."
            )
            return None
        metrics = compute_metrics(ref_img, tgt_img, lpips_model, device, lpips_lock)
        print(
            f"  {Path(ref_path).name} vs {Path(tgt_path).name}  "
            f"PSNR={metrics['PSNR']:.2f}  "
            f"SSIM={metrics['SSIM']:.4f}  "
            f"LPIPS={metrics['LPIPS']:.4f}"
        )
        return {
            "ref": Path(ref_path).name,
            "target": Path(tgt_path).name,
            "ref_path": ref_path,
            "target_path": tgt_path,
            **metrics,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_pair, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                rows.append(result)

    if not rows:
        print("[ERROR] No valid comparisons were computed.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["ref", "target", "ref_path", "target_path", "PSNR", "SSIM", "LPIPS"])

    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Results saved to: {args.output}")

    # Print summary table (short names only)
    print()
    print(tabulate(
        df[["ref", "target", "PSNR", "SSIM", "LPIPS"]].values.tolist(),
        headers=["ref", "target", "PSNR", "SSIM", "LPIPS"],
        floatfmt=("", "", ".2f", ".4f", ".4f"),
        tablefmt="simple",
    ))


if __name__ == "__main__":
    main()
