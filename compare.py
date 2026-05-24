#!/usr/bin/env python3
"""
ExrCompare: Compute PSNR, SSIM, and LPIPS between a set of reference images
and their corresponding target images. Metrics are computed on raw HDR linear
values (RGB only).

Usage:
    python compare.py [--config config.yaml] [--output results.csv]
"""

import argparse
import os
import sys
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
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
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
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    ref: np.ndarray,
    tgt: np.ndarray,
    lpips_model: lpips.LPIPS,
    device: torch.device,
) -> dict:
    """Return dict with PSNR, SSIM, LPIPS for a (ref, tgt) pair.

    Follows the convention used in the 3D Gaussian Splatting evaluation script:
    - PSNR / SSIM: computed on the raw pixel value range (data_range = ref.max())
    - SSIM: 11x11 Gaussian kernel (gaussian_weights=True) matching the 3DGS custom SSIM
    - LPIPS: net='vgg', input normalised to [0, 1] and passed directly
      (same convention as the reference; no gamma conversion, no [-1,1] remapping)
    """
    data_range = float(ref.max())
    if data_range == 0.0:
        data_range = 1.0  # avoid divide-by-zero for black reference

    psnr_val = peak_signal_noise_ratio(ref, tgt, data_range=data_range)
    ssim_val = structural_similarity(
        ref, tgt, data_range=data_range,
        gaussian_weights=True, sigma=1.5,  # 11x11 Gaussian kernel, matching 3DGS
        channel_axis=2,
    )

    # LPIPS: normalise to [0, 1] and pass directly — consistent with 3DGS / NeRF community.
    scale = data_range
    ref_t = torch.from_numpy(
        np.clip(ref / scale, 0.0, 1.0).astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(
        np.clip(tgt / scale, 0.0, 1.0).astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_val = lpips_model(ref_t, tgt_t).item()

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

    # Workers and torch threading: keep total threads ≈ CPU count.
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers > 0 else max(1, cpu_count // 2)
    torch_threads = max(1, cpu_count // workers)
    torch.set_num_threads(torch_threads)

    # Initialise LPIPS model (AlexNet, as it's the fastest and most standard)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Workers: {workers}  torch threads/worker: {torch_threads}")
    print("[INFO] Loading LPIPS model (vgg)...")
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    # Build flat list of (ref_img, ref_path, tgt_path) tasks.
    tasks: list[tuple] = []
    for entry in comparisons:
        ref_path = entry.get("ref", "")
        targets = entry.get("targets", [])
        if not ref_path or not targets:
            print(f"[WARN] Skipping incomplete entry: {entry}")
            continue
        try:
            ref_img = load_image(ref_path)
        except FileNotFoundError as e:
            print(f"[WARN] {e} — skipping all targets for this ref.")
            continue
        for tgt_path in targets:
            tasks.append((ref_img, ref_path, tgt_path))

    def _process_pair(task: tuple) -> dict | None:
        ref_img, ref_path, tgt_path = task
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
        metrics = compute_metrics(ref_img, tgt_img, lpips_model, device)
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
