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
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Must be set before cv2 is imported to enable EXR support.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
# Helps large-image CUDA workloads avoid allocator fragmentation where supported.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

CUDA_CLEANUP_ERRORS = (
    RuntimeError,
    torch.OutOfMemoryError,
    getattr(torch, "AcceleratorError", RuntimeError),
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def safe_clear_cuda_cache() -> None:
    """Best-effort CUDA cleanup after metric execution or OOM.

    CUDA can surface OOM asynchronously, so synchronize/empty_cache may raise
    during cleanup and would otherwise hide the original exception.
    """
    if not torch.cuda.is_available():
        return
    with contextlib.suppress(*CUDA_CLEANUP_ERRORS):
        torch.cuda.synchronize()
    with contextlib.suppress(*CUDA_CLEANUP_ERRORS):
        torch.cuda.empty_cache()

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

def compute_metrics(ref, tgt, lpips_model, device, lpips_lock=None, lpips_dtype: torch.dtype = torch.float32):
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

    # Build normalised CPU tensors first. When a shared lock is provided, all
    # device work below becomes single-file so only one pair occupies GPU memory
    # at a time, while another worker can still overlap image I/O.
    scale = data_range
    ref_cpu = torch.from_numpy(
        np.clip(ref / scale, 0.0, 1.0).astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0)
    tgt_cpu = torch.from_numpy(
        np.clip(tgt / scale, 0.0, 1.0).astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0)

    ref_t = None
    tgt_t = None
    _ctx = lpips_lock if lpips_lock is not None else contextlib.nullcontext()

    try:
        with _ctx:
            try:
                ref_t = ref_cpu.to(device)
                tgt_t = tgt_cpu.to(device)

                psnr_val = _psnr_pt(ref_t, tgt_t)
                ssim_val = _ssim_pt(ref_t, tgt_t)
                lpips_ref = ref_t
                lpips_tgt = tgt_t
                if lpips_dtype != ref_t.dtype:
                    lpips_ref = ref_t.to(dtype=lpips_dtype)
                    lpips_tgt = tgt_t.to(dtype=lpips_dtype)
                try:
                    with torch.no_grad():
                        lpips_val = lpips_model(lpips_ref, lpips_tgt).item()
                finally:
                    if lpips_ref is not ref_t:
                        del lpips_ref
                    if lpips_tgt is not tgt_t:
                        del lpips_tgt

                return {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}
            finally:
                if ref_t is not None:
                    del ref_t
                if tgt_t is not None:
                    del tgt_t
                if device.type == "cuda":
                    safe_clear_cuda_cache()
    finally:
        del ref_cpu, tgt_cpu


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS between images.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--output", default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker threads. 0 = safe default (1). Use values >1 only for smaller images."
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

    # Batch CLI defaults to a single worker. LPIPS execution is serialised
    # anyway, and one worker avoids stale GPU fallback state and extra memory
    # pressure on large images.
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers > 0 else 1
    torch_threads = max(1, cpu_count // 2)
    torch.set_num_threads(torch_threads)

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
            tasks.append((len(tasks) + 1, ref_path, tgt_path))

    total_tasks = len(tasks)
    print(f"[INFO] {total_tasks} task(s) queued")

    # Initialise primary LPIPS model after a light image-size preflight so we can
    # start oversized images directly in fp16 mode on GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Workers: {workers}  torch threads/worker: {torch_threads}")

    lpips_lock = threading.Lock()  # serialise VGG inference: one pass at a time
    model_init_lock = threading.Lock()
    gpu_fp16_fallback = threading.Event()
    cpu_fallback = threading.Event()
    lpips_half_model = None
    lpips_cpu_model = None
    prefer_gpu_fp16 = False

    if device.type == "cuda" and tasks:
        sample_ref_path = tasks[0][1]
        try:
            sample_ref = load_image(sample_ref_path)
            pixel_count = sample_ref.shape[0] * sample_ref.shape[1]
            if pixel_count >= 3000 * 3000:
                prefer_gpu_fp16 = True
                print(
                    f"[INFO] Large image mode detected ({sample_ref.shape[1]}x{sample_ref.shape[0]}); "
                    f"starting LPIPS in cuda fp16",
                    flush=True,
                )
            del sample_ref
        except Exception as e:
            print(f"[WARN] LPIPS precision preflight failed for {sample_ref_path}: {e}", flush=True)

    print("[INFO] Loading LPIPS model (vgg)...")
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    if prefer_gpu_fp16:
        lpips_model = lpips_model.half()
        gpu_fp16_fallback.set()
        lpips_half_model = lpips_model
    lpips_model.eval()

    def _get_gpu_half_model():
        nonlocal lpips_half_model, lpips_model, lpips_cpu_model
        with model_init_lock:
            if lpips_half_model is None:
                if lpips_model is not None and device.type == "cuda":
                    print("[INFO] Moving LPIPS fp32 model to CPU before loading cuda fp16 fallback...", flush=True)
                    lpips_cpu_model = lpips_model.to("cpu")
                    lpips_cpu_model.eval()
                    lpips_model = None
                    safe_clear_cuda_cache()
                print("[INFO] Loading LPIPS fallback model (vgg, cuda fp16)...", flush=True)
                lpips_half_model = lpips.LPIPS(net="vgg").to(device).half()
                lpips_half_model.eval()
        return lpips_half_model

    def _get_cpu_model():
        nonlocal lpips_cpu_model, lpips_model
        with model_init_lock:
            if lpips_cpu_model is None:
                if lpips_model is not None and device.type == "cpu":
                    lpips_cpu_model = lpips_model
                else:
                    print("[INFO] Loading LPIPS fallback model (vgg, cpu)...", flush=True)
                    lpips_cpu_model = lpips.LPIPS(net="vgg").to("cpu")
                    lpips_cpu_model.eval()
        return lpips_cpu_model

    def _process_pair(task: tuple) -> dict | None:
        pair_idx, ref_path, tgt_path = task
        ref_name = Path(ref_path).name
        tgt_name = Path(tgt_path).name

        def _stage(message: str):
            print(f"[{pair_idx}/{total_tasks}] {message}: {ref_name} -> {tgt_name}", flush=True)

        pair_start = time.perf_counter()

        _stage("Loading ref")
        ref_load_start = time.perf_counter()
        try:
            ref_img = load_image(ref_path)
        except FileNotFoundError as e:
            print(f"[WARN] [{pair_idx}/{total_tasks}] {e}", flush=True)
            return None
        ref_load_elapsed = time.perf_counter() - ref_load_start

        _stage(f"Loading tgt after {ref_load_elapsed:.2f}s")
        tgt_load_start = time.perf_counter()
        try:
            tgt_img = load_image(tgt_path)
        except FileNotFoundError as e:
            print(f"[WARN] [{pair_idx}/{total_tasks}] {e}", flush=True)
            return None
        tgt_load_elapsed = time.perf_counter() - tgt_load_start

        if ref_img.shape != tgt_img.shape:
            print(
                f"[WARN] [{pair_idx}/{total_tasks}] Shape mismatch: ref {ref_img.shape} "
                f"vs tgt {tgt_img.shape} ({tgt_path}) - skipping.",
                flush=True,
            )
            return None

        def _select_run_mode():
            if cpu_fallback.is_set():
                return _get_cpu_model(), torch.device("cpu"), torch.float32, "cpu"
            if device.type == "cuda" and gpu_fp16_fallback.is_set():
                return _get_gpu_half_model(), device, torch.float16, "cuda fp16"
            return lpips_model, device, torch.float32, str(device)

        run_model, run_device, lpips_dtype, run_label = _select_run_mode()
        _stage(
            f"Computing metrics on {run_label} after load {ref_load_elapsed + tgt_load_elapsed:.2f}s"
        )
        metrics_start = time.perf_counter()

        try:
            metrics = compute_metrics(
                ref_img,
                tgt_img,
                run_model,
                run_device,
                lpips_lock,
                lpips_dtype=lpips_dtype,
            )
        except torch.OutOfMemoryError as e:
            if run_device.type != "cuda":
                print(
                    f"[ERROR] [{pair_idx}/{total_tasks}] Metric error ({tgt_name}): {e}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

            if lpips_dtype == torch.float32:
                gpu_fp16_fallback.set()
                safe_clear_cuda_cache()
                print(
                    f"[WARN] [{pair_idx}/{total_tasks}] CUDA OOM on {tgt_name}; "
                    f"retrying this and remaining pairs with LPIPS fp16 on GPU",
                    file=sys.stderr,
                    flush=True,
                )
                _stage("Retrying with LPIPS fp16")
                try:
                    metrics = compute_metrics(
                        ref_img,
                        tgt_img,
                        _get_gpu_half_model(),
                        device,
                        lpips_lock,
                        lpips_dtype=torch.float16,
                    )
                except torch.OutOfMemoryError:
                    cpu_fallback.set()
                    safe_clear_cuda_cache()
                    print(
                        f"[WARN] [{pair_idx}/{total_tasks}] CUDA OOM persists on {tgt_name}; "
                        f"retrying this and remaining pairs on CPU",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stage("Retrying on CPU")
                    try:
                        metrics = compute_metrics(
                            ref_img,
                            tgt_img,
                            _get_cpu_model(),
                            torch.device("cpu"),
                            lpips_lock,
                        )
                    except Exception as cpu_retry_error:
                        print(
                            f"[ERROR] [{pair_idx}/{total_tasks}] CPU fallback error ({tgt_name}): {cpu_retry_error}\n"
                            f"{traceback.format_exc()}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return None
            else:
                cpu_fallback.set()
                safe_clear_cuda_cache()
                print(
                    f"[WARN] [{pair_idx}/{total_tasks}] CUDA OOM on {tgt_name}; "
                    f"retrying this and remaining pairs on CPU",
                    file=sys.stderr,
                    flush=True,
                )
                _stage("Retrying on CPU")
                try:
                    metrics = compute_metrics(
                        ref_img,
                        tgt_img,
                        _get_cpu_model(),
                        torch.device("cpu"),
                        lpips_lock,
                    )
                except Exception as cpu_retry_error:
                    print(
                        f"[ERROR] [{pair_idx}/{total_tasks}] CPU fallback error ({tgt_name}): {cpu_retry_error}\n"
                        f"{traceback.format_exc()}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return None
        except Exception as e:
            print(
                f"[ERROR] [{pair_idx}/{total_tasks}] Metric error ({tgt_name}): {e}\n"
                f"{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            return None

        metrics_elapsed = time.perf_counter() - metrics_start
        total_elapsed = time.perf_counter() - pair_start
        print(
            f"[OK] [{pair_idx}/{total_tasks}] {ref_name} vs {tgt_name} "
            f"| metrics {metrics_elapsed:.2f}s | total {total_elapsed:.2f}s "
            f"| PSNR={metrics['PSNR']:.2f} SSIM={metrics['SSIM']:.4f} LPIPS={metrics['LPIPS']:.4f}",
            flush=True,
        )
        return {
            "ref": ref_name,
            "target": tgt_name,
            "ref_path": ref_path,
            "target_path": tgt_path,
            **metrics,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_pair, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                pair_idx, ref_path, tgt_path = futures[fut]
                print(
                    f"[ERROR] [{pair_idx}/{total_tasks}] Unexpected worker failure for "
                    f"{Path(ref_path).name} -> {Path(tgt_path).name}: {e}\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
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
