# ImgCompare

Batch image quality metrics (PSNR / SSIM / LPIPS) for EXR and PNG images.  
Supports both a CLI mode (config-driven, CSV export) and a GUI (drag-and-drop, real-time table).

Metric conventions follow the [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) evaluation script:
- **SSIM** — 11×11 Gaussian kernel (`gaussian_weights=True`, σ = 1.5)
- **LPIPS** — `net='vgg'`, input normalised to \[0, 1\] (no gamma conversion)
- **PSNR** — computed over the native pixel value range (`data_range = ref.max()`)

Results are therefore directly comparable with numbers reported in NeRF / 3DGS papers.

---

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA) — install separately from [pytorch.org](https://pytorch.org/get-started/locally/) if GPU support is needed

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **EXR support**: `opencv-python-headless` is used instead of `opencv-python`.  
> The `OPENCV_IO_ENABLE_OPENEXR=1` environment variable is set automatically by the scripts.

---

## Usage

### CLI

```bash
python compare.py --config config.yaml --output results.csv
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to YAML config file |
| `--output` | *(none)* | Optional CSV output path |
| `--workers` | `0` (auto) | Thread count; 0 = `cpu_count // 2` |

### GUI

```bash
python app.py
```

The GUI lets you add reference / target pairs interactively. Results update in real time and can be exported to CSV.

---

## Config Format

```yaml
comparisons:
  - ref: D:/renders/ref/scene1.exr
    targets:
      - D:/renders/methodA/scene1.exr
      - D:/renders/methodB/scene1.exr

  - ref: D:/renders/ref/scene2.png
    targets:
      - D:/renders/methodA/scene2.png
```

Both EXR and PNG (and any OpenCV-readable format) are supported. Images in each pair must have the same resolution and channel count.

---

## Output

```
ref               target              PSNR    SSIM    LPIPS
scene1.exr        methodA/scene1.exr  32.41  0.9123  0.0871
scene1.exr        methodB/scene1.exr  29.87  0.8804  0.1143
```

CSV export preserves full file paths alongside metric values.
