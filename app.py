#!/usr/bin/env python3
"""
ExrCompare GUI — app.py

Launch: python app.py
"""

import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lpips
import pandas as pd
import torch
import yaml

from compare import compute_metrics, load_image


# ─────────────────────────────────────────────────────────────
# Scrollable frame
# ─────────────────────────────────────────────────────────────

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas)
        self.inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_resize(self, event):
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def scroll_to_bottom(self):
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)


# ─────────────────────────────────────────────────────────────
# Group widget (one ref + N targets)
# ─────────────────────────────────────────────────────────────

class GroupWidget(tk.LabelFrame):
    _counter = 0

    def __init__(self, parent, on_delete, **kwargs):
        GroupWidget._counter += 1
        self._id = GroupWidget._counter
        super().__init__(parent, text=f"Group {self._id}", padx=6, pady=4, **kwargs)
        self._on_delete = on_delete
        self._ref_var = tk.StringVar()
        self._target_rows: list[tuple[tk.Frame, tk.StringVar]] = []
        self._build()

    def _build(self):
        # Ref row
        ref_row = tk.Frame(self)
        ref_row.pack(fill=tk.X)
        tk.Label(ref_row, text="Ref:", width=5, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(ref_row, textvariable=self._ref_var).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        tk.Button(ref_row, text="…", width=2, command=self._browse_ref).pack(
            side=tk.LEFT, padx=(2, 0)
        )

        # Target header
        tgt_hdr = tk.Frame(self)
        tgt_hdr.pack(fill=tk.X, pady=(6, 0))
        tk.Label(tgt_hdr, text="Targets:").pack(side=tk.LEFT)
        tk.Button(tgt_hdr, text="+ Add", command=self._browse_targets).pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(
            tgt_hdr, text="Remove Group", fg="red", command=lambda: self._on_delete(self)
        ).pack(side=tk.RIGHT)

        # Target list container
        self._tgt_frame = tk.Frame(self)
        self._tgt_frame.pack(fill=tk.X)

    def _browse_ref(self):
        path = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[
                ("Images", "*.exr *.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._ref_var.set(path)

    def _browse_targets(self):
        paths = filedialog.askopenfilenames(
            title="Select target images",
            filetypes=[
                ("Images", "*.exr *.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        for p in paths:
            self._add_target_row(p)

    def _add_target_row(self, path: str = ""):
        var = tk.StringVar(value=path)
        row = tk.Frame(self._tgt_frame)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="  ╰─", anchor=tk.W, width=4).pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(
            row,
            text="✕",
            width=2,
            command=lambda r=row, v=var: self._remove_target(r, v),
        ).pack(side=tk.LEFT, padx=(2, 0))
        self._target_rows.append((row, var))

    def _remove_target(self, row: tk.Frame, var: tk.StringVar):
        row.destroy()
        self._target_rows = [(r, v) for r, v in self._target_rows if v is not var]

    def get_data(self) -> dict:
        return {
            "ref": self._ref_var.get().strip(),
            "targets": [
                v.get().strip() for _, v in self._target_rows if v.get().strip()
            ],
        }

    def set_data(self, data: dict):
        self._ref_var.set(data.get("ref", ""))
        for t in data.get("targets", []):
            self._add_target_row(t)


# ─────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ExrCompare")
        self.geometry("1100x660")
        self.minsize(800, 500)

        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass

        self._groups: list[GroupWidget] = []
        self._results: list[dict] = []
        self._result_queue: queue.Queue = queue.Queue()
        self._lpips_model: lpips.LPIPS | None = None
        self._lpips_device: torch.device | None = None
        self._lpips_cpu_model: lpips.LPIPS | None = None
        self._running = False

        self._build_ui()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        # ── Left panel ──────────────────────────────────────
        left = tk.Frame(self, width=370, bd=1, relief=tk.SUNKEN)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 3), pady=6)
        left.pack_propagate(False)

        lhdr = tk.Frame(left)
        lhdr.pack(fill=tk.X, padx=4, pady=(6, 2))
        tk.Label(lhdr, text="Comparison Groups", font=("", 10, "bold")).pack(side=tk.LEFT)
        tk.Button(lhdr, text="Save YAML", command=self._save_yaml).pack(side=tk.RIGHT, padx=2)
        tk.Button(lhdr, text="Load YAML", command=self._load_yaml).pack(side=tk.RIGHT, padx=2)

        sep = ttk.Separator(left, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=4, pady=2)

        self._scroll = ScrollableFrame(left)
        self._scroll.pack(fill=tk.BOTH, expand=True, padx=2)

        tk.Button(left, text="+ Add Group", command=self._add_group).pack(pady=6)

        # ── Right panel ─────────────────────────────────────
        right = tk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 6), pady=6)

        tk.Label(right, text="Results", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))

        tree_frame = tk.Frame(right)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("ref", "target", "PSNR", "SSIM", "LPIPS")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        widths = {"ref": 160, "target": 160, "PSNR": 80, "SSIM": 80, "LPIPS": 80}
        for col in cols:
            self._tree.heading(col, text=col, command=lambda c=col: self._sort_column(c))
            anchor = tk.W if col in ("ref", "target") else tk.CENTER
            self._tree.column(col, width=widths[col], anchor=anchor, minwidth=60)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        # ── Bottom bar ───────────────────────────────────────
        sep2 = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep2.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        self._run_btn = tk.Button(
            bottom, text="▶  Run Comparison", command=self._run,
            bg="#4CAF50", fg="white", activebackground="#45a049",
            width=18, relief=tk.FLAT, padx=6,
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._export_btn = tk.Button(
            bottom, text="Export CSV", command=self._export_csv, state=tk.DISABLED
        )
        self._export_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._progress = ttk.Progressbar(bottom, mode="determinate", length=180)
        self._progress.pack(side=tk.RIGHT)

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(bottom, textvariable=self._status_var, anchor=tk.W).pack(
            side=tk.LEFT, padx=4
        )

    # ── Group management ─────────────────────────────────────

    def _add_group(self, data: dict | None = None):
        g = GroupWidget(self._scroll.inner, on_delete=self._remove_group)
        g.pack(fill=tk.X, padx=4, pady=(0, 6))
        if data:
            g.set_data(data)
        self._groups.append(g)
        self._scroll.scroll_to_bottom()

    def _remove_group(self, group: GroupWidget):
        group.destroy()
        self._groups.remove(group)

    # ── YAML load / save ─────────────────────────────────────

    def _load_yaml(self):
        path = filedialog.askopenfilename(
            title="Load YAML config",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YAML:\n{e}")
            return
        for g in list(self._groups):
            g.destroy()
        self._groups.clear()
        GroupWidget._counter = 0
        for entry in config.get("comparisons", []):
            self._add_group(entry)
        self._status_var.set(f"Loaded: {Path(path).name}")

    def _save_yaml(self):
        path = filedialog.asksaveasfilename(
            title="Save YAML config",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        data = {"comparisons": [g.get_data() for g in self._groups]}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        self._status_var.set(f"Saved: {Path(path).name}")

    # ── Export CSV ───────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            return
        path = filedialog.asksaveasfilename(
            title="Export results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        df = pd.DataFrame(self._results, columns=["ref", "target", "ref_path", "target_path", "PSNR", "SSIM", "LPIPS"])
        df.to_csv(path, index=False)
        self._status_var.set(f"Exported → {Path(path).name}")

    # ── Table column sorting ──────────────────────────────────

    def _sort_column(self, col: str):
        items = [(self._tree.set(k, col), k) for k in self._tree.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0]))
        except ValueError:
            items.sort(key=lambda t: t[0])
        for index, (_, k) in enumerate(items):
            self._tree.move(k, "", index)

    # ── Run / worker thread ───────────────────────────────────

    def _run(self):
        if self._running:
            return
        comparisons = [g.get_data() for g in self._groups]
        comparisons = [c for c in comparisons if c["ref"] and c["targets"]]
        if not comparisons:
            messagebox.showwarning(
                "Nothing to run", "Add at least one group with a ref image and targets."
            )
            return

        total = sum(len(c["targets"]) for c in comparisons)
        self._progress.configure(maximum=total, value=0)
        for row in self._tree.get_children():
            self._tree.delete(row)
        self._results.clear()
        self._export_btn.configure(state=tk.DISABLED)
        self._run_btn.configure(state=tk.DISABLED)
        self._running = True
        self._status_var.set("Loading LPIPS model…")

        threading.Thread(
            target=self._worker, args=(comparisons,), daemon=True
        ).start()
        self.after(100, self._poll_queue)

    def _worker(self, comparisons: list[dict]):
        q = self._result_queue
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self._lpips_model is None or self._lpips_device != device:
                self._lpips_model = lpips.LPIPS(net="vgg").to(device)
                self._lpips_model.eval()
                self._lpips_device = device

            lpips_lock = threading.Lock()  # serialise VGG inference: one pass at a time
            cpu_model_lock = threading.Lock()
            gpu_fallback = threading.Event()

            # Workers: capped at 2. LPIPS inference is serialised (lpips_lock),
            # so extra workers only multiply memory by loading many images at once.
            # 2 workers is sufficient to overlap image I/O with computation.
            cpu_count = os.cpu_count() or 1
            workers = 2
            torch.set_num_threads(max(1, cpu_count // 2))

            # Store only paths — images are loaded on demand inside _process_pair
            # to avoid keeping all ref/target arrays in memory simultaneously.
            tasks: list[tuple] = []
            for entry in comparisons:
                ref_path = entry["ref"]
                for tgt_path in entry["targets"]:
                    tasks.append((len(tasks) + 1, ref_path, tgt_path))

            total_tasks = len(tasks)
            q.put(("status", f"Running on {device} | {workers} workers | {total_tasks} pair(s) queued"))
            print(f"[INFO] Running on {device} with {workers} worker(s)", flush=True)
            print(f"[INFO] {total_tasks} task(s) queued", flush=True)

            model = self._lpips_model

            def _get_cpu_model():
                with cpu_model_lock:
                    if self._lpips_cpu_model is None:
                        print("[INFO] Loading CPU LPIPS fallback model", flush=True)
                        self._lpips_cpu_model = lpips.LPIPS(net="vgg").to("cpu")
                        self._lpips_cpu_model.eval()
                return self._lpips_cpu_model

            def _process_pair(task: tuple):
                pair_idx, ref_path, tgt_path = task
                ref_name = Path(ref_path).name
                tgt_name = Path(tgt_path).name

                def _stage(message: str):
                    status = f"[{pair_idx}/{total_tasks}] {message}: {ref_name} -> {tgt_name}"
                    print(status, flush=True)
                    q.put(("status", status))

                pair_start = time.perf_counter()

                _stage("Loading ref")
                ref_load_start = time.perf_counter()
                try:
                    ref_img = load_image(ref_path)
                except Exception as e:
                    msg = f"Ref load failed ({ref_name}): {e}"
                    print(f"[WARN] [{pair_idx}/{total_tasks}] {msg}", file=sys.stderr, flush=True)
                    return ("warn", msg)

                ref_load_elapsed = time.perf_counter() - ref_load_start

                _stage(f"Loading tgt after {ref_load_elapsed:.2f}s")
                tgt_load_start = time.perf_counter()
                try:
                    tgt_img = load_image(tgt_path)
                except Exception as e:
                    msg = f"Target load failed ({tgt_name}): {e}"
                    print(f"[WARN] [{pair_idx}/{total_tasks}] {msg}", file=sys.stderr, flush=True)
                    return ("warn", msg)

                tgt_load_elapsed = time.perf_counter() - tgt_load_start

                if ref_img.shape != tgt_img.shape:
                    msg = f"Shape mismatch - skipped: {tgt_name}"
                    print(f"[WARN] [{pair_idx}/{total_tasks}] {msg}", file=sys.stderr, flush=True)
                    return ("warn", msg)

                run_device = torch.device("cpu") if gpu_fallback.is_set() else device
                run_model = _get_cpu_model() if run_device.type == "cpu" else model
                run_label = "CPU fallback" if run_device.type == "cpu" and device.type == "cuda" else str(run_device)

                _stage(
                    f"Computing metrics on {run_label} after load {ref_load_elapsed + tgt_load_elapsed:.2f}s"
                )
                metrics_start = time.perf_counter()
                try:
                    metrics = compute_metrics(ref_img, tgt_img, run_model, run_device, lpips_lock)
                except torch.OutOfMemoryError as e:
                    if run_device.type != "cuda":
                        msg = f"Metric error ({tgt_name}): {e}"
                        print(f"[ERROR] {msg}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                        return ("warn", msg)

                    gpu_fallback.set()
                    torch.cuda.empty_cache()
                    warn_msg = (
                        f"CUDA OOM on {tgt_name}; retrying this and remaining pairs on CPU"
                    )
                    print(f"[WARN] [{pair_idx}/{total_tasks}] {warn_msg}", file=sys.stderr, flush=True)
                    q.put(("status", warn_msg))
                    _stage("Retrying on CPU")
                    try:
                        metrics = compute_metrics(
                            ref_img,
                            tgt_img,
                            _get_cpu_model(),
                            torch.device("cpu"),
                            lpips_lock,
                        )
                    except Exception as retry_error:
                        msg = f"CPU fallback error ({tgt_name}): {retry_error}"
                        print(
                            f"[ERROR] {msg}\n{traceback.format_exc()}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return ("warn", msg)
                except Exception as e:
                    msg = f"Metric error ({tgt_name}): {e}"
                    print(f"[ERROR] {msg}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                    return ("warn", msg)

                metrics_elapsed = time.perf_counter() - metrics_start
                total_elapsed = time.perf_counter() - pair_start
                del ref_img, tgt_img
                print(
                    f"[OK] [{pair_idx}/{total_tasks}] {ref_name} vs {tgt_name} "
                    f"| metrics {metrics_elapsed:.2f}s | total {total_elapsed:.2f}s",
                    flush=True,
                )
                return ("result", {
                    "ref": Path(ref_path).name,
                    "target": Path(tgt_path).name,
                    "ref_path": ref_path,
                    "target_path": tgt_path,
                    **metrics,
                })

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process_pair, t): t for t in tasks}
                for fut in as_completed(futures):
                    kind, data = fut.result()
                    q.put((kind, data))
                    q.put(("tick", None))
        except Exception as e:
            msg = f"Worker error: {e}"
            print(f"[ERROR] {msg}\n{traceback.format_exc()}", file=sys.stderr)
            q.put(("warn", msg))
        finally:
            q.put(("done", None))

    def _poll_queue(self):
        try:
            while True:
                kind, data = self._result_queue.get_nowait()
                if kind == "result":
                    self._results.append(data)
                    self._tree.insert(
                        "",
                        tk.END,
                        values=(
                            data["ref"],
                            data["target"],
                            f"{data['PSNR']:.2f}",
                            f"{data['SSIM']:.4f}",
                            f"{data['LPIPS']:.4f}",
                        ),
                    )
                elif kind == "tick":
                    self._progress.step(1)
                    done = int(self._progress["value"])
                    total = int(self._progress["maximum"])
                    self._status_var.set(f"Processing… {done}/{total}")
                elif kind == "status":
                    self._status_var.set(data)
                elif kind == "warn":
                    self._status_var.set(f"⚠ {data}")
                elif kind == "done":
                    self._running = False
                    self._run_btn.configure(state=tk.NORMAL)
                    if self._results:
                        self._export_btn.configure(state=tk.NORMAL)
                        self._status_var.set(
                            f"Done — {len(self._results)} pair(s) computed."
                        )
                    else:
                        self._status_var.set("Done — no valid pairs.")
                    return
        except queue.Empty:
            pass
        if self._running:
            self.after(100, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()
