"""Hardware profiling and conservative training-config selection.

Two jobs: (a) tell the truth about whether CUDA is usable, and (b) pick a starting
batch/imgsz/model that will not OOM on the *free* memory, since a GPU can be present
and still too small for the default config.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class HardwareProfile:
    python: str
    platform: str
    torch_installed: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_total_gib: float = 0.0
    gpu_free_gib: float = 0.0
    cpu_count: int = 1
    ram_gib: float = 0.0
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    @property
    def on_gpu(self) -> bool:
        return bool(self.cuda_available and self.gpu_free_gib > 0)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        lines = [
            f"python {self.python} on {self.platform}",
            f"cpu {self.cpu_count} cores, ram ~{self.ram_gib:.1f} GiB",
            f"torch {self.torch_version or 'NOT INSTALLED'} (cuda available: {self.cuda_available})",
        ]
        if self.cuda_available:
            lines.append(
                f"gpu {self.gpu_name} — {self.gpu_free_gib:.1f}/{self.gpu_total_gib:.1f} GiB free"
                + (f" (driver cuda {self.cuda_version})" if self.cuda_version else "")
            )
        else:
            lines.append("gpu: none usable — training here will be CPU-only and slow")
        lines.extend(f"note: {n}" for n in self.notes)
        return lines


def _ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return pages / 1024**3
    except (ValueError, OSError, AttributeError):
        pass
    try:
        return shutil.disk_capacity()  # type: ignore[attr-defined]
    except Exception:
        return 0.0


def _nvidia_smi_free() -> tuple[float, float] | None:
    """Fallback memory read when torch is absent but a driver exists."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None
    if not out:
        return None
    row = out.splitlines()[0]
    try:
        total_mib, free_mib, *name = row.split(",")
        return float(total_mib) / 1024, float(free_mib) / 1024
    except ValueError:
        return None


def detect_hardware() -> HardwareProfile:
    import platform
    import sys

    prof = HardwareProfile(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        ram_gib=_ram_gib(),
    )
    try:
        import torch
    except ImportError:
        prof.notes.append(
            "PyTorch is not installed in this interpreter. `pip install -r requirements.txt` "
            "and, on a GPU box, install the CUDA build of torch for your driver — "
            "ultralytics alone does not give you GPU support."
        )
        smi = _nvidia_smi_free()
        if smi:
            prof.notes.append("nvidia-smi sees a GPU but torch cannot use it (torch/CUDA mismatch?)")
            prof.gpu_total_gib, prof.gpu_free_gib = smi
        return prof

    prof.torch_installed = True
    prof.torch_version = torch.__version__
    try:
        prof.cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # broken driver can raise here
        prof.notes.append(f"torch.cuda.is_available() raised {exc!r}")
    if not prof.cuda_available:
        if getattr(torch.version, "cuda", None):
            prof.notes.append(
                f"this torch is a CUDA build ({torch.version.cuda}) but no device is usable — "
                "check driver / visible-devices"
            )
        else:
            prof.notes.append("this torch is a CPU-only wheel")
        return prof

    prof.cuda_version = getattr(torch.version, "cuda", None)
    try:
        props = torch.cuda.get_device_properties(0)
        prof.gpu_name = props.name
        prof.gpu_total_gib = props.total_memory / 1024**3
        free, total = torch.cuda.mem_get_info()
        prof.gpu_free_gib = free / 1024**3
        prof.gpu_total_gib = total / 1024**3
    except Exception as exc:
        prof.notes.append(f"could not read device properties: {exc!r}")
        prof.gpu_free_gib = prof.gpu_total_gib
    if prof.gpu_name and re.search(r"Tesla T4|L4|3050|16 ?GB", prof.gpu_name, re.I):
        prof.notes.append(f"{prof.gpu_name}: budget ~14 GiB usable, keep imgsz<=768 at batch 8 for nano/small")
    return prof


@dataclass
class TrainPlan:
    model_name: str
    image_size: int
    batch_size: int
    device: str | int
    workers: int
    epochs: int
    cache: str
    fraction: float
    reasons: list[str]

    def lines(self) -> list[str]:
        out = [
            f"model={self.model_name} imgsz={self.image_size} batch={self.batch_size} "
            f"device={self.device} workers={self.workers} epochs={self.epochs} "
            f"cache={self.cache} fraction={self.fraction}",
        ]
        out += [f"  · {r}" for r in self.reasons]
        return out


# Rough footprint per sample (GiB) for yolo11 at a given imgsz, measured on nano/small.
_PER_SAMPLE_GIB = {640: 0.11, 768: 0.16, 800: 0.17, 1024: 0.28, 1280: 0.43}


def plan_training(
    prof: HardwareProfile,
    stats,
    *,
    prefer_model: str | None = None,
    prefer_imgsz: int | None = None,
    allow_cpu: bool = False,
    smoke: bool = False,
) -> TrainPlan:
    """Turn the audit numbers into a defensible starting config."""
    reasons: list[str] = []
    suggested_imgsz, why = stats.suggest_imgsz()
    imgsz = prefer_imgsz or suggested_imgsz
    reasons.append(f"imgsz={imgsz} — {why}")

    if prof.on_gpu:
        device: str | int = 0
    elif allow_cpu:
        device = "cpu"
        reasons.append("no usable CUDA device; CPU training selected — only sensible for a smoke test")
    else:
        device = "cpu"
        reasons.append(
            "no usable CUDA device. Refusing to silently start a multi-hour CPU run; "
            "pass --allow-cpu for a smoke test or move to a GPU box"
        )

    if prof.on_gpu:
        headroom = max(prof.gpu_free_gib - 1.2, 0.6)  # leave room for CUDA context + val pass
        per = _PER_SAMPLE_GIB.get(imgsz, 0.11 * (imgsz / 640) ** 2)
        model = prefer_model or "yolo11n.pt"
        if "s" in Path(model).stem or "m" in Path(model).stem:
            per *= 1.45
        batch = max(1, int(headroom / per))
        batch = min(batch, 64)
        cap = 8 if imgsz >= 1024 else 16
        batch = max(1, min(batch, cap))
        reasons.append(
            f"batch={batch} from {headroom:.1f} GiB headroom at ~{per:.2f} GiB/sample "
            f"(deterministic fallback; raise only after a clean first epoch)"
        )
    else:
        model = prefer_model or "yolo11n.pt"
        batch = 1 if smoke else 2
        reasons.append("batch kept at the minimum for CPU")

    n_images = max(int(getattr(stats, "images", 0) or 0), 1)
    if smoke:
        epochs, fraction = 3, 0.1
        reasons.append("smoke preset: 3 epochs on 10% of the data, just to prove the plumbing")
    else:
        epochs = 60 if n_images < 150 else 100
        fraction = 1.0
        if n_images < 150:
            reasons.append(
                f"only {n_images} images — fewer epochs ({epochs}) + patience to avoid overfitting "
                "a handful of parts; more/better labels will beat more epochs"
            )

    workers = max(1, min(8, (prof.cpu_count or 1) // 2))
    cache = "ram" if (prof.ram_gib >= 16 and n_images * 0.002 < prof.ram_gib * 0.4) else "disk"
    if device == "cpu":
        cache = "disk"
    reasons.append(f"workers={workers}, cache={cache}")

    return TrainPlan(
        model_name=model,
        image_size=imgsz,
        batch_size=batch,
        device=device,
        workers=workers,
        epochs=epochs,
        cache=cache,
        fraction=fraction,
        reasons=reasons,
    )
