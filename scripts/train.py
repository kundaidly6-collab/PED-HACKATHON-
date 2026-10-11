#!/usr/bin/env python3
"""Phase 3 — hardware-adaptive training.

Defaults come from the audit (target size, image count, free VRAM), not from a
copy-pasted config. Anything you pass on the command line wins.

    python scripts/train.py --dry-run          # print the chosen config, train nothing
    python scripts/train.py                    # baseline
    python scripts/train.py --smoke            # prove the plumbing in ~1 epoch on 10% of data
    python scripts/train.py --model yolo11s.pt --imgsz 1024 --name ped_1024
    python scripts/train.py --resume runs/ped_baseline/weights/last.pt

Every run writes runs/<name>/config.json so a comparison later means something.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ped.discovery import build_layout, iter_images  # noqa: E402
from _ped.hardware import detect_hardware, plan_training  # noqa: E402
from _ped.labels import summarize  # noqa: E402
from _ped.split import choose_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PREPARED = ROOT / "configs" / "data.prepared.yaml"
DATA_FALLBACK = ROOT / "configs" / "data.yaml"


def resolve_data_yaml(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"[ERROR] --data {p} not found")
        return p
    if DATA_PREPARED.is_file():
        return DATA_PREPARED
    if DATA_FALLBACK.is_file():
        print(
            f"note: using configs/data.yaml. If it contains a relative 'path:',\n"
            f"      Ultralytics resolves it against the global datasets_dir setting,\n"
            f"      NOT against configs/ — run prepare_dataset.py to get absolute paths."
        )
        return DATA_FALLBACK
    raise SystemExit(
        "[ERROR] no dataset config. Run:  python scripts/prepare_dataset.py\n"
        "        (it writes configs/data.prepared.yaml with absolute paths)"
    )


def resolve_split_dirs(data_yaml: Path) -> tuple[Path | None, Path | None, list[str]]:
    """Read the dataset YAML and find (train_images, train_labels, notes).

    A relative 'path:' is ambiguous — Ultralytics resolves it against its global
    datasets_dir, which we cannot see from here — so we try the two plausible readings
    and say which one was used. Absolute paths (what prepare_dataset.py writes) just work.
    """
    import yaml as _yaml

    notes: list[str] = []
    try:
        cfg = _yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return None, None, [f"could not parse {data_yaml}: {exc}"]
    train_rel = cfg.get("train") or "images/train"
    base = cfg.get("path")
    if base is None:
        img_dir = (data_yaml.parent / train_rel).resolve()
        notes.append(f"no 'path:' in the yaml; assuming images/{train_rel} next to the yaml")
    else:
        base_p = Path(str(base)).expanduser()
        if base_p.is_absolute():
            img_dir = (base_p / train_rel).resolve()
        else:
            for cand in (data_yaml.parent / base_p, ROOT / base_p):
                if cand.exists():
                    img_dir = (cand / train_rel).resolve()
                    notes.append(f"relative 'path:' resolved to {img_dir} (tried relative to the yaml first)")
                    break
            else:
                img_dir = (data_yaml.parent / base_p / train_rel).resolve()
                notes.append(
                    f"neither {data_yaml.parent / base_p} nor {ROOT / base_p} exists — "
                    "if the trainer also cannot find the data, rewrite 'path:' as an absolute path"
                )
    if not img_dir.is_dir():
        return None, None, notes + [f"image directory does not exist: {img_dir}"]
    # Standard ultralytics layout: <root>/images/<split> with <root>/labels/<split>.
    # Check the plausible variants instead of assuming, because a wrong labels path
    # silently reports "0 boxes" and then the imgsz planning is blind.
    split = img_dir.name
    candidates = [
        img_dir.parent.parent / "labels" / split,   # dataset/images/train -> dataset/labels/train
        img_dir.parent / "labels" / split,          # dataset/train/images -> dataset/train/labels
        img_dir.parent / "labels",                  # labels/ holds the split name already
    ]
    if "images" in img_dir.parts:                   # replace the last 'images' component
        parts = list(img_dir.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                candidates.insert(0, Path(*parts))
                break
    lbl_dir = next((c for c in candidates if c.is_dir()), None)
    if lbl_dir is None:
        notes.append(f"no labels directory found near {img_dir} (tried {[str(c) for c in candidates[:3]]})")
    return img_dir, lbl_dir, notes


def dataset_stats(data_yaml: Path) -> tuple[object, dict]:
    """Stats for the actual training split named by *data_yaml*.

    This used to hardcode dataset/prepared, which silently reported "no images" for any
    dataset configured elsewhere. Read what the trainer will read.
    """
    from PIL import Image

    from _ped.labels import DatasetStats, read_yolo_file

    img_dir, lbl_dir, notes = resolve_split_dirs(data_yaml)
    for n in notes:
        print(f"  · {n}")
    if img_dir is not None:
        records, findings = [], []
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            try:
                with Image.open(img) as im:
                    w, h = im.size
            except Exception:
                continue
            lbl = (lbl_dir / f"{img.stem}.txt") if lbl_dir else None
            boxes, f = (read_yolo_file(lbl) if lbl and lbl.is_file() else ([], []))
            findings.extend(f)
            records.append((img, lbl if lbl and lbl.is_file() else None, w, h, boxes))
        if records:
            return summarize(records, findings), {"source": str(img_dir)}

    raw = ROOT / "dataset" / "raw"
    if raw.exists() and any(iter_images(raw)):
        from _ped.discovery import build_layout
        from _ped.split import choose_source

        layout = build_layout(raw)
        src, why = choose_source(layout)
        if src is not None:
            labels = src.load()
            records = [(l.path, None, l.width, l.height, l.boxes) for l in labels.values()]
            return summarize(records, list(src.findings)), {"source": f"raw/{raw.name} ({why})"}
    return DatasetStats(), {"source": "none"}


def weights_reachable(model_name: str, timeout: float = 6.0) -> tuple[bool, str]:
    """Is the pretrained checkpoint actually fetchable right now?

    Ultralytics retries a blocked download 3 times with backoff (~40 s) before giving up,
    which is a long silence when the real answer is "this network cannot reach the asset
    CDN". So probe the exact URL first: a wrong answer here just costs one HEAD request,
    while the alternative costs the user several minutes of confusing warning spam.
    """
    import urllib.error
    import urllib.request

    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}"
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        if Path(model_name).name in GITHUB_ASSETS_NAMES:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ped-hackathon"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status < 400, f"HTTP {resp.status}"
        return True, "not a known ultralytics asset; letting YOLO() resolve it"
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            # 404 on a wrong tag is possible, so don't block on it -- just note it.
            return True, f"HTTP {exc.code} (may be a stale release tag; letting YOLO() try)"
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"


def resolve_model(model_name: str, from_scratch: bool, verbose: bool = True) -> tuple[str, str | None]:
    """Return (model_arg, warning).

    A local .pt is used as-is. A remote .pt is only attempted if the CDN actually
    answers, otherwise the architecture YAML is used so the run still happens -- with a
    loud warning, because from-scratch training on a small industrial dataset is a last
    resort, not a strategy.
    """
    if from_scratch:
        yml = Path(model_name).with_suffix(".yaml").name
        return yml, (
            "training from scratch (architecture YAML): no pretrained features, so this "
            "needs many more images and epochs. Fine to validate the pipeline; not the "
            "way to win mAP@45."
        )
    if Path(model_name).suffix == ".pt" and not Path(model_name).is_file():
        ok, why = weights_reachable(Path(model_name).name)
        if not ok:
            yml = Path(model_name).with_suffix(".yaml").name
            return yml, (
                f"cannot fetch {model_name} ({why}); building {yml} from scratch instead. "
                "On a machine that can reach the ultralytics release assets this downloads "
                "automatically and gives a large head start."
            )
    return model_name, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the PED detector.")
    ap.add_argument("--data", default=None, help="dataset yaml (default: configs/data.prepared.yaml)")
    ap.add_argument("--model", default=None, help="yolo11n.pt / yolo11s.pt / path to your .pt")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, help="'0', 'cpu', or 'cuda:1' — overrides auto-pick")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="ped_baseline")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--resume", default=None, help="path to last.pt to continue")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    ap.add_argument("--allow-cpu", action="store_true", help="permit a (slow) CPU run without a GPU")
    ap.add_argument("--fraction", type=float, default=None, help="subset of training data, e.g. 0.5")
    ap.add_argument("--mosaic", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="build the architecture from yaml instead of loading pretrained weights "
        "(needs far more data; mainly for offline smoke tests)",
    )
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)
    prof = detect_hardware()
    print("=== HARDWARE ===")
    for line in prof.summary_lines():
        print(f"  {line}")

    stats, meta = dataset_stats(data_yaml)
    print(f"\n=== DATASET ({meta['source']}) ===")
    print(f"  images {stats.images}, boxes {stats.boxes}, per class {dict(stats.per_class)}")
    if stats.images == 0:
        print(
            "\n[ERROR] no training images found. If configs/data.prepared.yaml is missing or stale,\n"
            "        run: python scripts/prepare_dataset.py"
        )
        return 2

    plan = plan_training(
        prof,
        stats,
        prefer_model=args.model,
        prefer_imgsz=args.imgsz,
        allow_cpu=args.allow_cpu or args.smoke,
        smoke=args.smoke,
    )
    if args.batch:
        plan.batch_size = args.batch
        plan.reasons.append(f"batch overridden to {args.batch}")
    if args.epochs:
        plan.epochs = args.epochs
    if args.fraction:
        plan.fraction = args.fraction
    if args.device is not None:
        plan.device = int(args.device) if args.device.isdigit() else args.device

    print("\n=== CHOSEN CONFIG ===")
    for line in plan.lines():
        print(f"  {line}")

    if plan.device == "cpu" and not args.allow_cpu and not args.smoke:
        print(
            f"\n[ABORT] no usable GPU. CPU training for {plan.epochs} epochs is not a hackathon strategy.\n"
            "        Use --allow-cpu to proceed anyway, or --smoke for a plumbing test, or move\n"
            "        this repo to a GPU box (Colab/Kaggle: git clone, pip install -r requirements.txt)."
        )
        return 3

    if args.dry_run:
        arg, warn = resolve_model(plan.model_name, args.from_scratch)
        print(f"  weights: {arg}" + (f"\n  [WARN] {warn}" if warn else "  (pretrained download expected)"))
        print("\n[dry-run] nothing trained. Remove --dry-run to start.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed: python -m pip install -r requirements.txt", file=sys.stderr)
        return 4

    overrides = dict(
        data=str(data_yaml),
        epochs=plan.epochs,
        imgsz=plan.image_size,
        batch=plan.batch_size,
        device=plan.device,
        workers=plan.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        cache=plan.cache,
        fraction=plan.fraction,
        save_period=10,
        # Small industrial datasets overfit fast; keep augmentation sane, not maximal.
        mosaic=0.5 if args.mosaic is None else args.mosaic,
        mixup=0.05,
        copy_paste=0.1 if stats.boxes < 400 else 0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.35,
        fliplr=0.5,
        flipud=0.0,
    )
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"plan": plan.__dict__, "hardware": prof.as_dict(), "data_yaml": str(data_yaml), "overrides": {k: str(v) for k, v in overrides.items()}}, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== TRAINING → {out_dir} ===")
    started = time.time()
    model_arg, model_warning = (plan.model_name, None) if args.resume else resolve_model(plan.model_name, args.from_scratch)
    if model_warning:
        print(f"  [WARN] {model_warning}")
    try:
        model = YOLO(args.resume if args.resume else model_arg)
    except Exception as exc:
        # Most common real cause: a sandbox/proxy that blocks the asset CDN, so the
        # pretrained checkpoint cannot be fetched. Don't lose the whole run over it.
        if args.resume or args.from_scratch:
            raise
        yml = Path(model_arg).with_suffix(".yaml").name
        print(
            f"  [WARN] could not load {model_arg}: {type(exc).__name__}: {str(exc)[:160]}\n"
            f"         falling back to {yml} (from scratch). Pretrained weights give a large, "
            f"cheap accuracy head start on small datasets — retry where the download works "
            f"before trusting this model's numbers."
        )
        model = YOLO(yml)
    if args.resume:
        overrides.pop("epochs", None)
        model.train(resume=True, **{k: v for k, v in overrides.items() if k in {"data", "device", "workers", "batch"}})
    else:
        results = model.train(**overrides)

    best = out_dir / "weights" / "best.pt"
    print(f"\ntrain wall-time: {(time.time() - started) / 60:.1f} min")
    print(f"best weights: {best}")
    if best.is_file():
        print(f"size: {best.stat().st_size / 1024 / 1024:.1f} MiB")
        try:
            mb = best.stat().st_size / 1024 / 1024
            if mb > 100:
                print("  [WARN] most competitions cap submission size; note the number now, not at the deadline")
        except Exception:
            pass
    print(
        "\nNOW — Ultralytics mAP@50 is NOT the scored metric. Run:\n"
        f"  python scripts/evaluate.py --weights {best} --name {args.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
