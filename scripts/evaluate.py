#!/usr/bin/env python3
"""Phase 4 — Ultralytics validation (mAP@50, mAP@50:95, curves, plots) for context.

The scored number is mAP@45 and that is NOT what this prints — run scripts/map45.py
for it. This script exists for the *diagnosis*: PR curves, per-class AP, and the
confusion matrix, which tell you whether you are losing points on missed welds,
missed holes, or localization error.

    python scripts/evaluate.py --name ped_baseline
    python scripts/evaluate.py --weights runs/ped_1024/weights/best.pt --imgsz 1024
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="default: runs/<name>/weights/best.pt")
    ap.add_argument("--name", default="ped_baseline")
    ap.add_argument("--data", default=str(ROOT / "configs" / "data.prepared.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.01, help="low floor; the sweep decides the submit threshold")
    ap.add_argument("--skip-map45", action="store_true")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else ROOT / "runs" / args.name / "weights" / "best.pt"
    if not weights.is_file():
        print(f"[ERROR] weights not found: {weights}\n        train first: python scripts/train.py", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    print(f"=== ultralytics val ({weights}) ===")
    kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        plots=True,
        project=str(ROOT / "runs"),
        name=f"{args.name}_val",
        exist_ok=True,
        verbose=False,
    )
    if args.device:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)

    print(f"\nmAP@50      : {metrics.box.map50:.4f}   (NOT the scored metric)")
    print(f"mAP@50:95   : {metrics.box.map:.4f}")
    print(f"precision   : {metrics.box.mp:.4f}   recall: {metrics.box.mr:.4f}")
    print("\nper-class AP@50:")
    names = model.names if isinstance(model.names, dict) else dict(enumerate(["Spot", "Hole"]))
    for i, ap50 in enumerate(metrics.box.ap50):
        print(f"  {names.get(i, i):<6}: {ap50:.4f}")
    gap = metrics.box.map50 - metrics.box.map
    print(f"\nlocalization gap (mAP50 - mAP50:95) = {gap:.3f}")
    print("  large gap  → boxes are found but sloppy: raise imgsz, check aspect ratio, more scale aug")
    print("  low mAP50  → detections are missing: that is a data/labels/recall problem, not a box problem")

    if not args.skip_map45:
        print("\n=== mAP@45 (the metric you are scored on) ===")
        sys.path.insert(0, str(ROOT / "scripts"))
        from _ped.dataset_cfg import resolve_split_dirs

        # Resolve from the dataset yaml instead of assuming dataset/prepared: the data
        # may have come from Kaggle with any folder names, and a hardcoded path here
        # silently scores against nothing.
        img_dir, lbl_dir, notes = resolve_split_dirs(Path(args.data), args.split, root_hint=ROOT)
        for n in notes:
            print(f"  · {n}")
        if img_dir is None or lbl_dir is None:
            print("  [SKIP] cannot locate the split for scoring — pass --data pointing at the yaml")
            return 0
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "map45.py"),
            "--weights", str(weights),
            "--imgsz", str(args.imgsz),
            "--labels", str(lbl_dir),
            "--images", str(img_dir),
            "--sweep",
            "--report-json", str(ROOT / "runs" / f"{args.name}_map45.json"),
        ]
        if args.device:
            cmd += ["--device", args.device]
        rc = subprocess.call(cmd)
        rep = ROOT / "runs" / f"{args.name}_map45.json"
        if rep.is_file():
            data = json.loads(rep.read_text(encoding="utf-8"))
            print(f"\nHEADLINE mAP@45 = {data['map']:.4f}   best conf = {data.get('best_conf')}")
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
