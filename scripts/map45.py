#!/usr/bin/env python3
"""The metric that actually wins the hackathon: mAP@45, computed properly.

Ultralytics' ``model.val()`` gives mAP@50 and mAP@50:95. ``--iou`` at inference only
changes NMS, not the evaluation threshold. So this scores the model ourselves:

  ground truth : YOLO label files of the split
  predictions  : YOLO-format .txt files (per image) OR a trained model run live
  matching     : one-to-one greedy, descending confidence, IoU >= 0.45
  AP           : 101-point interpolation over a 0..1 recall grid, per class
  mAP          : mean over classes that have ground truth  (COCO convention)

Also sweeps the confidence threshold, because the organizers' score does depend on
which boxes you submit — that choice should be measured, not defaulted to 0.25.

    python scripts/map45.py --preds runs/ped_predictions/labels --labels dataset/prepared/labels/val
    python scripts/map45.py --weights runs/ped_baseline/weights/best.pt --split val
    python scripts/map45.py ... --report-json runs/map45_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ped.labels import CLASS_NAMES  # noqa: E402
from _ped.metrics import GTBox, PredBox, best_confidence, interpolate_ap, iou_matrix, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_sizes(img_dir: Path) -> dict[str, tuple[int, int]]:
    from PIL import Image

    sizes: dict[str, tuple[int, int]] = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            with Image.open(p) as im:
                sizes[p.stem] = (im.width, im.height)
        except Exception:
            continue
    return sizes


def load_yolo_boxes(label_dir: Path, sizes: dict[str, tuple[int, int]], conf_default: float = 1.0) -> dict[str, list]:
    """Parse YOLO txt into absolute-pixel boxes for each image present in *sizes*."""
    out: dict[str, list] = {}
    for stem, (w, h) in sizes.items():
        boxes: list = []
        lp = label_dir / f"{stem}.txt"
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                xy = np.array([max(0, x1), max(0, y1), min(w, x2), min(h, y2)], dtype=float)
                conf = float(parts[5]) if len(parts) > 5 else conf_default
                if xy[2] <= xy[0] or xy[3] <= xy[1]:
                    continue
                boxes.append((cls, xy, conf))
        out[stem] = boxes
    return out


def gt_from_labels(label_dir: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[GTBox]]:
    return {s: [GTBox(cls=c, xyxy=xy) for c, xy, _ in boxes] for s, boxes in load_yolo_boxes(label_dir, sizes).items()}


def preds_from_files(pred_dir: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[PredBox]]:
    """Accepts two submission flavours, auto-detected per file:
       'cls cx cy w h conf'   (YOLO + conf)
       'label_id conf x y w h' (Kaggle object-detection style)
    plus a bare COCO-JSON in the same directory if no .txt files exist.
    """
    txts = sorted(pred_dir.glob("*.txt"))
    if txts:
        out: dict[str, list[PredBox]] = defaultdict(list)
        for tp in txts:
            stem = tp.stem.split("__")[0]  # 'img__kaggle.txt' selects the 2nd layout
            fmt = "kaggle" if tp.stem.endswith("__kaggle") else "yolo"
            w, h = sizes.get(stem, (1, 1))
            for line in tp.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.replace(",", " ").split()
                if len(parts) < 5:
                    continue
                vals = [float(p) for p in parts[:6]] if len(parts) >= 6 else [*parts[:5], "1"]
                try:
                    a, b, c, d, e, f = (float(v) for v in vals)
                except ValueError:
                    continue
                if fmt == "kaggle" or (b <= 1.0 and c > 1.0):
                    # label conf x y w h (kaggle) — x,y are top-left, pixel units
                    cls, conf, x, y, bw, bh = int(a), b, c, d, e, f
                    xy = np.array([x, y, x + bw, y + bh], dtype=float)
                else:
                    cls, x, y, bw, bh, conf = int(a), b, c, d, e, f
                    xy = np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
                if xy[2] <= xy[0] or xy[3] <= xy[1]:
                    continue
                out[stem].append(PredBox(cls=cls, xyxy=xy, conf=conf))
        return dict(out)
    js = sorted(pred_dir.glob("*.json"))
    if js:
        return preds_from_coco_json(js[0], sizes)
    raise SystemExit(f"[ERROR] no .txt or .json predictions under {pred_dir}")


def preds_from_coco_json(path: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[PredBox]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    id2img = {im["id"]: Path(str(im.get("file_name", ""))).stem for im in data.get("images", [])}
    cat2cls = {c["id"]: i for i, c in enumerate(data.get("categories", []))} if data.get("categories") else {}
    out: dict[str, list[PredBox]] = defaultdict(list)
    for ann in data.get("annotations", []):
        stem = id2img.get(ann.get("image_id"))
        if stem is None:
            continue
        x, y, w, h = ann["bbox"]
        cls = cat2cls.get(ann.get("category_id"), 0)
        out[stem].append(PredBox(cls=cls, xyxy=np.array([x, y, x + w, y + h], dtype=float), conf=float(ann.get("score", 1.0))))
    return dict(out)


def run_model(weights: Path, img_dir: Path, sizes: dict[str, tuple[int, int]], imgsz: int, conf: float, device: str | None, tiled: int = 0, pred_out: Path | None = None) -> dict[str, list[PredBox]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[str, list[PredBox]] = {}
    if device:
        model.overrides["device"] = device
    sources = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    results = model.predict([str(s) for s in sources], imgsz=imgsz, conf=conf, verbose=False, max_det=3000, retina_masks=False)
    for res in results:
        stem = Path(res.path).stem
        w, h = sizes.get(stem, (res.orig_shape[1], res.orig_shape[0]))
        boxes: list[PredBox] = []
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls.item())
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                boxes.append(PredBox(cls=cls, xyxy=np.array([x1, y1, x2, y2]), conf=float(b.conf.item())))
        out[stem] = boxes
    return out


def sweep(weights: Path, img_dir: Path, gt: dict, sizes, imgsz: int, devicestr: str | None, thresholds: list[float]) -> list[tuple[float, float]]:
    """Single inference pass at the lowest threshold, then re-threshold for free.
    This is how you find the submission confidence without N re-runs."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if devicestr:
        model.overrides["device"] = devicestr
    lo = min(thresholds)
    sources = [str(p) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    results = model.predict(sources, imgsz=imgsz, conf=lo, verbose=False, max_det=5000)
    curve = []
    for thr in thresholds:
        preds: dict[str, list[PredBox]] = {}
        for res in results:
            stem = Path(res.path).stem
            kept = []
            if res.boxes is not None:
                for b in res.boxes:
                    if float(b.conf.item()) < thr:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    kept.append(PredBox(cls=int(b.cls.item()), xyxy=np.array([x1, y1, x2, y2]), conf=float(b.conf.item())))
            preds[stem] = kept
        card = score(gt, preds, iou_threshold=0.45, class_names={i: n for i, n in enumerate(CLASS_NAMES)}, conf_steps=0)
        curve.append((thr, card.map))
        print(f"  conf>={thr:<5} mAP@45={card.map:.4f}  preds={card.n_pred}")
    return curve


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="run this model on --images then score it")
    src.add_argument("--preds", help="score a directory of prediction files instead")
    ap.add_argument("--images", default=str(ROOT / "dataset/prepared/images/val"))
    ap.add_argument("--labels", default=str(ROOT / "dataset/prepared/labels/val"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01, help="inference floor; sweep on top of it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou", type=float, default=0.45, help="0.45 is the competition metric")
    ap.add_argument("--sweep", action="store_true", help="sweep confidence thresholds (models only)")
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--dump-preds", default=None, help="write model output as YOLO txt here (for validate/submission)")
    args = ap.parse_args()

    img_dir, label_dir = Path(args.images), Path(args.labels)
    if not img_dir.is_dir():
        print(f"[ERROR] image dir not found: {img_dir}", file=sys.stderr)
        return 2
    sizes = read_sizes(img_dir)
    gt = gt_from_labels(label_dir, sizes)
    n_gt = sum(len(v) for v in gt.values())
    if n_gt == 0:
        print(f"[ERROR] no ground truth in {label_dir} — nothing to score against.", file=sys.stderr)
        return 2
    print(f"scoring {len(sizes)} images · {n_gt} GT boxes · IoU threshold {args.iou}")

    weights = Path(args.weights) if args.weights else None
    if args.weights:
        if not weights.is_file():
            print(f"[ERROR] weights not found: {weights}", file=sys.stderr)
            return 2
        preds = run_model(weights, img_dir, sizes, args.imgsz, args.conf, args.device)
        if args.dump_preds:
            dump(preds, sizes, Path(args.dump_preds))
            print(f"predictions written to {args.dump_preds}")
    else:
        preds = preds_from_files(Path(args.preds), sizes)

    card = score(
        gt,
        preds,
        iou_threshold=args.iou,
        class_names={i: n for i, n in enumerate(CLASS_NAMES)},
        conf_steps=40 if args.weights is None else 0,
    )
    print("\n" + card.table())
    for note in card.notes:
        print(f"  ! {note}")
    best_conf, best_map = best_confidence(card)
    if card.conf_curve:
        print(f"\nbest confidence threshold in sweep: {best_conf:.3f} → mAP@{args.iou:g} = {best_map:.4f}")
    else:
        print("\n(run with --weights --sweep to measure the best submission confidence)")
    if args.sweep and args.weights:
        print("\n=== confidence sweep (single inference pass, re-thresholded) ===")
        curve = sweep(weights, img_dir, gt, sizes, args.imgsz, args.device, [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
        swept_conf, swept_map = max(curve, key=lambda t: t[1])
        print(f"\nSUBMIT AT conf≈{swept_conf:g} (validation mAP@45 {swept_map:.4f}); pass that to submit.py --conf")
        card.conf_curve = [(t, a) for t, a in curve]
        best_conf, best_map = swept_conf, swept_map

    misses = error_report(gt, preds, args.iou)
    if misses:
        print("\n=== worst images (missed objects) ===")
        for stem, n_fn, n_fp in misses[:12]:
            print(f"  {stem}: {n_fn} missed, {n_fp} false positives")
        print("  → draw these with validate_labels.py --sheet, or check for occluded/ambiguous parts")

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "map": card.map,
                    "iou": card.iou_threshold,
                    "per_class": [s.__dict__ for s in card.per_class],
                    "conf_curve": card.conf_curve,
                    "best_conf": best_conf,
                    "n_gt": card.n_gt,
                    "n_pred": card.n_pred,
                    "notes": card.notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nreport: {out}")
    return 0


def dump(preds: dict[str, list[PredBox]], sizes: dict[str, tuple[int, int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, boxes in preds.items():
        w, h = sizes.get(stem, (1, 1))
        lines = []
        for b in sorted(boxes, key=lambda x: -x.conf):
            cx = (b.xyxy[0] + b.xyxy[2]) / 2 / w
            cy = (b.xyxy[1] + b.xyxy[3]) / 2 / h
            bw = (b.xyxy[2] - b.xyxy[0]) / w
            bh = (b.xyxy[3] - b.xyxy[1]) / h
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {b.conf:.5f}")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def error_report(gt: dict[str, list[GTBox]], preds: dict[str, list[PredBox]], iou_thr: float) -> list[tuple[str, int, int]]:
    """Per-image missed / false-positive counts, worst first — this drives what to fix."""
    rows = []
    for stem, gboxes in gt.items():
        gm = np.stack([g.xyxy for g in gboxes]) if gboxes else np.zeros((0, 4))
        pboxes = preds.get(stem, [])
        pm = np.stack([p.xyxy for p in pboxes]) if pboxes else np.zeros((0, 4))
        matched: set[int] = set()
        n_fp = 0
        for p in sorted(pboxes, key=lambda x: -x.conf):
            if gm.size == 0:
                n_fp += 1
                continue
            ious = iou_matrix(p.xyxy[None, :], gm)[0]
            hit = -1
            for gi in np.argsort(-ious):
                if ious[gi] < iou_thr:
                    break
                if int(gi) not in matched:
                    hit = int(gi)
                    break
            if hit >= 0:
                matched.add(hit)
            else:
                n_fp += 1
        n_fn = len(gboxes) - len(matched)
        if n_fn or n_fp:
            rows.append((stem, n_fn, n_fp))
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
