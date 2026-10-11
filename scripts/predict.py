#!/usr/bin/env python3
"""Phase 5 — inference on new components, with optional tiling for tiny welds.

Tiling matters more than people expect: a 4096px frame letterboxed to 640px shrinks a
20px spot to ~3px, which is below the stride-32 detection grid. --tile splits the
image, runs the model per tile, then merges boxes with global NMS (an object cut by a
tile border is kept if it survives once, which is the point).

    python scripts/predict.py --source component.jpg
    python scripts/predict.py --source dataset/raw/images --tile 640 --overlap 0.25
    python scripts/predict.py --source test/ --out runs/preds --conf 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "ped_baseline" / "weights" / "best.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def nms(boxes: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
    """Global NMS across tiles. boxes rows are x1,y1,x2,y2,conf,cls. Returns kept indices."""
    if boxes.size == 0 or boxes.shape[1] < 6:
        return np.arange(len(boxes), dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-boxes[:, 4])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        # only suppress the same class — a Spot and a Hole may legitimately overlap
        same = boxes[rest, 5] == boxes[i, 5]
        order = rest[(iou < iou_thr) | ~same]
    return np.array(keep, dtype=int)


def tile_coords(W: int, H: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, W - tile + 1), step))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    ys = list(range(0, max(1, H - tile + 1), step))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in sorted(set(ys)) for x in sorted(set(xs))]


def predict_one(model, path: Path, args) -> list[tuple[float, float, float, float, float, int]]:
    """Returns boxes as x1,y1,x2,y2,conf,cls in ORIGINAL image pixels."""
    if not args.tile:
        res = model.predict(str(path), imgsz=args.imgsz, conf=args.conf, verbose=False, max_det=3000)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        arr = res.boxes.numpy().data
        return [(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), int(a[5])) for a in arr]

    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"could not read {path}")
    H, W = img.shape[:2]
    boxes: list[tuple[float, float, float, float, float, int]] = []
    for (x0, y0, x1, y1) in tile_coords(W, H, args.tile, args.overlap):
        crop = img[y0:y1, x0:x1]
        if min(crop.shape[0], crop.shape[1]) < 32:
            continue
        res = model.predict(crop, imgsz=args.imgsz, conf=args.conf, verbose=False, max_det=3000)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        # Ultralytics returns box coords in the scale it received (the crop), so the
        # only correction needed is the crop origin offset -- no letterbox maths here.
        for a in res.boxes.numpy().data:
            boxes.append((float(a[0]) + x0, float(a[1]) + y0, float(a[2]) + x0, float(a[3]) + y0, float(a[4]), int(a[5])))
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    keep = nms(arr)  # nms reads cols 0..3 as xyxy, col 4 as conf, col 5 as class
    return [tuple(v) for v in arr[keep]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image, directory, or glob")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "ped_predictions"))
    ap.add_argument("--save-images", action="store_true", help="write annotated copies")
    ap.add_argument("--tile", type=int, default=0, help="tile size (e.g. 640) for large images with tiny targets")
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"[ERROR] weights not found: {weights}\n        (paths must be absolute-pixel aware: see --tile)", file=sys.stderr)
        return 2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if args.device:
        model.overrides["device"] = args.device

    src = Path(args.source)
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    elif any(ch in str(src) for ch in "*?["):
        files = sorted(Path().glob(str(src)))
    else:
        files = [src]
    if not files:
        print(f"[ERROR] no images at {src}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} image(s) · imgsz={args.imgsz} conf={args.conf} tile={args.tile or 'off'}")
    totals = {0: 0, 1: 0}
    for path in files:
        from PIL import Image

        try:
            with Image.open(path) as im:
                W, H = im.size
        except Exception as exc:
            print(f"  [skip] {path.name}: {exc}")
            continue
        boxes = predict_one(model, path, args)
        lines = []
        for (x1, y1, x2, y2, conf, cls) in boxes:
            totals[cls] = totals.get(cls, 0) + 1
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.5f}")
        (out_dir / "labels" / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        named = [f"{model.names.get(c, c)}@{cf:.2f}" for (_, _, _, _, cf, c) in boxes[:6]]
        print(f"  {path.name}: {len(boxes)} box(es)" + (f" — {', '.join(named)}" if named else " (none)"))
        if args.save_images:
            try:
                import cv2

                img = cv2.imread(str(path))
                for (x1, y1, x2, y2, conf, cls) in boxes:
                    color = (0, 220, 255) if cls == 0 else (0, 60, 255)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img, f"{model.names.get(cls, cls)} {conf:.2f}", (int(x1), max(0, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imwrite(str(out_dir / "images" / path.name), img)
            except Exception as exc:
                print(f"    (could not render overlay: {exc})")

    print(f"\ntotals: " + ", ".join(f"{model.names.get(k, k)}={v}" for k, v in sorted(totals.items())))
    print(f"labels (YOLO format, with confidence) → {out_dir / 'labels'}")
    print("next: score them with  python scripts/map45.py --preds " + str(out_dir / "labels") + " --labels <gt dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
