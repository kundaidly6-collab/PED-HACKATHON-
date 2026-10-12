#!/usr/bin/env python3
"""Generate a FAKE dataset that looks structurally like a real one.

Purpose: verify the pipeline end-to-end (audit → convert → validate → train → mAP@50
→ submit) on a machine with no GPU and no data, so that when the real dataset arrives
you are debugging your model and not your plumbing.

It is not a substitute for real data and it is not for submission: the "welds" are
drawn circles. Two deliberate wrinkles make it a useful test:
  * one class (Hole) is rarer and smaller than the other  -> exercises class balance
  * a few images carry NO objects                          -> exercises empty labels
  * a "component id" is embedded in filenames              -> exercises leakage-safe splits

    python scripts/make_synthetic_data.py --images 40 --out dataset/raw
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def metal_plate(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    base = rng.normal(150, 22, size=(h, w))
    # brushed-metal streaks: low vertical frequency, constant along x
    rows = max(1, h // 3)
    streak = np.repeat(rng.normal(0, 9, size=(rows, 1)), h // rows + 1, axis=0)[:h, :]
    arr = np.clip(base + np.repeat(streak, w, axis=1), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(0.8))
    return img.convert("RGB")


def draw_hole(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    """A hole: dark interior + bright rim highlight on the upper edge."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 18, 22))
    draw.arc([cx - r, cy - r, cx + r, cy + r], start=200, end=340, fill=(235, 235, 235), width=max(1, r // 5))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 70, 75), width=max(1, r // 8))


def draw_spot(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, rng: np.random.Generator) -> None:
    """A spot weld: bright nugget, heat-affected ring, some spatter."""
    heat = int(r * 1.6)
    draw.ellipse([cx - heat, cy - heat, cx + heat, cy + heat], fill=(96, 92, 88))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(246, 244, 238))
    for _ in range(4):
        a = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(r, heat * 1.3)
        px, py = int(cx + np.cos(a) * rr), int(cy + np.sin(a) * rr)
        draw.point((px, py), fill=(250, 250, 245))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(150, 148, 140), width=1)


def make_dataset(out: Path, n_parts: int, seed: int, fmt: str, size: int) -> dict:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    img_dir = out / "images"
    ann_dir = out / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    gt_records, coco_images, coco_anns, next_ann = [], [], [], 1
    for part in range(n_parts):
        component = f"part{part:03d}"
        frames = random.randint(2, 4)
        for frame in range(frames):
            w, h = size, int(size * rng.uniform(0.7, 1.0))
            img = metal_plate(rng, w, h)
            draw = ImageDraw.Draw(img)
            boxes = []
            # A true negative must contain NO objects. Deciding this after drawing would
            # produce images with visible welds and empty label files, which teaches the
            # detector that real objects are background -- the exact label poison the
            # audit is meant to catch.
            if random.random() < 0.12:
                name = f"{component}_frame{frame:02d}"
                img.save(img_dir / f"{name}.jpg", quality=92)
                gt_records.append({"image": f"{name}.jpg", "width": w, "height": h, "boxes": []})
                if fmt in ("yolo", "both"):
                    (ann_dir / f"{name}.txt").write_text("", encoding="utf-8")
                if fmt == "coco":
                    coco_images.append({"id": part * 10 + frame, "file_name": f"{name}.jpg", "width": w, "height": h})
                if fmt == "voc":
                    (ann_dir / f"{name}.xml").write_text(
                        "<annotation>\n"
                        f"  <filename>{name}.jpg</filename>\n"
                        f"  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>\n"
                        "</annotation>\n",
                        encoding="utf-8",
                    )
                continue
            n_spots = random.randint(2, 6)
            n_holes = random.randint(0, 2)  # rare, small -> tests imbalance
            for _ in range(n_spots):
                r = int(rng.uniform(0.02, 0.05) * min(w, h))
                cx, cy = int(rng.uniform(r * 2, w - r * 2)), int(rng.uniform(r * 2, h - r * 2))
                draw_spot(draw, cx, cy, r, rng)
                boxes.append((0, cx - r, cy - r, cx + r, cy + r))
            for _ in range(n_holes):
                r = int(rng.uniform(0.012, 0.028) * min(w, h))
                cx, cy = int(rng.uniform(r * 2, w - r * 2)), int(rng.uniform(r * 2, h - r * 2))
                draw_hole(draw, cx, cy, r)
                boxes.append((1, cx - r, cy - r, cx + r, cy + r))
            name = f"{component}_frame{frame:02d}"
            img.save(img_dir / f"{name}.jpg", quality=92)
            gt_records.append({"image": f"{name}.jpg", "width": w, "height": h, "boxes": [list(b) for b in boxes]})

            if fmt in ("yolo", "both"):
                lines = [
                    f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                    for cls, x1, y1, x2, y2 in boxes
                ]
                (ann_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if fmt == "coco":
                coco_images.append({"id": part * 10 + frame, "file_name": f"{name}.jpg", "width": w, "height": h})
                for cls, x1, y1, x2, y2 in boxes:
                    coco_anns.append(
                        {
                            "id": next_ann,
                            "image_id": part * 10 + frame,
                            "category_id": 7 + cls,  # deliberately NOT 0/1
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0,
                        }
                    )
                    next_ann += 1
            if fmt == "voc":
                xml = [
                    "<annotation>",
                    f"  <filename>{name}.jpg</filename>",
                    f"  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>",
                ]
                for cls, x1, y1, x2, y2 in boxes:
                    xml.append(
                        f"  <object><name>{'Spot' if cls == 0 else 'Hole'}</name>"
                        f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
                    )
                xml.append("</annotation>")
                (ann_dir / f"{name}.xml").write_text("\n".join(xml), encoding="utf-8")

    if fmt == "coco":
        coco = {
            "images": coco_images,
            "annotations": coco_anns,
            "categories": [{"id": 7, "name": "Spot"}, {"id": 8, "name": "Hole"}],
        }
        (ann_dir / "instances_train.json").write_text(json.dumps(coco, indent=1), encoding="utf-8")

    (out / "_ground_truth.json").write_text(json.dumps(gt_records, indent=1), encoding="utf-8")
    return {
        "images": len(gt_records),
        "boxes": sum(len(r["boxes"]) for r in gt_records),
        "empty_images": sum(1 for r in gt_records if not r["boxes"]),
        "format": fmt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dataset" / "raw"))
    ap.add_argument("--images", type=int, default=30, help="number of components (each gets 2-4 frames)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--format", default="yolo", choices=["yolo", "coco", "voc", "both"])
    ap.add_argument("--defects", action="store_true", help="inject pixel-coords + bad class id so the audit has something to catch")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "_ground_truth.json").exists():
        print(f"[skip] {out} already looks generated. Delete it to regenerate.")
        return 0
    info = make_dataset(out, args.images, args.seed, args.format, args.size)
    print(f"SYNTHETIC dataset written to {out}")
    print(f"  images: {info['images']}  boxes: {info['boxes']}  empty labels: {info['empty_images']}  format: {info['format']}")
    print("\nThis is fake data for pipeline testing only. It must not be submitted as a model.")

    # A couple of *deliberate* defects so the audit has something to catch.
    if not args.defects:
        return 0
    txts = sorted((out / "annotations").glob("*.txt"))
    if txts:
        victim = txts[1 % len(txts)]
        lines = victim.read_text().splitlines()
        if lines:
            cls = lines[0].split()[0]
            victim.write_text(f"{cls} 320.0 240.0 40.0 40.0\n", encoding="utf-8")  # pixel coords, by mistake
            print(f"  injected defect: {victim.name} now holds PIXEL coordinates (audit must flag it)")
        if len(txts) > 3:
            txts[3].write_text("5 0.5 0.5 0.1 0.1\n", encoding="utf-8")  # class id 5 does not exist
            print(f"  injected defect: {txts[3].name} now holds class id 5 (audit must flag it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
