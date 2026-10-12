#!/usr/bin/env python3
"""Phase 5 — produce the submission in the organizers' format.

THE format is fixed by the competition and is the default here:

  ped        ID,predictions as a two-column CSV, predictions a JSON list:
             [{"class":"Hole","xcenter":0.52,"ycenter":0.43,"width":0.18,
               "height":0.21,"confidence":0.91}]
             IDs are the exact test filenames; an image with no detections is [].
             Validate the file with scripts/validate_submission.py before uploading.

The other formats below are auxiliary exports, not submission candidates. Coordinates
are always converted back to each ORIGINAL image's own size — predicting in the
letterboxed 640×640 space is a silent, total loss of score.

Formats:
  ped        ID,predictions (organizers' format)                        (default)
  yolo       <cls> <cx> <cy> <w> <h> <conf>     normalized 0..1
  kaggle     <cls> <conf> <x> <y> <w> <h>       pixels, top-left
  coco       one COCO JSON with images+annotations+categories
  csv        <image>,<cls>,<conf>,<xmin>,<ymin>,<xmax>,<ymax>       pixels
  xml        Pascal VOC detection XML per image

    python scripts/submit.py --weights runs/ped_baseline/weights/best.pt --source dataset/test/images
    python scripts/submit.py --preds runs/ped_predictions/labels --source dataset/test/images \
        --out runs/submission.csv
    python scripts/validate_submission.py --csv runs/submission.csv --test-images dataset/test/images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ped.labels import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_boxes(label_dir: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[tuple]]:
    """Read YOLO txt with confidence (from predict.py) back into pixel boxes."""
    out: dict[str, list[tuple]] = {}
    for stem, (w, h) in sizes.items():
        rows = []
        lp = label_dir / f"{stem}.txt"
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                conf = float(p[5]) if len(p) > 5 else 1.0
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                rows.append((x1, y1, x2, y2, conf, cls))
        out[stem] = rows
    return out


def write_ped(boxes, sizes, out: Path, names: list[str] | None = None,
              ids: dict[int, int] | None = None,
              filenames: dict[str, str] | None = None,
              key_style: str = "plain") -> Path:
    """The organizers' submission file: exactly two columns, ID and predictions.

        ID,predictions
        image001.jpg,"[{""class"":""Hole"",""xcenter"":0.52,""ycenter"":0.43,
                        ""width"":0.18,""height"":0.21,""confidence"":0.91}]"
        image_003.jpg,"[]"

    Details that decide whether this scores at all:
      * ID is the EXACT filename, extension included -- not the stem.
      * Every test image needs a row, and an image with nothing detected must be an
        empty list. A missing row is a missing image, not a missing detection.
      * class travels as the NAME (Hole/Spot), so the internal numeric ids -- and the
        order they are in -- cannot leak into the submission.
      * width and height must be non-zero, so a box that rounds away is dropped with a
        count rather than written as an invalid 0.
    """
    names = names or list(CLASS_NAMES)
    ids = ids or {i: i for i in range(len(CLASS_NAMES))}
    kx, ky = ("xcenter", "ycenter") if key_style == "plain" else ("x_center", "y_center")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, n_det, n_empty, dropped = [], 0, 0, 0
    for stem, dets in boxes.items():
        fname = (filenames or {}).get(stem, f"{stem}.jpg")
        w, h = sizes.get(stem, (1, 1))
        preds = []
        for x1, y1, x2, y2, conf, cls in sorted(dets, key=lambda r: -r[4]):
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                dropped += 1
                continue
            preds.append({
                "class": names[ids[cls]],
                kx: round(((x1 + x2) / 2) / w, 6),
                ky: round(((y1 + y2) / 2) / h, 6),
                "width": round(bw, 6),
                "height": round(bh, 6),
                "confidence": round(min(max(float(conf), 0.0), 1.0), 6),
            })
        n_det += len(preds)
        n_empty += not preds
        # Compact separators to match the organizers' example byte-for-byte in shape.
        rows.append((fname, json.dumps(preds, separators=(",", ":"))))

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "predictions"])
        w.writerows(rows)

    print(f"  wrote {len(rows)} rows / {n_det} detections to {out}")
    if n_empty:
        print(f"    {n_empty} image(s) with no detections written as [] (required, not optional)")
    if dropped:
        print(f"    [WARN] dropped {dropped} box(es) with zero width/height — the spec forbids them")
    return out


def class_map(order: str) -> tuple[list[tuple[int, str]], dict[int, int]]:
    """(id, name) pairs in output order, plus internal-id -> output-id.

    Internal ids are this repo's training convention (0=Spot, 1=Hole). The submission's
    ids are a separate choice. Keeping them in one function means a row can never show a
    flipped id beside a name that came from the other ordering.
    """
    names = list(CLASS_NAMES) if order == "spot-first" else list(reversed(CLASS_NAMES))
    pairs = list(enumerate(names))
    ids = {i: i for i in range(len(CLASS_NAMES))} if order == "spot-first" \
        else {i: len(CLASS_NAMES) - 1 - i for i in range(len(CLASS_NAMES))}
    return pairs, ids


def write_yolo(boxes, sizes, out: Path, names: list[str] | None = None,
         ids: dict[int, int] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem, rows in boxes.items():
        w, h = sizes[stem]
        lines = [
            f"{ids[cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {conf:.5f}"
            for x1, y1, x2, y2, conf, cls in sorted(rows, key=lambda r: -r[4])
        ]
        n += len(lines)
        (out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"  wrote {n} boxes across {len(boxes)} files to {out}/")
    return out


def write_kaggle(boxes, sizes, out: Path, names: list[str] | None = None,
         ids: dict[int, int] | None = None) -> Path:
    ids = ids or {i: i for i in range(len(CLASS_NAMES))}
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax", "width", "height"])
        for stem, rows in boxes.items():
            w, h = sizes[stem]
            for x1, y1, x2, y2, conf, cls in sorted(rows, key=lambda r: -r[4]):
                wcsv.writerow([f"{stem}.jpg", ids[cls], f"{conf:.5f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", w, h])
                n += 1
    print(f"  wrote {n} rows to {out}")
    return out


def write_csv(boxes, sizes, out: Path, names: list[str] | None = None,
         ids: dict[int, int] | None = None) -> Path:
    names = names or list(CLASS_NAMES)
    ids = ids or {i: i for i in range(len(CLASS_NAMES))}
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["image", "class_id", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"])
        for stem, rows in boxes.items():
            for x1, y1, x2, y2, conf, cls in sorted(rows, key=lambda r: -r[4]):
                wcsv.writerow([f"{stem}.jpg", ids[cls], names[ids[cls]], f"{conf:.5f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"])
                n += 1
    print(f"  wrote {n} rows to {out}")
    return out


def write_coco(boxes, sizes, out: Path, names: list[str] | None = None,
         ids: dict[int, int] | None = None) -> Path:
    import datetime

    images, anns, next_id = [], [], 1
    for i, (stem, rows) in enumerate(sorted(boxes.items()), start=1):
        w, h = sizes[stem]
        images.append({"id": i, "file_name": f"{stem}.jpg", "width": w, "height": h})
        for x1, y1, x2, y2, conf, cls in rows:
            anns.append(
                {
                    "id": next_id,
                    "image_id": i,
                    "category_id": ids[cls] + 1,  # COCO ids are 1-based
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round((x2 - x1) * (y2 - y1), 2),
                    "iscrowd": 0,
                    "score": round(conf, 5),
                }
            )
            next_id += 1
    doc = {
        "info": {"description": "PED detection submission", "date_created": datetime.datetime.now().isoformat()},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": [{"id": ids[c] + 1, "name": names[ids[c]], "supercategory": "ped"}
                           for c in range(len(CLASS_NAMES))],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    print(f"  wrote {len(anns)} annotations / {len(images)} images to {out}")
    return out


def write_voc(boxes, sizes, out: Path, names: list[str] | None = None,
         ids: dict[int, int] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for stem, rows in boxes.items():
        w, h = sizes[stem]
        lines = [f"<annotation><filename>{stem}.jpg</filename>", f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"]
        for x1, y1, x2, y2, conf, cls in rows:
            lines.append(
                f"<object><name>{names[ids[cls]]}</name><confidence>{conf:.5f}</confidence>"
                f"<bndbox><xmin>{x1:.0f}</xmin><ymin>{y1:.0f}</ymin><xmax>{x2:.0f}</xmax><ymax>{y2:.0f}</ymax></bndbox></object>"
            )
        lines.append("</annotation>")
        (out / f"{stem}.xml").write_text("".join(lines), encoding="utf-8")
    print(f"  wrote {len(boxes)} XML files to {out}/")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="run the model on --source and submit those predictions")
    src.add_argument("--preds", help="use an existing directory of YOLO+conf txt files")
    ap.add_argument("--source", default=None, help="images to predict on "
                    "(default: the validation split named by configs/data.prepared.yaml)")
    ap.add_argument("--format", default="ped", choices=["ped", "yolo", "kaggle", "coco", "csv", "xml"],
                    help="'ped' is the organizers' ID,predictions format and the only one you should "
                         "upload; the rest are auxiliary exports")
    ap.add_argument("--test-images", default=None,
                    help="the test image folder. Guarantees EVERY image gets exactly one row "
                         "(empty detections become []), which the rules require")
    ap.add_argument("--keys", default="plain", choices=["plain", "underscore"],
                    help="JSON key style: 'plain' = xcenter/ycenter (matches the rules' example and "
                         "the sentence 'xcenter and ycenter represent the center'); "
                         "'underscore' = x_center/y_center (matches the field list). The rules show "
                         "BOTH, so this is worth confirming with the organizers — see README")
    ap.add_argument("--out", default=None, help="output file or directory (default: runs/submission_<format>)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="use the value eval_map.py --sweep recommended")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU (not the eval threshold)")
    ap.add_argument("--max-det", type=int, default=1000, help="raise if a frame has many welds")
    ap.add_argument("--tile", type=int, default=0, help="tiled inference for large images (see predict.py)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--agnostic-nms", action="store_true", help="suppress across classes (only if the organizers do)")
    ap.add_argument("--class-order", default="spot-first", choices=["spot-first", "hole-first"],
                    help="numeric id order in the OUTPUT. The competition rules fix the class NAMES "
                         "but never publish a numeric order, so if their reader assumes the other "
                         "mapping this is the one flag that fixes it (default: spot-first, this repo's "
                         "convention everywhere else)")
    args = ap.parse_args()

    from PIL import Image

    if args.source is None:
        from _ped.dataset_cfg import find_data_yaml, resolve_split_dirs

        yml = find_data_yaml(ROOT)
        resolved = None
        if yml:
            resolved, _lbl, _notes = resolve_split_dirs(yml, "val", root_hint=ROOT)
        if resolved is None:
            resolved = ROOT / "dataset" / "raw"
            print(f"[note] no dataset yaml found; defaulting --source to {resolved}")
        args.source = str(resolved)

    if args.weights:
        from ultralytics import YOLO

        files: list[Path] = []
        for s in args.source.split(","):
            p = Path(s.strip())
            files += sorted(p.rglob("*")) if p.is_dir() else [p]
        files = [f for f in files if f.suffix.lower() in IMG_EXTS]
        if not files:
            print(f"[ERROR] no images at {args.source}", file=sys.stderr)
            return 2
        model = YOLO(args.weights)
        if args.device:
            model.overrides["device"] = args.device
        sizes: dict[str, tuple[int, int]] = {}
        boxes: dict[str, list[tuple]] = {}
        filenames: dict[str, str] = {}
        tiled = bool(args.tile)
        for f in files:
            with Image.open(f) as im:
                sizes[f.stem] = (im.width, im.height)
            # The organizers want the EXACT filename in the ID column, so keep it
            # alongside the stem that every other script is keyed by.
            filenames[f.stem] = f.name
            if tiled:
                import subprocess

                tmp = ROOT / "runs" / "_submit_tiled"
                subprocess.check_call(
                    [
                        sys.executable, str(ROOT / "scripts" / "predict.py"),
                        "--source", str(f), "--weights", args.weights,
                        "--imgsz", str(args.imgsz), "--conf", str(args.conf),
                        "--tile", str(args.tile), "--out", str(tmp),
                    ]
                )
                from_ = read_boxes(tmp / "labels", sizes)
                boxes[f.stem] = from_.get(f.stem, [])
                continue
            res = model.predict(str(f), imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, verbose=False, agnostic_nms=args.agnostic_nms)[0]
            rows = []
            if res.boxes is not None:
                for b in res.boxes.numpy().data:
                    rows.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), int(b[5])))
            boxes[f.stem] = rows
    else:
        pred_dir = Path(args.preds)
        if not pred_dir.is_dir():
            print(f"[ERROR] {pred_dir} is not a directory", file=sys.stderr)
            return 2
        stems = {p.stem.split("__")[0] for p in pred_dir.glob("*.txt")}
        img_dir = Path(args.source)
        sizes = {}
        filenames = {}
        for stem in stems:
            for ext in IMG_EXTS:
                cand = img_dir / f"{stem}{ext}"
                if cand.is_file():
                    with Image.open(cand) as im:
                        sizes[stem] = (im.width, im.height)
                    filenames[stem] = cand.name
                    break
            else:
                sizes[stem] = (1, 1)
                print(f"  [WARN] no source image for {stem}; assuming 1×1 (normalized output only)")
        boxes = read_boxes(pred_dir, sizes)

    # The rules require EVERY test image to appear exactly once, with [] when nothing was
    # detected. Driving the file from the prediction set cannot guarantee that: an image
    # with no detections simply produces no label file. So when the test folder is known,
    # it decides the rows and the predictions only fill them.
    if args.test_images:
        tdir = Path(args.test_images)
        if not tdir.is_dir():
            print(f"[ERROR] --test-images {tdir} is not a directory", file=sys.stderr)
            return 2
        test_files = sorted(p for p in tdir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not test_files:
            print(f"[ERROR] no images under {tdir}", file=sys.stderr)
            return 2
        by_stem: dict[str, list[Path]] = {}
        for p in test_files:
            by_stem.setdefault(p.stem, []).append(p)
        collisions = {k: v for k, v in by_stem.items() if len(v) > 1}
        if collisions:
            print(f"[ERROR] test images share a stem across extensions, so an ID cannot be "
                  f"resolved unambiguously: {list(collisions)[:3]}", file=sys.stderr)
            return 2
        from PIL import Image as _I

        covered, empty, missing = 0, 0, 0
        for stem, (p,) in by_stem.items():
            filenames[stem] = p.name
            if stem in boxes:
                covered += 1
            else:
                with _I.open(p) as im:
                    sizes[stem] = (im.width, im.height)
                boxes[stem] = []
                empty += 1
        extra = set(boxes) - set(by_stem)
        if extra:
            missing = len(extra)
            print(f"[ERROR] {missing} prediction(s) do not correspond to any test image: "
                  f"{sorted(extra)[:3]}", file=sys.stderr)
            return 2
        print(f"\n=== test coverage ({tdir}) ===")
        print(f"  test images: {len(test_files)}   with detections: {covered}   "
              f"empty (written as []): {empty}")

    out = Path(args.out) if args.out else ROOT / "runs" / f"submission_{args.format}"
    n_boxes = sum(len(v) for v in boxes.values())

    # The rules fix the class NAMES and never state a numeric order. If the organizers'
    # reader disagrees with ours, every box is misclassified and the score collapses, so
    # the order is an explicit, printed choice rather than a silent assumption.
    pairs, ids = class_map(args.class_order)
    out_names = [n for _i, n in pairs]

    print(f"\n=== writing submission ({args.format}) ===")
    print(f"  images: {len(boxes)}   boxes: {n_boxes}   conf>={args.conf}")
    if args.format == "ped":
        # This format carries the class NAME, so it is immune to the id-order question
        # entirely -- which is why the id order no longer needs to be got right here.
        print(f"  classes   : {', '.join(out_names)} (by NAME, so no numeric order can be misread)")
    else:
        print(f"  class ids : " + ", ".join(f"{i}={n}" for i, n in pairs)
              + ("   [--class-order hole-first, NOT this repo's training ids]" if args.class_order == "hole-first" else ""))
    print(f"  coordinates converted back to each image's own pixel size (not the letterboxed {args.imgsz}²)")
    if args.format == "ped":
        target = write_ped(boxes, sizes, out if out.suffix else out / "submission.csv",
                           out_names, ids, filenames, args.keys)
        print(f"  ID column : exact test filenames, one row per image")
        print(f"  JSON keys : {('xcenter/ycenter' if args.keys == 'plain' else 'x_center/y_center')}"
              f"   [--keys {args.keys}]")
    elif args.format == "yolo":
        target = write_yolo(boxes, sizes, out, out_names, ids)
        # YOLO's own convention for decoding bare integer ids. Without it the output is
        # unreadable-by-construction and the class order is anyone's guess.
        print(f"  legend    : {Path(target) / 'classes.txt'} (ids, in order)")
    elif args.format == "kaggle":
        target = write_kaggle(boxes, sizes, out if out.suffix else out / "submission.csv", out_names, ids)
    elif args.format == "csv":
        target = write_csv(boxes, sizes, out if out.suffix else out / "submission.csv", out_names, ids)
    elif args.format == "coco":
        target = write_coco(boxes, sizes, out if out.suffix else out / "instances.json", out_names, ids)
    else:
        target = write_voc(boxes, sizes, out, out_names, ids)

    # The yolo and kaggle schemas carry ids only, with no room for a name. A legend
    # written from a DIFFERENT source than the ids is worse than none: built once, from
    # the same pairs that are printed above, it is the record of what was submitted.
    if args.format == "yolo":
        # YOLO's convention: one class name per line, id = line number, no keys.
        (Path(target) / "classes.txt").write_text("\n".join(out_names) + "\n", encoding="utf-8")
    elif args.format == "kaggle":
        # id-only schema, so the mapping travels beside it
        (Path(target).parent / "submission_classes.txt").write_text(
            "\n".join(f"{i}={n}" for i, n in pairs) + "\n", encoding="utf-8")

    if args.weights:
        wp = Path(args.weights)
        if wp.is_file():
            print(f"  weights: {wp} ({wp.stat().st_size / 1024 / 1024:.1f} MiB) — check the competition's size cap")
    print(f"\nOUTPUT: {target}")
    if args.format == "ped":
        print(f"\nValidate before uploading:  python scripts/validate_submission.py --csv {target}"
              + (f" --test-images {args.test_images}" if args.test_images else ""))
    print("\nBefore submitting, confirm against the organizers' example:")
    print(f"  · the class order above is what their reader expects ({', '.join(out_names)})")
    print("    — they publish names, not numbers, so this is the one thing worth asking about")
    print("  · column order / key names / class-id base (0 vs 1)")
    print("  · whether confidence is required, and the accepted range")
    print("  · whether coordinates are pixels or normalized, absolute or top-left")
    print("  · that the count matches their expected number of test images")
    print("  · run their validator/scorer on this file if one was provided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
