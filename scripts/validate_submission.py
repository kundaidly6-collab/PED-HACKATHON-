#!/usr/bin/env python3
"""Check a submission file against the competition's rules, line by line.

The rules are explicit about what a submission must contain, and every one of those
requirements is a way to score zero: a missing image row, a comma that breaks the CSV
into three columns, a class spelled "hole", a box that rounds to zero width, a
confidence of 1.2. None of that is visible by looking at the file, and none of it is
caught by the training pipeline. So this script reads the finished file the way a grader
would and reports every rule separately:

  columns       exactly two, ID and predictions
  ids           every test image exactly once, filename spelled exactly as on disk
  keys          class, xcenter/ycenter (or x_center/y_center), width, height, confidence
  classes       Hole or Spot, nothing else
  ranges        coordinates 0..1, width/height > 0, confidence 0..1
  empties       an image with no detections is [] -- present, not omitted

Exit code is 0 only when nothing failed, so it can gate a script or a CI step.

    python scripts/validate_submission.py --csv runs/submission_ped/submission.csv \\
        --test-images dataset/raw/test/images
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = {"Hole", "Spot"}
# The rules give both spellings: the field list says x_center/y_center, the worked
# example and the sentence "xcenter and ycenter represent the center" say the plain
# form. We accept either, report which one the file used, and warn when a single file
# mixes them, because a strict reader would key on one.
KEY_STYLES = {
    "plain": {"x": "xcenter", "y": "ycenter"},
    "underscore": {"x": "x_center", "y": "y_center"},
}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def validate(path: Path, test_images: Path | None = None,
             expect_keys: str = "auto") -> tuple[list[str], list[str], dict]:
    """Return (errors, warnings, stats). Errors mean 'do not upload this'."""
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {"images": 0, "detections": 0, "empty": 0, "per_class": {}, "key_styles": {}}

    if not path.is_file():
        return ([f"{path} does not exist"], [], stats)

    # utf-8-sig: a spreadsheet round-trip can leave a BOM, which corrupts the first
    # header cell and makes "ID" not match.
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration:
        return (["the file is empty"], [], stats)
    header = [h.strip() for h in header]
    if header != ["ID", "predictions"]:
        errors.append(f"header must be exactly ['ID', 'predictions'], found {header}")

    ids: list[str] = []
    for lineno, row in enumerate(reader, start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != 2:
            errors.append(
                f"line {lineno}: {len(row)} columns, expected 2 — the predictions field must be "
                f"quoted so its internal commas stay inside one field"
            )
            continue
        img_id, raw = row[0].strip(), row[1].strip()
        ids.append(img_id)
        stats["images"] += 1

        if not raw:
            errors.append(f"line {lineno} ({img_id}): empty predictions field; use [] for no detections")
            continue
        try:
            preds = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno} ({img_id}): predictions is not valid JSON ({exc.msg})")
            continue
        if not isinstance(preds, list):
            errors.append(f"line {lineno} ({img_id}): predictions must be a JSON list, "
                          f"found {type(preds).__name__}")
            continue
        if not preds:
            stats["empty"] += 1
            continue

        for k, p in enumerate(preds):
            where = f"line {lineno} ({img_id}) detection {k}"
            if not isinstance(p, dict):
                errors.append(f"{where}: each detection must be a JSON object")
                continue

            cls = p.get("class")
            if cls not in CLASSES:
                hint = ""
                if isinstance(cls, str) and cls.strip().lower() in {c.lower() for c in CLASSES}:
                    hint = f" — spelling/case must be exactly 'Hole' or 'Spot'"
                elif cls in (0, 1):
                    hint = (" — this format wants the class NAME, not the training id "
                            "(the ids are only used inside the repo)")
                errors.append(f"{where}: invalid class {cls!r}{hint}")
            elif isinstance(cls, str):
                stats["per_class"][cls] = stats["per_class"].get(cls, 0) + 1
            stats["detections"] += 1

            # which spelling of the centre keys does this detection use?
            style = None
            for name, keys in KEY_STYLES.items():
                if keys["x"] in p or keys["y"] in p:
                    style = name
                    break
            if style is None:
                errors.append(f"{where}: missing centre keys — need xcenter/ycenter "
                              f"(or x_center/y_center), found {sorted(p)}")
                continue
            stats["key_styles"][style] = stats["key_styles"].get(style, 0) + 1
            if expect_keys != "auto" and style != expect_keys:
                errors.append(f"{where}: uses {style} keys, but --keys {expect_keys} was expected")
            K = KEY_STYLES[style]
            if K["x"] in p and K["y"] in p and (KEY_STYLES["plain"]["x"] in p or
                                                KEY_STYLES["underscore"]["x"] in p):
                both = {KEY_STYLES["plain"]["x"], KEY_STYLES["underscore"]["x"]}
                if both.issubset(p):
                    warnings.append(f"{where}: both key spellings present; a strict reader may "
                                    f"key on one — prefer a single style")

            fields = [("class", cls), (K["x"], p.get(K["x"])), (K["y"], p.get(K["y"])),
                      ("width", p.get("width")), ("height", p.get("height")),
                      ("confidence", p.get("confidence"))]
            missing = [n for n, v in fields if v is None]
            if missing:
                errors.append(f"{where}: missing {', '.join(missing)}")
                continue
            nonnum = [n for n, v in fields if n != "class" and not _num(v)]
            if nonnum:
                errors.append(f"{where}: non-numeric {', '.join(nonnum)}")
                continue

            cx, cy = float(p[K["x"]]), float(p[K["y"]])
            bw, bh = float(p["width"]), float(p["height"])
            conf = float(p["confidence"])

            if not (0.0 <= cx <= 1.0):
                errors.append(f"{where}: {K['x']}={cx} is outside 0..1")
            if not (0.0 <= cy <= 1.0):
                errors.append(f"{where}: {K['y']}={cy} is outside 0..1")
            if not (0.0 < bw <= 1.0):
                errors.append(f"{where}: width={bw} must be greater than 0 and at most 1")
            if not (0.0 < bh <= 1.0):
                errors.append(f"{where}: height={bh} must be greater than 0 and at most 1")
            if not (0.0 <= conf <= 1.0):
                errors.append(f"{where}: confidence={conf} is outside 0..1")

            if 0.0 < bw <= 1.0 and not (0.0 <= cx - bw / 2 and cx + bw / 2 <= 1.0):
                warnings.append(f"{where}: box extends past the left/right edge "
                                f"(x {cx - bw / 2:.3f}..{cx + bw / 2:.3f})")
            if 0.0 < bh <= 1.0 and not (0.0 <= cy - bh / 2 and cy + bh / 2 <= 1.0):
                warnings.append(f"{where}: box extends past the top/bottom edge "
                                f"(y {cy - bh / 2:.3f}..{cy + bh / 2:.3f})")

    # --- uniqueness and test-set agreement -------------------------------------------
    dupes = {i for i in ids if ids.count(i) > 1} if len(ids) != len(set(ids)) else set()
    for d in sorted(dupes):
        errors.append(f"ID {d!r} appears more than once; each test image must appear exactly once")

    if test_images is not None:
        tdir = Path(test_images)
        if not tdir.is_dir():
            errors.append(f"--test-images {tdir} is not a directory")
        else:
            on_disk = sorted(p.name for p in tdir.rglob("*")
                             if p.is_file() and p.suffix.lower() in IMG_EXTS)
            stats["test_images"] = len(on_disk)
            if not on_disk:
                errors.append(f"no images found under {tdir}")
            have = set(ids)
            want = set(on_disk)
            for m in sorted(want - have)[:10]:
                errors.append(f"missing row for test image {m!r}")
            if len(want - have) > 10:
                errors.append(f"...and {len(want - have) - 10} more missing rows")
            for e in sorted(have - want)[:10]:
                errors.append(f"ID {e!r} is not a file in the test folder")
            if len(have - want) > 10:
                errors.append(f"...and {len(have - want) - 10} more unknown IDs")
            if want and have == want and stats["images"] == len(on_disk):
                stats["coverage"] = "exact"
    return errors, warnings, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a submission against the competition rules.")
    ap.add_argument("--csv", required=True, help="the submission file to check")
    ap.add_argument("--test-images", default=None,
                    help="the test image folder, to verify every image appears exactly once")
    ap.add_argument("--keys", default="auto", choices=["auto", "plain", "underscore"],
                    help="expected key spelling for the centre coordinates")
    args = ap.parse_args()

    path = Path(args.csv)
    errors, warnings, st = validate(path, Path(args.test_images) if args.test_images else None, args.keys)

    print(f"=== validating {path} ===")
    print(f"  rows             : {st['images']}")
    if "test_images" in st:
        print(f"  test images      : {st['test_images']}   coverage: {st.get('coverage', 'INCOMPLETE')}")
    print(f"  detections       : {st['detections']}   images with none: {st['empty']}")
    if st["per_class"]:
        print(f"  per class        : " + ", ".join(f"{k}={v}" for k, v in sorted(st["per_class"].items())))
    if st["key_styles"]:
        style = ", ".join(f"{k} ({v} detections)" for k, v in sorted(st["key_styles"].items()))
        print(f"  centre key style : {style}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}) — legal, but worth a look:")
        for w in warnings[:20]:
            print(f"  ! {w}")
        if len(warnings) > 20:
            print(f"  ! ...and {len(warnings) - 20} more")
    if errors:
        print(f"\nERRORS ({len(errors)}) — do NOT upload this file:")
        for e in errors[:40]:
            print(f"  x {e}")
        if len(errors) > 40:
            print(f"  x ...and {len(errors) - 40} more")
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
