#!/usr/bin/env python3
"""Pre-training label gate for dataset/prepared (the boring checks that catch real bugs).

Runs against the BUILT dataset, because that is the thing training will actually read:
  * every image has exactly one same-stem .txt, and no orphan labels
  * every line parses, class ids are in range, coordinates are normalized + in bounds
  * no box is degenerate after rounding
  * box counts agree with manifest.json (a converter bug shows up right here)
  * writes a contact sheet with boxes drawn — nobody catches a swapped Spot/Hole
    mapping from a table of numbers, but it is trivially visible in an image grid

    python scripts/validate_labels.py                 # both splits
    python scripts/validate_labels.py --split val --sheet 24
Exit code 1 = fix the labels before training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw  # noqa: E402

from _ped.labels import CLASS_NAMES, MIN_BOX_PIXELS, parse_yolo_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "dataset" / "prepared"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = {0: (0, 220, 255), 1: (255, 60, 60)}  # Spot cyan, Hole red


def check_split(prepared: Path, split: str) -> tuple[int, Counter, list[str]]:
    img_dir, lbl_dir = prepared / "images" / split, prepared / "labels" / split
    problems: list[str] = []
    counts: Counter = Counter()
    if not img_dir.is_dir():
        return 0, counts, [f"missing directory {img_dir}"]

    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    labels = {p.stem: p for p in (lbl_dir.glob("*.txt") if lbl_dir.is_dir() else [])}
    stems = Counter(p.stem for p in images)
    dupes = [s for s, c in stems.items() if c > 1]
    for s in dupes:
        problems.append(f"{split}: two images share the stem {s!r} — one label cannot serve both")
    for stem in sorted(set(stems) - set(labels)):
        problems.append(f"no label file for {stem} (a missing label is not a negative sample)")
    for stem in sorted(set(labels) - set(stems)):
        problems.append(f"orphan label {stem}.txt with no image")

    n_boxes = 0
    for img in images:
        lbl = labels.get(img.stem)
        if lbl is None:
            continue
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception as exc:
            problems.append(f"unreadable image {img.name}: {exc}")
            continue
        boxes, findings = parse_yolo_text(lbl.read_text(encoding="utf-8", errors="replace"), lbl.name)
        problems.extend(f.format() for f in findings)
        for b in boxes:
            n_boxes += 1
            counts[CLASS_NAMES[b.cls]] += 1
            bw, bh = b.w * w, b.h * h
            if bw < MIN_BOX_PIXELS or bh < MIN_BOX_PIXELS:
                problems.append(f"{img.name}: degenerate box {bw:.1f}×{bh:.1f}px on a {w}×{h} image")
            if b.cx - b.w / 2 < -0.01 or b.cy - b.h / 2 < -0.01 or b.cx + b.w / 2 > 1.01:
                problems.append(f"{img.name}: box extends materially outside the image")
    return n_boxes, counts, problems


def make_sheet(prepared: Path, split: str, n: int, out: Path, seed: int = 7) -> Path | None:
    """Contact sheet with boxes overlaid — the human-in-the-loop check."""
    img_dir = prepared / "images" / split
    if not img_dir.is_dir():
        return None
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None
    picks = random.Random(seed).sample(imgs, min(n, len(imgs)))
    tw, th, pad = 320, 240, 22
    cols = min(4, len(picks)) or 1
    rows = -(-len(picks) // cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + pad)), (16, 16, 18))
    for i, path in enumerate(picks):
        with Image.open(path) as im:
            im = im.convert("RGB")
            s = min(tw / im.width, th / im.height)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        ox, oy = (tw - im.width) // 2, (th - im.height) // 2
        bx, by = (i % cols) * tw, (i // cols) * (th + pad)
        sheet.paste(im, (bx + ox, by + oy))
        d = ImageDraw.Draw(sheet)
        cnt = Counter()
        lbl = prepared / "labels" / split / f"{path.stem}.txt"
        if lbl.is_file():
            for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x0, y0 = bx + ox + (cx - bw / 2) * im.width, by + oy + (cy - bh / 2) * im.height
                d.rectangle([x0, y0, x0 + bw * im.width, y0 + bh * im.height], outline=COLORS.get(cls, (255, 255, 255)), width=2)
                cnt[cls] += 1
        tag = " ".join(f"{CLASS_NAMES[c]}×{k}" for c, k in sorted(cnt.items())) or "no objects"
        d.text((bx + 4, by + th + 5), f"{path.name[:28]} · {tag}", fill=(235, 235, 235))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def check_class_names() -> list[str]:
    """Does the dataset yaml name the classes in the SAME order the labels imply?

    This is the quietest way to lose the whole competition. Every label file holds bare
    integers (0 = Spot, 1 = Hole in this repo). A yaml that declares the names in the
    other order -- ``names: ["Hole", "Spot"]``, which is what a lot of outside tutorials
    and starter scripts use -- renames every box without touching a single label file.
    The model then learns Hole features as Spot and vice versa, the training curves look
    healthy, and validation mAP looks plausible, because the metric is computed against
    the same swapped interpretation. Nothing shows it except comparing the two lists.
    """
    try:
        from _ped.dataset_cfg import find_data_yaml, load_yaml
    except Exception:
        return []
    yml = find_data_yaml(ROOT)
    if not yml:
        return []
    try:
        cfg = load_yaml(yml)
    except Exception as exc:
        return [f"{yml.name} could not be parsed: {exc}"]
    names = cfg.get("names")
    if names is None:
        return [f"{yml.name} has no 'names:' list, so the class order is undefined"]
    if isinstance(names, dict):                       # {0: 'Spot', 1: 'Hole'} is legal YAML
        names = [names[k] for k in sorted(names)]
    if [str(n) for n in names] != list(CLASS_NAMES):
        return [
            f"{yml.name} declares names {list(names)}, but the labels use "
            f"{list(CLASS_NAMES)} (0={CLASS_NAMES[0]}, 1={CLASS_NAMES[1]}). "
            f"Every class would be read as the OPPOSITE one. Fix the yaml or the labels — "
            f"do not train until they agree."
        ]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=str(PREPARED))
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--sheet", type=int, default=16, help="contact-sheet size per split (0 disables)")
    ap.add_argument("--max-problems", type=int, default=25)
    args = ap.parse_args()

    prepared = Path(args.prepared)
    if not prepared.is_dir():
        print(f"[ERROR] {prepared} does not exist — run prepare_dataset.py first.", file=sys.stderr)
        return 2

    manifest = {}
    if (prepared / "manifest.json").is_file():
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))

    names_problems = check_class_names()
    if names_problems:
        print("=== class names vs labels ===")
        for prob in names_problems:
            print(f"  [FAIL] {prob}")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    total = len(names_problems)
    for split in splits:
        n_boxes, counts, problems = check_split(prepared, split)
        print(f"\n=== {split} ===")
        print(f"  boxes: {n_boxes}  per class: {dict(counts)}")
        by_split = (manifest.get("boxes_by_split") or {}).get(split)
        claimed = by_split if by_split is not None else (
            manifest.get("boxes_per_class") if len(splits) == 1 else None
        )
        if claimed:
            keys = set(claimed) | set(counts)
            mismatch = {k: (claimed.get(k, 0), counts.get(k, 0)) for k in keys if claimed.get(k, 0) != counts.get(k, 0)}
            if mismatch:
                print(f"  [FAIL] manifest claims {claimed}, files contain {dict(counts)} — converter disagreement")
                problems.append(f"manifest/box-count disagreement on {split}: {mismatch}")
            else:
                print(f"  [OK] matches manifest.json {claimed}")
        if problems:
            print(f"  [FAIL] {len(problems)} problem(s):")
            for p in problems[: args.max_problems]:
                print(f"    · {p}")
            if len(problems) > args.max_problems:
                print(f"    ... {len(problems) - args.max_problems} more")
        else:
            print("  [OK] structure valid")
        total += len(problems)

    if args.sheet:
        for split in splits:
            out = ROOT / "runs" / f"label_sheet_{split}.jpg"
            sheet = make_sheet(prepared, split, args.sheet, out)
            if sheet:
                print(f"\n  contact sheet: {sheet}  ← LOOK at it. Cyan=Spot, red=Hole.")

    if total:
        print(f"\nRESULT: {total} problem(s) — fix labels before training.")
        return 1
    print("\nRESULT: labels valid — safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
