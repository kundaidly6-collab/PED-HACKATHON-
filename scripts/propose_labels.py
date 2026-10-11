#!/usr/bin/env python3
"""Batch pre-annotation: propose candidate boxes for every training image.

This competition ships **unlabelled** training images, so annotation is on the critical
path — nothing else can run until boxes exist. This script does not replace annotating;
it removes most of the *drawing* so the human effort goes into the part only a human can
do: deciding what each feature actually is.

What it proposes, and why:

  Hole (class 1)  circular features. Class rules: Nut -> Hole, opening -> Hole.
                  Mounting holes sit in the perimeter flange; bores read as a brightness
                  discontinuity across a circular rim. Two independent detectors
                  (multi-scale morphology + Hough) are unioned for recall.
  Spot (class 0)  compact features inside the panel. Class rules: Bolt -> Spot,
                  weld spot -> Spot. Isolated, filled, roughly equant blobs.

Everything it writes is a DRAFT. The class call on a round feature inside the panel is a
judgement call the tool refuses to make silently, so those are tagged `ambiguous` in the
sidecar and highlighted for review. Confirm or delete each one in scripts/annotate.py.

    python scripts/propose_labels.py --images dataset/raw/train/images
    python scripts/propose_labels.py --images ... --out dataset/drafts --limit 20
    python scripts/propose_labels.py --images ... --sheet runs/drafts_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recon_image import (  # noqa: E402
    IMG_EXTS,
    classify_appearance,
    compact_blob_candidates,
    dedupe,
    edge_distance_map,
    estimate_background,
    load_rgb,
    merge_contained,
    part_bbox,
    radial_profile,
    rim_contrast,
)

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Spot", "Hole"]  # index == class id. 0=Spot(weld,bolt), 1=Hole(nut,opening)


def detect(img: np.ndarray, gray: np.ndarray, verbose: bool = False) -> tuple[list[dict], dict]:
    """Candidate boxes as dicts: {cls, cx, cy, w, h, kind, confidence, note} in pixels."""
    import cv2

    h, w = img.shape[:2]
    _bg, fg_mask, _bg_frac = estimate_background(img)
    bbox, part_mask = part_bbox(fg_mask)

    min_area = max(20.0, (min(h, w) / 200.0) ** 2 * 8)
    max_area = (min(h, w) / 6.0) ** 2
    span = int(min(h, w) / 3)
    sizes = sorted({max(9, int(round(v))) | 1 for v in
                    (span * f for f in (0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.5))})

    # ---- circular features: two independent detectors, unioned for recall ----
    from recon_image import blob_candidates

    raw: list = []
    for k in sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        for resp in (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
                     cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)):
            floor = max(resp.mean() + 2.5 * resp.std(), 8.0)
            m = (resp > floor).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            raw.extend(blob_candidates(m, min_area, max_area))
    try:
        blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
        rmin = max(4, int(min(h, w) / 220))
        rmax = max(rmin + 4, int(min(h, w) / 7))
        circ = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                minDist=max(8, int(min(h, w) / 45)),
                                param1=110, param2=26, minRadius=rmin, maxRadius=rmax)
        if circ is not None:
            for (cx, cy, r) in np.round(circ[0]).astype(float):
                if not (0 <= int(cy) < h and 0 <= int(cx) < w) or r < 2:
                    continue
                if rim_contrast(gray, cx, cy, r, floor=10.0):
                    raw.append((float(cx), float(cy), float(r), float(np.pi * r * r), 1.0))
    except Exception:
        pass

    circles = dedupe(raw)
    circles, n_merged = merge_contained(circles)
    if n_merged:
        print(f"    (collapsed {n_merged} nested circle(s) — big loose circle over a tight one)")

    # Manufacturing consistency: the holes on one component are drilled by the same
    # tooling, so their radii cluster tightly. A circle 2x+ off the median is almost
    # always a detector artefact (a loose blob over a boss or shading), and it wastes
    # annotator time. Flagged as an outlier rather than deleted — nothing is removed
    # silently, and the note says exactly why it is suspicious.
    radii = sorted(float(c[2]) for c in circles)
    median_r = radii[len(radii) // 2] if radii else 0.0
    outliers = 0

    # Zone by DISTANCE TO THE PART EDGE, not by a percentage rectangle. Mounting holes
    # live in the blank outer flange (close to the outline); welds and studs live inside
    # the recess (far from it). A rectangular inset misfires on real parts because the
    # recess is not a uniform border — a weld near the recess floor gets called a
    # perimeter hole, which is exactly the mistake that would corrupt a label.
    dist = edge_distance_map(part_mask)
    ref = float(min(bbox[2], bbox[3])) if bbox else float(min(w, h))
    PERIM_FRAC, INNER_FRAC = 0.085, 0.115

    def zone(cx, cy):
        xi, yi = int(round(cx)), int(round(cy))
        if not (0 <= yi < dist.shape[0] and 0 <= xi < dist.shape[1]):
            return "unknown"
        d = float(dist[yi, xi]) / max(ref, 1.0)
        if d <= PERIM_FRAC:
            return "perimeter"
        if d >= INNER_FRAC:
            return "interior"
        return "boundary"

    boxes: list[dict] = []
    for (cx, cy, r, _area, _circ) in sorted(circles, key=lambda t: (round(t[1] / 40), t[0])):
        appearance, _info = classify_appearance(img, cx, cy, r)
        _d, shape = radial_profile(gray, cx, cy, r)
        z = zone(cx, cy)
        if median_r > 0 and len(circles) >= 4 and (r > 2.2 * median_r or r < 0.45 * median_r):
            outliers += 1
            note = (f"SIZE OUTLIER vs the other {len(circles)} circles (median dia "
                    f"{2 * median_r:.0f}px, this one {2 * r:.0f}px) — likely spurious, check first")
            conf = "low"
        elif z == "perimeter":
            note, conf = "mounting hole in the flange band", "high"
        elif z == "interior":
            # Nut -> Hole and weld spot -> Spot both land here. Only a human can call it.
            note, conf = "ROUND feature inside the panel — Nut/Hole or weld/Spot?", "ambiguous"
        elif z == "boundary":
            note, conf = "circular feature at the flange/recess boundary — check which it is", "ambiguous"
        else:
            note, conf = "circular feature, zone unknown", "low"
        boxes.append({"cls": 1, "cx": float(cx), "cy": float(cy),
                      "w": float(2 * r), "h": float(2 * r),
                      "kind": "circle", "confidence": conf,
                      "note": f"{note} | {appearance} | {shape}"})

    # ---- compact non-circular features inside the panel: bolt / weld stud --------
    if bbox:
        bx, by, bbw, bbh = bbox
        inner = np.zeros_like(gray)
        inner[by + int(0.08 * bbh):by + int(0.92 * bbh), bx + int(0.08 * bbw):bx + int(0.92 * bbw)] = 255
        small = (min_area * 0.5, (min(h, w) / 10.0) ** 2)
        sspan = int(min(h, w) / 8)
        ssizes = sorted({max(9, int(round(v))) | 1 for v in
                         (sspan * f for f in (0.5, 0.75, 1.1, 1.6, 2.2))})
        found: list = []
        for k2 in ssizes:
            ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
            for resp in (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker2),
                         cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker2)):
                floor = max(resp.mean() + 2.2 * resp.std(), 10.0)
                m = ((resp > floor) & (inner > 0) & (part_mask > 0)).astype(np.uint8) * 255
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                for (cx, cy, r, area, sol) in compact_blob_candidates(m, small[0], small[1]):
                    found.append((cx, cy, r, area, sol, float(resp[int(cy), int(cx)])))
        perim = [b for b in boxes if zone(b["cx"], b["cy"]) == "perimeter"]
        found = [f0 for f0 in found
                 if not any(np.hypot(f0[0] - p["cx"], f0[1] - p["cy"]) <
                            1.4 * (p["w"] / 2 + f0[2]) for p in perim)]
        spots: list = []
        for f0 in sorted(found, key=lambda t: -t[5]):
            if all(np.hypot(f0[0] - k[0], f0[1] - k[1]) > 0.8 * max(f0[2], k[2]) for k in spots):
                spots.append(f0)
        for (cx, cy, r, _area, sol, strength) in spots:
            appearance, _info = classify_appearance(img, cx, cy, r)
            _d, shape = radial_profile(gray, cx, cy, r)
            if any(np.hypot(cx - b["cx"], cy - b["cy"]) < 0.8 * max(2 * r, b["w"]) for b in boxes):
                continue
            # A "compact feature" that touches the part outline is almost always a
            # rounded plate corner or a shadow, not a bolt or a weld. Requiring real
            # interior clearance removed the corner artefacts from the review sheet.
            xi, yi = int(round(cx)), int(round(cy))
            if not (0 <= yi < dist.shape[0] and 0 <= xi < dist.shape[1]):
                continue
            if float(dist[yi, xi]) / max(ref, 1.0) < INNER_FRAC:
                continue
            boxes.append({"cls": 0, "cx": float(cx), "cy": float(cy),
                          "w": float(2 * r), "h": float(2 * r),
                          "kind": "compact", "confidence": "medium" if sol > 0.8 else "low",
                          "note": f"bolt/stud/weld candidate (solidity {sol:.2f}, response {strength:.0f}) | {appearance} | {shape}"})
    return boxes, {"component_bbox": list(bbox) if bbox else None, "size": [w, h]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Propose draft boxes for unlabelled training images.")
    ap.add_argument("--images", required=True, help="folder of training images (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "dataset" / "drafts"), help="where draft labels go")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--sheet", default=None, help="write a contact sheet of the proposals")
    ap.add_argument("--sheet-n", type=int, default=12)
    ap.add_argument("--json", default=None, help="write the full proposal report")
    args = ap.parse_args()

    src = Path(args.images).expanduser()
    if not src.is_dir():
        print(f"[ERROR] not a directory: {src}", file=sys.stderr)
        return 2
    images = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"[ERROR] no images under {src}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("DRAFT pre-annotation — these are PROPOSALS, not ground truth.")
    print("Class rules in play:  Nut / opening -> Hole(1)    Bolt / weld spot -> Spot(0)")
    print("=" * 74)
    print(f"images : {len(images)}   from {src}")
    print(f"drafts : {out}\n")

    import cv2

    counts: Counter = Counter()
    report = []
    for i, path in enumerate(images, 1):
        try:
            img = load_rgb(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            boxes, meta = detect(img, gray)
        except Exception as exc:
            print(f"  [{i}/{len(images)}] {path.name}: FAILED {type(exc).__name__}: {exc}")
            continue

        h, w = meta["size"]
        lines = []
        for b in boxes:
            lines.append(f"{b['cls']} {b['cx'] / w:.6f} {b['cy'] / h:.6f} {b['w'] / w:.6f} {b['h'] / h:.6f}")
            counts[CLASS_NAMES[b["cls"]]] += 1
            counts[b["confidence"]] += 1
        (out / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        report.append({"image": str(path), "stem": path.stem, "size": [w, h],
                       "boxes": boxes, "n": len(boxes)})

        amb = sum(1 for b in boxes if b["confidence"] == "ambiguous")
        flag = f"  <-- {amb} need a class decision" if amb else ""
        print(f"  [{i}/{len(images)}] {path.name[:40]:40s} {len(boxes):3d} boxes{flag}")

    # ---- the review sheet: proposals drawn on the images -------------------------
    if args.sheet and report:
        picks = report[: args.sheet_n]
        cell = 340
        cols = min(3, len(picks))
        rows = -(-len(picks) // cols)
        sheet = np.full((rows * (cell + 16), cols * cell, 3), 22, np.uint8)
        for k, rec in enumerate(picks):
            im = cv2.imread(rec["image"])
            if im is None:
                continue
            h, w = im.shape[:2]
            s = min(cell / w, cell / h)
            im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
            oy, ox = (k // cols) * (cell + 16), (k % cols) * cell
            sheet[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
            for b in rec["boxes"]:
                # cv2.imwrite takes BGR, so spell out the BGR tuple for the RGB colour
                # the legend promises. Passing RGB here renders yellow and makes the
                # sheet contradict its own caption.
                col = (255, 200, 0) if b["cls"] == 1 else (60, 60, 255)    # Hole cyan, Spot red
                if b["confidence"] == "ambiguous":
                    col = (255, 0, 255)                                    # magenta = your call
                x1 = int(ox + (b["cx"] - b["w"] / 2) * s); y1 = int(oy + (b["cy"] - b["h"] / 2) * s)
                x2 = int(ox + (b["cx"] + b["w"] / 2) * s); y2 = int(oy + (b["cy"] + b["h"] / 2) * s)
                cv2.rectangle(sheet, (x1, y1), (x2, y2), col, 2)
            label = f"{Path(rec['image']).name[:26]} n={rec['n']}"
            cv2.putText(sheet, label, (ox + 4, oy + cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        sp = Path(args.sheet); sp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sp), sheet)
        print(f"\nreview sheet: {sp}")
        print("  cyan = Hole proposal · red = Spot proposal · magenta = YOU DECIDE (round feature in panel)")

    # Always drop the notes next to the drafts. The UI looks for <drafts>/notes.json by
    # default, and forgetting to pass --json previously meant every "you decide" flag
    # silently disappeared from the review sheet.
    notes_path = out / "notes.json"
    notes_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nproposals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(
        "\nNEXT (this is the step that actually creates labels):\n"
        "  python scripts/annotate.py --images "
        + str(src)
        + " --drafts "
        + str(out)
        + "\n"
        "  Review every box there: confirm, move, reclass or delete. The drafts load as a\n"
        "  starting point; nothing is a training label until you save it.\n"
        "  Then verify with:  python scripts/validate_labels.py --prepared <your dataset>"
    )
    print(f"report: {notes_path}  (annotate.py picks this up automatically)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"copy:   {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
