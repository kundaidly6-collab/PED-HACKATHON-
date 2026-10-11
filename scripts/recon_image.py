#!/usr/bin/env python3
"""Single-image reconnaissance: measure a real component before annotating it.

Purpose: when the first real image arrives, answer "what am I actually looking at?"
with numbers instead of impressions — how big the frame is, how big the targets are,
and how many distinct *appearances* each class has.

It proposes CANDIDATE regions. It does not produce labels. Feeding these straight into
training is exactly the shortcut the plan forbids: a dark circular region is a hole only
if an annotator says so, and `--candidates` output is for a human to confirm, move or
delete. The overlay it draws is an annotation aid, not ground truth.

    python scripts/recon_image.py path/to/part.png --out runs/recon/part.png
    python scripts/recon_image.py dataset/raw/images/*.png --summary
    python scripts/recon_image.py part.png --candidates runs/recon/part_candidates.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        from PIL import Image

        with Image.open(path) as im:
            img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def estimate_background(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Background colour + foreground mask + background fraction.

    The backdrop and a light metal part can differ by only a few percent of intensity,
    so a hard-coded distance threshold either swallows the part or counts the whole
    frame as part. Otsu on the distance-to-backdrop map picks the split from the image
    itself. Border strips (not just corners) are used because a part can sit near one
    edge, and shadows live at the edges.
    """
    import cv2

    h, w = img.shape[:2]
    band = max(4, int(0.015 * min(h, w)))
    strips = np.concatenate([
        img[:band].reshape(-1, 3), img[-band:].reshape(-1, 3),
        img[:, :band].reshape(-1, 3), img[:, -band:].reshape(-1, 3),
    ])
    bg = np.median(strips, axis=0)
    dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr, 12)  # never go below a sane noise floor
    mask = (d8 > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return bg, mask, float((mask == 0).mean())


def part_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, np.ndarray]:
    """Bounding box of the largest solid non-background region (the component)."""
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.02 * mask.size:
        return None, mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [big], -1, 255, -1)   # holes inside the part count as part
    return cv2.boundingRect(big), filled


def blob_candidates(mask: np.ndarray, min_area: float, max_area: float, circularity: float = 0.55):
    """Circular blobs in a binary mask -> [(cx, cy, r, area, circ)]."""
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < circularity:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        out.append((float(cx), float(cy), float(r), float(area), float(circ)))
    return out


def compact_blob_candidates(mask: np.ndarray, min_area: float, max_area: float,
                           min_solidity: float = 0.55, max_aspect: float = 3.0):
    """Blobs that are FILLED and roughly equant — i.e. not a panel edge or a rib line.

    A long edge response and an isolated weld both light up a black-hat filter; the
    difference is shape, so this filters on solidity (area / convex-hull area) and
    bounding-box aspect ratio. Without it the weld detector reports panel corners.
    """
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        sol = area / ha if ha > 0 else 0.0
        if sol < min_solidity:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) / max(1, min(bw, bh)) > max_aspect:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        out.append((float(cx), float(cy), float(r), float(area), float(sol)))
    return out


def dedupe(cands: list, tol: float = 0.6) -> list:
    """Drop near-duplicates that both blackhat and tophat can fire on."""
    kept: list = []
    for c in sorted(cands, key=lambda t: -t[2]):
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > tol * max(c[2], k[2]) for k in kept):
            kept.append(c)
    return kept


def classify_appearance(img: np.ndarray, cx: float, cy: float, r: float) -> tuple[str, dict]:
    """What does the INSIDE of this circle look like?

    This matters more than it sounds. The same physical class ('Hole') can present as a
    bright through-hole against the backdrop, a dark shadowed bore, or a coloured bore
    showing whatever is behind the part. A detector trained on only one of those
    appearances learns that appearance, not the object.
    """
    h, w = img.shape[:2]
    rr = max(2.0, r * 0.55)
    x0, x1 = max(0, int(cx - rr)), min(w, int(cx + rr) + 1)
    y0, y1 = max(0, int(cy - rr)), min(h, int(cy + rr) + 1)
    if x1 <= x0 or y1 <= y0:
        return "unknown", {}
    patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr
    if inside.sum() < 4:
        patch = patch
        b, g, r_ = patch.mean(0)
    else:
        px = img[y0:y1, x0:x1][inside].astype(np.float32)
        b, g, r_ = px.mean(0)
    import cv2

    hsv = cv2.cvtColor(np.uint8([[[b, g, r_]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat, val = float(hsv[0]), float(hsv[1]), float(hsv[2])
    info = {"bgr": (round(float(b), 1), round(float(g), 1), round(float(r_), 1)), "hsv": (round(hue), round(sat), round(val))}

    if val > 235 and sat < 30:
        label = "bright/through (backdrop visible)"
    elif sat >= 45 and (hue <= 30 or hue >= 165) and val > 60:
        label = "coloured bore (substrate visible)"
    elif val < 120:
        label = "dark bore (shadowed)"
    else:
        label = "mid-tone bore"
    return label, info


def rim_contrast(gray: np.ndarray, cx: float, cy: float, r: float, floor: float = 10.0) -> bool:
    """Mean |inside - outside| across a circle's rim.

    A bore, a stud or a weld nugget all put a visible edge in the image. A spurious
    Hough circle lands on smooth shading and has almost no step. This is the cheapest
    way to separate the two without hand-tuning an accumulator threshold per image.
    """
    h, w = gray.shape[:2]
    rr = max(2, int(round(r)))
    x0, x1 = max(0, int(cx - 1.45 * rr)), min(w, int(cx + 1.45 * rr) + 1)
    y0, y1 = max(0, int(cy - 1.45 * rr)), min(h, int(cy + 1.45 * rr) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    inside = patch[d <= 0.6 * rr]
    outside = patch[(d >= 1.08 * rr) & (d <= 1.4 * rr)]
    if inside.size < 3 or outside.size < 3:
        return False
    return abs(float(inside.mean()) - float(outside.mean())) >= floor


def radial_profile(gray: np.ndarray, cx: float, cy: float, r: float) -> tuple[float, str]:
    """Compare the core with a ring at 0.75r.

    A through-bore is fairly uniform inside; a weld nugget / stud / ring has a centre
    that differs from its own rim. This is a hint for the annotator, not a decision.
    """
    import cv2

    h, w = gray.shape[:2]
    rr = max(3, int(round(r)))
    x0, x1 = max(0, int(cx - rr)), min(w, int(cx + rr) + 1)
    y0, y1 = max(0, int(cy - rr)), min(h, int(cy + rr) + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0, "unmeasured"
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    core = patch[d <= max(1.0, 0.35 * r)]
    rim = patch[(d > 0.68 * r) & (d <= 0.85 * r)]
    if core.size == 0 or rim.size == 0:
        return 0.0, "unmeasured"
    diff = float(rim.mean() - core.mean())
    if abs(diff) < 12:
        return diff, "uniform interior (bore-like)"
    return diff, ("bright rim / dark core (ring- or nugget-like)" if diff > 0
                  else "dark rim / bright core (raised feature)")


def recon(path: Path, out_path: Path | None = None, cand_path: Path | None = None) -> dict:
    import cv2

    img = load_rgb(path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    bg, fg_mask, bg_frac = estimate_background(img)
    bbox, part_mask = part_bbox(fg_mask)

    print(f"\n=== {path.name} ===")
    print(f"  size          : {w}×{h} px ({img.shape[2]} channels)")
    print(f"  background    : BGR {tuple(int(v) for v in bg)} covering ~{bg_frac * 100:.0f}% of the frame")
    if bbox:
        x, y, bw, bh = bbox
        print(f"  component     : {bw}×{bh} px at ({x},{y}), i.e. {100 * bw / w:.0f}% × {100 * bh / h:.0f}% of the frame")
    else:
        print("  component     : not isolated (busy background or the part fills the frame)")

    # Holes are compact blobs that are either darker OR brighter than their
    # surroundings, and they come in a range of sizes -- a single structuring-element
    # size only responds to features smaller than itself, which silently drops the
    # bigger holes. So sweep kernel sizes and union the results, then de-duplicate.
    import cv2 as _cv2

    min_area = max(20.0, (min(h, w) / 200.0) ** 2 * 8)
    max_area = (min(h, w) / 6.0) ** 2
    span = int(min(h, w) / 3)
    sizes = sorted({max(9, int(round(v))) | 1 for v in
                    (span * f for f in (0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.5))})
    cands: list = []
    for k in sizes:
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
        for resp in (_cv2.morphologyEx(gray, _cv2.MORPH_BLACKHAT, kernel),
                     _cv2.morphologyEx(gray, _cv2.MORPH_TOPHAT, kernel)):
            floor = max(resp.mean() + 2.5 * resp.std(), 8.0)
            m = (resp > floor).astype(np.uint8) * 255
            m = _cv2.morphologyEx(m, _cv2.MORPH_CLOSE, _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3)))
            cands.extend(blob_candidates(m, min_area, max_area))
    # Second, independent detector. Morphology misses a blob when a large/low-frequency
    # response swallows it during de-duplication; Hough is strong exactly where
    # morphology is weak (well-formed circular rims). Union the two — recall matters
    # more than tidiness here, because a missed hole is a missed annotation.
    hough: list = []
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
                # Accept only circles with a real brightness step across the rim. A Hough
                # circle can sit on a smooth gradient or a panel edge; a genuine bore
                # shows a discontinuity between its inside and the material around it.
                if not rim_contrast(gray, cx, cy, r, floor=10.0):
                    continue
                hough.append((float(cx), float(cy), float(r), float(np.pi * r * r), 1.0))
    except Exception as exc:
        print(f"  (hough detector unavailable: {exc})")

    hough_new = [c for c in hough if not any(np.hypot(c[0] - k[0], c[1] - k[1]) < 0.7 * max(c[2], k[2]) for k in cands)]
    print(f"  (swept {len(sizes)} template sizes {sizes}: {len(cands)} blobs; "
          f"+{len(hough_new)} more from a Hough pass)")
    cands = dedupe(cands + hough_new)

    # Position is the one discriminator geometry can actually offer here: mounting
    # holes sit in the perimeter flange, welds/studs sit in the recessed panel. A round
    # feature in the middle could be either, so the report says which is which and
    # refuses to pretend it knows.
    band = 0.18
    if bbox:
        bx, by, bbw, bbh = bbox
        def zone(cx, cy):
            inner_x = bx + band * bbw <= cx <= bx + (1 - band) * bbw
            inner_y = by + band * bbh <= cy <= by + (1 - band) * bbh
            return "interior" if (inner_x and inner_y) else "perimeter"
    else:
        def zone(cx, cy):
            return "unknown"

    appearances: Counter = Counter()
    rows = []
    for (cx, cy, r, area, circ) in sorted(cands, key=lambda t: (round(t[1] / 40), t[0])):
        label, info = classify_appearance(img, cx, cy, r)
        diff, shape = radial_profile(gray, cx, cy, r)
        appearances[label] += 1
        rows.append({"x": round(cx), "y": round(cy), "d_px": round(2 * r, 1), "circ": round(circ, 2),
                     "appearance": label, "inside": info, "zone": zone(cx, cy), "shape": shape,
                     "rim_minus_core": round(diff, 1)})

    perim = [r0 for r0 in rows if r0["zone"] == "perimeter"]
    inner_c = [r0 for r0 in rows if r0["zone"] == "interior"]
    diam = [r["d_px"] for r in rows]
    print(f"\n  circular candidates: {len(rows)}  ({len(perim)} in the perimeter band, {len(inner_c)} toward the centre)")
    if diam:
        ds = sorted(diam)
        print(f"    diameter  : min {ds[0]:.0f}px · median {ds[len(ds) // 2]:.0f}px · max {ds[-1]:.0f}px")
        print(f"    as % of frame width : min {100 * ds[0] / w:.2f}% · median {100 * ds[len(ds) // 2] / w:.2f}%")
        px640 = ds[len(ds) // 2] * 640 / w
        print(f"    at imgsz 640 a {ds[len(ds) // 2]:.0f}px circle becomes ~{px640:.0f}px  "
              f"({'below' if px640 < 12 else 'above'} the stride-32 floor)")
    print("    appearance breakdown (the part that decides your augmentation + labels):")
    for label, n in appearances.most_common():
        print(f"      {n:3d} × {label}")
    if len(appearances) > 1:
        print("    >>> the SAME class appears in several guises. Check that your labels cover all of")
        print("        them, and that your training set does not contain only one appearance.")

    # Non-circular, small, high-contrast features inside the panel are the spot-weld
    # candidates. They are reported separately because they are NOT circles.
    if bbox:
        x, y, bw, bh = bbox
        inner = np.zeros_like(gray)
        inner[y + int(0.08 * bh):y + int(0.92 * bh), x + int(0.08 * bw):x + int(0.92 * bw)] = 255
        # A weld/stud is an isolated compact feature, typically ~10-40 px wide. The
        # same rule applies as for holes: the structuring element must be LARGER than
        # the feature, so sweep sizes instead of guessing one.
        small = (min_area * 0.5, (min(h, w) / 10.0) ** 2)
        sspan = int(min(h, w) / 8)
        ssizes = sorted({max(9, int(round(v))) | 1 for v in
                         (sspan * f for f in (0.5, 0.75, 1.1, 1.6, 2.2))})
        found: list[tuple] = []
        for k2 in ssizes:
            ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
            bh_resp = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker2)
            th_resp = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker2)
            for resp in (bh_resp, th_resp):
                floor = max(resp.mean() + 2.2 * resp.std(), 10.0)
                m = ((resp > floor) & (inner > 0) & (part_mask > 0)).astype(np.uint8) * 255
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                for (cx, cy, r, area, sol) in compact_blob_candidates(m, small[0], small[1]):
                    found.append((cx, cy, r, area, sol, float(resp[int(cy), int(cx)])))
        # Only exclude features that sit on a PERIMETER circle (an unambiguous mounting
        # hole). A round feature in the panel centre is kept on purpose: that is exactly
        # the hole-vs-weld ambiguity an annotator has to resolve, and silently choosing
        # one interpretation is how a wrong label gets made.
        perim_rows = [r0 for r0 in rows if r0["zone"] == "perimeter"]
        found = [f0 for f0 in found
                 if not any(np.hypot(f0[0] - r0["x"], f0[1] - r0["y"]) < 1.4 * (r0["d_px"] / 2 + f0[2])
                            for r0 in perim_rows)]
        # collapse multi-scale repeats, keeping the strongest response
        spots: list = []
        for f0 in sorted(found, key=lambda t: -t[5]):
            if all(np.hypot(f0[0] - k[0], f0[1] - k[1]) > 0.8 * max(f0[2], k[2]) for k in spots):
                spots.append(f0)
        # one entry per feature, not one per detector
        merged: list[dict] = []
        for r0 in inner_c:
            merged.append({"x": r0["x"], "y": r0["y"], "w": r0["d_px"], "round": True,
                           "appearance": r0["appearance"], "shape": r0["shape"], "src": "circular"})
        for (cx, cy, r, area, sol, strength) in spots[:10]:
            label, _ = classify_appearance(img, cx, cy, r)
            _d, shape = radial_profile(gray, cx, cy, r)
            if any(np.hypot(cx - m0["x"], cy - m0["y"]) < 0.8 * max(2 * r, m0["w"]) for m0 in merged):
                continue
            merged.append({"x": round(cx), "y": round(cy), "w": round(2 * r, 1), "round": sol > 0.8,
                           "appearance": label, "shape": shape, "src": f"solidity {sol:.2f} response {strength:.0f}"})
        print(f"\n  Spot/weld candidates (compact features toward the panel centre): {len(merged)}")
        print("    Ranked 'look here' pointers — LOW PRECISION. A ROUND feature in the panel is")
        print("    exactly the hole-vs-weld call that only an annotator can make.")
        for m0 in merged:
            kind = "ROUND  → Hole OR weld nugget?" if m0["round"] else "irregular shape"
            print(f"    ({m0['x']},{m0['y']})  ~{m0['w']:.0f}px wide  {kind}  | {m0['appearance']} | {m0['shape']}")

    if out_path:
        vis = img.copy()
        for i, r0 in enumerate(rows, 1):
            cv2.circle(vis, (int(r0["x"]), int(r0["y"])), int(r0["d_px"] / 2), (0, 220, 255), 2)
            cv2.putText(vis, str(i), (int(r0["x"]) - 8, int(r0["y"]) - int(r0["d_px"] / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (120, 120, 120), 1)
            cv2.putText(vis, "component", (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY if out_path.suffix.lower() in {".jpg", ".jpeg"} else cv2.IMWRITE_PNG_COMPRESSION, 90 if out_path.suffix.lower() in {".jpg", ".jpeg"} else 3])
        print(f"\n  overlay       : {out_path}   ← CANDIDATES ONLY, verify before labeling")

    if cand_path and rows:
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# CANDIDATE boxes from recon_image.py — NOT labels. Verify each one.",
                 "# cls cx cy w h  (normalized, class 1 = Hole as a first guess)"]
        for r0 in rows:
            d = r0["d_px"]
            lines.append(f"1 {r0['x'] / w:.6f} {r0['y'] / h:.6f} {d / w:.6f} {d / h:.6f}  # {r0['appearance']}")
        cand_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  candidates    : {cand_path}")

    return {"path": str(path), "size": [w, h], "background_fraction": round(bg_frac, 3),
            "component_bbox": list(bbox) if bbox else None, "circular_candidates": rows,
            "appearance_counts": dict(appearances)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure a component image before annotating it.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=None, help="overlay path (single image only)")
    ap.add_argument("--candidates", default=None, help="candidate box file (single image only)")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--summary", action="store_true", help="one line per image")
    args = ap.parse_args()

    paths: list[Path] = []
    for spec in args.images:
        p = Path(spec)
        if p.is_dir():
            paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS)
        else:
            paths.append(p)
    if not paths:
        print("[ERROR] no images given", file=sys.stderr)
        return 2

    reports = []
    for p in paths:
        if not p.is_file():
            print(f"[skip] not found: {p}")
            continue
        try:
            rep = recon(p, Path(args.out) if (args.out and len(paths) == 1) else None,
                        Path(args.candidates) if (args.candidates and len(paths) == 1) else None)
            reports.append(rep)
        except Exception as exc:
            print(f"[ERROR] {p}: {type(exc).__name__}: {exc}")

    if args.summary or len(reports) > 1:
        print("\n=== summary ===")
        for r in reports:
            print(f"  {Path(r['path']).name:<28} {r['size'][0]}×{r['size'][1]}  "
                  f"circles={len(r['circular_candidates'])}  "
                  f"appearances={list(r['appearance_counts'])}")
    if args.json:
        import json

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nreport: {args.json}")

    print("\nREMINDER: these are proposals for an annotator, not labels. Confirm or correct each")
    print("one before it becomes training data — a wrong box teaches a confident wrong lesson.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
