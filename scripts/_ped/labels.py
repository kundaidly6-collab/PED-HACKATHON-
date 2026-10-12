"""Label parsing + validation for the formats a hackathon dataset plausibly arrives in:
YOLO .txt, COCO JSON, Pascal VOC XML, and flat CSV (Kaggle-style).

Parsing is strict on purpose. Every anomaly is returned as a finding with a severity,
so `audit_dataset.py` can tell you *which* file is broken instead of letting the
trainer silently drop or, worse, half-use a malformed box.
"""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Classes for this competition. IDs are fixed here and echoed into data.yaml.
#
# THE RULES FIX THE NAMES, NOT THE NUMBERS. The competition document says only:
#     Nut              -> Hole
#     physical opening -> Hole
#     Bolt             -> Spot
#     weld spot        -> Spot
# It never states a numeric order. "Spot = 0, Hole = 1" is therefore THIS repo's
# convention: it is consistent everywhere here (data.yaml, labels, the annotator, the
# trainer) and is safe for training, because we generate both the labels and the config.
#
# It is NOT safe to assume a grader agrees. A submission that carries bare integers and
# no names is a coin flip on the ordering, and getting it wrong misclassifies every box
# and collapses the score. So: every submission format either carries the class NAME
# (csv, coco, xml) or ships a classes.txt beside it (yolo), and submit.py prints the
# mapping it used. Change the ids in exactly this one place if the organisers publish an
# order that differs.
CLASS_NAMES = ["Spot", "Hole"]
CLASS_TO_ID = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}

MIN_BOX_PIXELS = 3.0  # below this a box is almost certainly an annotation error


@dataclass
class Box:
    """A box in *normalized* YOLO order: (cls, cx, cy, w, h)."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def as_xyxy_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2) * width
        y1 = (self.cy - self.h / 2) * height
        x2 = (self.cx + self.w / 2) * width
        y2 = (self.cy + self.h / 2) * height
        return (
            max(0.0, min(x1, width)),
            max(0.0, min(y1, height)),
            max(0.0, min(x2, width)),
            max(0.0, min(y2, height)),
        )


@dataclass
class Finding:
    severity: str  # 'error' | 'warn' | 'info'
    where: str
    message: str

    def format(self) -> str:
        tag = {"error": "ERROR", "warn": "WARN ", "info": "INFO "}[self.severity]
        return f"[{tag}] {self.where}: {self.message}"


@dataclass
class ImageLabels:
    path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


# ---------------------------------------------------------------- YOLO text labels

def parse_yolo_text(text: str, where: str) -> tuple[list[Box], list[Finding]]:
    boxes: list[Box] = []
    findings: list[Finding] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            findings.append(Finding("error", where, f"line {lineno}: non-numeric token in {line!r}"))
            continue
        if len(nums) == 4:
            findings.append(
                Finding("error", where, f"line {lineno}: 4 values with no class id -> {line!r}")
            )
            continue
        if len(nums) < 5:
            findings.append(
                Finding("error", where, f"line {lineno}: expected >=5 values, got {len(nums)}")
            )
            continue
        if len(nums) > 5:
            findings.append(
                Finding("warn", where, f"line {lineno}: {len(nums)} fields; extra fields ignored")
            )
        cls_raw, cx, cy, w, h = nums[0], nums[1], nums[2], nums[3], nums[4]
        if abs(cls_raw - round(cls_raw)) > 1e-9:
            findings.append(Finding("error", where, f"line {lineno}: non-integer class id {cls_raw}"))
            continue
        cls = int(cls_raw)
        if cls < 0 or cls >= len(CLASS_NAMES):
            findings.append(
                Finding(
                    "error",
                    where,
                    f"line {lineno}: class id {cls} outside 0..{len(CLASS_NAMES) - 1} "
                    f"({', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))})",
                )
            )
            continue
        if max(abs(cx), abs(cy), abs(w), abs(h)) > 1.5:
            findings.append(
                Finding(
                    "error",
                    where,
                    f"line {lineno}: values look like PIXELS not normalized coords "
                    f"({cx:g} {cy:g} {w:g} {h:g}) — needs conversion, not training",
                )
            )
            continue
        if min(cx, cy, w, h) < -1e-6 or cx > 1 + 1e-6 or cy > 1 + 1e-6:
            findings.append(
                Finding("error", where, f"line {lineno}: center outside [0,1] ({cx:g},{cy:g})")
            )
            continue
        if w <= 0 or h <= 0:
            findings.append(Finding("error", where, f"line {lineno}: non-positive size {w:g}×{h:g}"))
            continue
        if w > 1 + 1e-6 or h > 1 + 1e-6:
            findings.append(
                Finding(
                    "warn",
                    where,
                    f"line {lineno}: w/h > 1 ({w:g}×{h:g}); box will be clipped to the image",
                )
            )
            w, h = min(w, 1.0), min(h, 1.0)
        if abs(w - 1.0) < 1e-6 and abs(h - 1.0) < 1e-6:
            findings.append(Finding("warn", where, f"line {lineno}: full-image box (likely a placeholder)"))
        boxes.append(Box(cls, min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), w, h))
    return boxes, findings


def read_yolo_file(path: Path) -> tuple[list[Box], list[Finding]]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return [], [Finding("error", str(path), f"unreadable: {exc}")]
    return parse_yolo_text(text, str(path))


# ---------------------------------------------------------------- COCO JSON

def load_coco(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        return {"__error__": str(exc)}
    return data if isinstance(data, dict) else {"__error__": f"top-level {type(data).__name__}"}


def coco_category_map(coco: dict) -> dict[int, int]:
    """COCO category id -> our class id. Unknown names map to None-ish (-1)."""
    mapping: dict[int, int] = {}
    for cat in coco.get("categories", []):
        cid = cat.get("id")
        name = str(cat.get("name", ""))
        sup = str(cat.get("supercategory", ""))
        key = name.strip().lower()
        ours = CLASS_TO_ID.get(key)
        if ours is None:
            for our_name, our_id in CLASS_TO_ID.items():
                if our_name in key or our_name in sup.lower():
                    ours = our_id
                    break
        if cid is not None:
            mapping[cid] = ours if ours is not None else -1
    return mapping


def index_coco(coco: dict, findings: list[Finding]) -> dict[str, list[dict]]:
    """Group annotation entries by image file *name* (basename), which is the only
    key we can rely on when the dataset folder was flattened."""
    images = {im["id"]: im for im in coco.get("images", [])}
    by_name: dict[str, list[dict]] = {}
    for ann in coco.get("annotations", []):
        im = images.get(ann.get("image_id"))
        if im is None:
            findings.append(
                Finding("warn", "coco", f"annotation {ann.get('id')} has unknown image_id {ann.get('image_id')}")
            )
            continue
        file_name = str(im.get("file_name", ""))
        by_name.setdefault(Path(file_name).name, []).append(ann)
        by_name.setdefault(Path(file_name).stem, []).append(ann)
    return by_name


def coco_ann_to_box(ann: dict, img_w: int, img_h: int, cat_map: dict[int, int], where: str) -> tuple[Box | None, list[Finding]]:
    out: list[Finding] = []
    bbox = ann.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, [Finding("error", where, f"bad bbox {bbox!r}")]
    x, y, w, h = (float(v) for v in bbox[:4])
    if ann.get("iscrowd"):
        return None, [Finding("info", where, "iscrowd annotation skipped")]
    cid = cat_map.get(ann.get("category_id"))
    if cid is None:
        out.append(Finding("warn", where, f"category_id {ann.get('category_id')} not in categories[]"))
        return None, out
    if cid == -1:
        out.append(
            Finding(
                "warn",
                where,
                f"category_id {ann.get('category_id')} is not Spot/Hole — will be dropped; "
                f"confirm the class mapping before training",
            )
        )
        return None, out
    if w <= 0 or h <= 0:
        out.append(Finding("error", where, f"non-positive box {bbox}"))
        return None, out
    cx, cy = x + w / 2.0, y + h / 2.0
    return Box(cid, cx / img_w, cy / img_h, w / img_w, h / img_h), out


# ---------------------------------------------------------------- Pascal VOC XML

def _find_text(node: ET.Element, tag: str) -> str | None:
    child = node.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_voc_xml(path: Path) -> dict | None:
    """Return {filename, width, height, boxes:[(cls,x1,y1,x2,y2)], truncated, diff}."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {"__error__": f"XML parse error: {exc}"}
    size = root.find("size")
    out = {
        "filename": _find_text(root, "filename"),
        "path": _find_text(root, "path"),
        "width": int(float(_find_text(size, "width") or 0)) if size is not None else 0,
        "height": int(float(_find_text(size, "height") or 0)) if size is not None else 0,
        "boxes": [],
        "findings": [],
    }
    for obj in root.findall("object"):
        name = (_find_text(obj, "name") or "").strip()
        cls = CLASS_TO_ID.get(name.lower())
        bnd = obj.find("bndbox")
        if bnd is None:
            out["findings"].append(Finding("error", str(path), f"object {name!r} has no <bndbox>"))
            continue
        try:
            x1 = float(_find_text(bnd, "xmin"))
            y1 = float(_find_text(bnd, "ymin"))
            x2 = float(_find_text(bnd, "xmax"))
            y2 = float(_find_text(bnd, "ymax"))
        except (TypeError, ValueError) as exc:
            out["findings"].append(Finding("error", str(path), f"object {name!r}: bad bndbox ({exc})"))
            continue
        if cls is None:
            out["findings"].append(
                Finding("warn", str(path), f"object name {name!r} is not Spot/Hole — skipped")
            )
            continue
        if x2 <= x1 or y2 <= y1:
            out["findings"].append(
                Finding("error", str(path), f"object {name!r}: inverted box {x1:g},{y1:g},{x2:g},{y2:g}")
            )
            continue
        out["boxes"].append((cls, x1, y1, x2, y2))
    return out


# ---------------------------------------------------------------- flat CSV

CSV_X_KEYS = ("xmin", "x_min", "left", "x1")
CSV_Y_KEYS = ("ymin", "y_min", "top", "y1")
CSV_W_KEYS = ("width", "w", "bbox_w", "bbox_width")
CSV_H_KEYS = ("height", "h", "bbox_h", "bbox_height")
CSV_X2_KEYS = ("xmax", "x_max", "right", "x2")
CSV_Y2_KEYS = ("ymax", "y_max", "bottom", "y2")
CSV_IMG_KEYS = ("image", "image_path", "filename", "file_name", "img_path", "source")
CSV_CLS_KEYS = ("label", "class", "class_name", "category", "category_name", "classname")


def _pick(fieldnames: list[str], options: tuple[str, ...]) -> str | None:
    lowered = {f.lower().strip(): f for f in fieldnames if f}
    for opt in options:
        if opt in lowered:
            return lowered[opt]
    return None


def index_csv(path: Path) -> tuple[dict[str, list[dict]], list[Finding], list[str]]:
    """Group CSV rows by image name. Returns (rows_by_image, findings, notes)."""
    findings: list[Finding] = []
    notes: list[str] = []
    by_image: dict[str, list[dict]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|") if sample else csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            fieldnames = [f for f in (reader.fieldnames or [])]
            img_col = _pick(fieldnames, CSV_IMG_KEYS)
            cls_col = _pick(fieldnames, CSV_CLS_KEYS)
            x_col = _pick(fieldnames, CSV_X_KEYS)
            y_col = _pick(fieldnames, CSV_Y_KEYS)
            w_col = _pick(fieldnames, CSV_W_KEYS)
            h_col = _pick(fieldnames, CSV_H_KEYS)
            x2_col = _pick(fieldnames, CSV_X2_KEYS)
            y2_col = _pick(fieldnames, CSV_Y2_KEYS)
            if not img_col or not cls_col or not x_col or not y_col:
                return {}, findings, [
                    f"{path.name}: not recognised as an object-detection CSV "
                    f"(columns={fieldnames[:12]}); inspect manually"
                ]
            notes.append(
                f"{path.name}: image={img_col} label={cls_col} box=({x_col},{y_col},"
                f"{w_col or x2_col},{h_col or y2_col})"
            )
            for rowno, row in enumerate(reader, start=2):
                name = (row.get(img_col) or "").strip()
                if not name:
                    continue
                cls = CLASS_TO_ID.get((row.get(cls_col) or "").strip().lower())
                if cls is None:
                    findings.append(
                        Finding("warn", f"{path.name}:{rowno}", f"class {row.get(cls_col)!r} unknown — skipped")
                    )
                    continue

                def num(col: str | None) -> float | None:
                    if not col or row.get(col) in (None, ""):
                        return None
                    try:
                        return float(row[col])
                    except ValueError:
                        return None

                x, y = num(x_col), num(y_col)
                w, h = num(w_col), num(h_col)
                x2, y2 = num(x2_col), num(y2_col)
                if x is None or y is None:
                    findings.append(Finding("error", f"{path.name}:{rowno}", "missing x/y"))
                    continue
                if w is not None and h is not None:
                    box = (cls, x, y, x + w, y + h)
                elif x2 is not None and y2 is not None:
                    box = (cls, x, y, x2, y2)
                else:
                    findings.append(Finding("error", f"{path.name}:{rowno}", "no width/height or xmax/ymax"))
                    continue
                by_image.setdefault(Path(name).name, []).append(box)
                by_image.setdefault(Path(name).stem, []).append(box)
    except Exception as exc:
        findings.append(Finding("error", str(path), f"could not read CSV: {exc}"))
    return by_image, findings, notes


# ---------------------------------------------------------------- aggregate stats

@dataclass
class DatasetStats:
    images: int = 0
    labeled_images: int = 0
    empty_label_images: int = 0
    boxes: int = 0
    per_class: Counter = field(default_factory=Counter)
    tiny: Counter = field(default_factory=Counter)
    areas_pct: list[float] = field(default_factory=list)
    heights_px: list[float] = field(default_factory=list)
    aspect: Counter = field(default_factory=Counter)
    findings: list[Finding] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    duplicate_names: list[str] = field(default_factory=list)
    resolutions: Counter = field(default_factory=Counter)
    modes: Counter = field(default_factory=Counter)

    def median(self, value: float, pool: list[float]) -> float:
        s = sorted(pool)
        return s[len(s) // 2] if s else value

    def mean_area_pct(self) -> float:
        return sum(self.areas_pct) / len(self.areas_pct) if self.areas_pct else 0.0

    def suggest_imgsz(self) -> tuple[int, str]:
        """Pick a defensible input resolution from the *small* end of the target
        distribution, not the median: mAP@50 is usually lost on the minority class
        (here: the tiny holes), and 640 is only right if it keeps those visible.
        """
        if not self.heights_px:
            return 640, "no valid boxes found — keep imgsz=640 until labels are verified"
        hs = sorted(self.heights_px)
        med_h = hs[len(hs) // 2]
        p25 = hs[int(len(hs) * 0.25)]
        max_side = max(self.max_side(), 1)
        FLOOR = 12  # px at the stride-32 head; below this a target is effectively invisible
        chosen, ok = 1280, False
        for size in (640, 768, 1024, 1280):
            if p25 * size / max_side >= FLOOR:
                chosen, ok = size, True
                break
        projected = p25 * chosen / max_side
        note = (
            f"median target {med_h:.0f}px, smallest quartile {p25:.0f}px on {max_side}px-native images; "
            f"imgsz {chosen} puts the small quartile at ~{projected:.0f}px "
            + ("(clears the stride-32 floor)" if ok else "(STILL below the floor)")
        )
        if chosen > 640 and max_side <= 640:
            note += (
                f" — but images are only {max_side}px natively, so imgsz>native upsamples and "
                "invents no detail: train at 640 and use crop/tiling augmentation instead"
            )
            chosen = 640
        elif not ok:
            note += (
                f" — even 1280 leaves {projected:.0f}px; slice the frames into tiles "
                "(predict.py --tile) rather than raising imgsz further"
            )
        return chosen, note

    def max_side(self) -> int:
        best = 0
        for (w, h) in self.resolutions:
            best = max(best, w, h)
        return best or 640

    def class_balance_note(self) -> str:
        counts = {name: self.per_class.get(idx, 0) for idx, name in enumerate(CLASS_NAMES)}
        vals = [v for v in counts.values() if v]
        if not vals:
            return "no boxes parsed"
        if max(vals) / max(min(vals), 1) >= 3:
            minority = min(counts, key=lambda k: counts[k])
            return (
                f"IMBALANCED {counts} — minority class '{minority}' will dominate the mAP "
                f"loss; expect it to gate your score"
            )
        return f"balanced-ish {counts}"


def summarize(
    records: list[tuple[Path, Path | None, int, int, list[Box]]],
    findings: list[Finding],
) -> DatasetStats:
    stats = DatasetStats(findings=list(findings))
    for path, _, w, h, boxes in records:
        stats.images += 1
        stats.resolutions[(w, h)] += 1
        if boxes:
            stats.labeled_images += 1
        elif path.suffix.lower() == ".txt" or boxes == []:
            stats.empty_label_images += 1
        for b in boxes:
            stats.boxes += 1
            stats.per_class[b.cls] += 1
            bw, bh = b.w * w, b.h * h
            stats.areas_pct.append(b.w * b.h * 100.0)
            stats.heights_px.append(bh)
            ratio = (max(bw, bh) / max(min(bw, bh), 1e-6)) if min(bw, bh) > 0 else 999
            stats.aspect["square" if ratio < 1.4 else ("wide" if ratio < 3 else "extreme")] += 1
            if bw < MIN_BOX_PIXELS or bh < MIN_BOX_PIXELS:
                stats.tiny[b.cls] += 1
                stats.findings.append(
                    Finding("warn", str(path), f"degenerate box {bw:.1f}×{bh:.1f}px (class {b.cls})")
                )
    return stats
