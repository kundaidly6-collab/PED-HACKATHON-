"""Annotation normalization + leakage-safe splitting.

Takes whatever label source discovery found (YOLO .txt, COCO JSON, VOC XML, CSV),
writes a canonical YOLO dataset under ``dataset/prepared/`` and a ``data.prepared.yaml``
with **absolute** paths.

Why absolute: Ultralytics resolves a relative ``path:`` in a dataset YAML against the
global ``datasets_dir`` setting, not against the YAML's own location. A relative path
that "looks right" in the file can silently point at the wrong folder on another
machine — a nasty failure mode in a hackathon where you switch between laptop and cloud.

Two hard rules enforced here:
  * originals in ``dataset/raw/`` are only ever read;
  * an error-level label problem aborts the build rather than training on bad boxes.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .discovery import Layout, strip_extension
from .labels import (
    CLASS_NAMES,
    DatasetStats,
    Finding,
    ImageLabels,
    coco_ann_to_box,
    coco_category_map,
    index_coco,
    index_csv,
    load_coco,
    parse_voc_xml,
    read_yolo_file,
    summarize,
)

Image.open  # keep PIL import explicit for readability


@dataclass
class SplitPlan:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


# --------------------------------------------------------------- source -> boxes

class LabelSource:
    """Normalises one concrete label layout into {stem: ImageLabels}."""

    kind: str = "unknown"

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.findings: list[Finding] = []

    def load(self) -> dict[str, ImageLabels]:
        raise NotImplementedError


class YoloFileSource(LabelSource):
    kind = "yolo-txt"

    def load(self) -> dict[str, ImageLabels]:
        out: dict[str, ImageLabels] = {}
        for rec in self.layout.images:
            if rec.yolo_label is None:
                continue
            size = image_size(rec.path)
            if size is None:
                self.findings.append(Finding("error", str(rec.path), "unreadable image, cannot validate boxes"))
                continue
            boxes, findings = read_yolo_file(rec.yolo_label)
            self.findings.extend(findings)
            out[rec.stem] = ImageLabels(rec.path, size[0], size[1], boxes, [])
        return out


class CocoSource(LabelSource):
    kind = "coco-json"

    def __init__(self, layout: Layout, json_path: Path) -> None:
        super().__init__(layout)
        self.json_path = json_path

    def load(self) -> dict[str, ImageLabels]:
        out: dict[str, ImageLabels] = {}
        coco = load_coco(self.json_path)
        if coco is None:
            self.findings.append(Finding("error", str(self.json_path), "not valid JSON"))
            return out
        if "__error__" in coco:
            self.findings.append(Finding("error", str(self.json_path), coco["__error__"]))
            return out
        cat_map = coco_category_map(coco)
        unknown = sorted(cid for cid, ours in cat_map.items() if ours == -1)
        if unknown:
            self.findings.append(
                Finding(
                    "error",
                    self.json_path.name,
                    f"categories {unknown} map to neither Spot nor Hole — set the mapping explicitly "
                    "before converting; do not guess",
                )
            )
        by_name = index_coco(coco, self.findings)
        coco_images = {im.get("file_name"): im for im in coco.get("images", [])}
        for rec in self.layout.images:
            name = rec.path.name
            entries = by_name.get(name) or by_name.get(rec.stem) or []
            meta = coco_images.get(name) or coco_images.get(rec.path.name)
            w, h = (meta.get("width"), meta.get("height")) if meta else (None, None)
            real = image_size(rec.path)
            if real and (w, h) != real:
                if w and h:
                    self.findings.append(
                        Finding(
                            "warn",
                            name,
                            f"COCO says {w}×{h} but the file is {real[0]}×{real[1]} — using the real size",
                        )
                    )
                w, h = real if real else (w, h)
            if not real:
                self.findings.append(Finding("error", name, "image unreadable"))
                continue
            boxes: list = []
            for ann in entries:
                box, fnd = coco_ann_to_box(ann, w, h, cat_map, f"{name}#{ann.get('id')}")
                self.findings.extend(fnd)
                if box is not None:
                    boxes.append(box)
            out[rec.stem] = ImageLabels(rec.path, w, h, boxes, [])
        return out


class VocSource(LabelSource):
    kind = "pascal-voc"

    def __init__(self, layout: Layout, xml_paths: list[Path]) -> None:
        super().__init__(layout)
        self.xml_paths = xml_paths

    def load(self) -> dict[str, ImageLabels]:
        out: dict[str, ImageLabels] = {}
        by_stem: dict[str, Path] = {}
        for xp in self.xml_paths:
            by_stem.setdefault(strip_extension(xp.name), xp)
        for rec in self.layout.images:
            xp = by_stem.get(rec.stem)
            if xp is None:
                continue
            parsed = parse_voc_xml(xp)
            if parsed is None or "__error__" in (parsed or {}):
                self.findings.append(Finding("error", str(xp), (parsed or {}).get("__error__", "unparseable")))
                continue
            self.findings.extend(parsed["findings"])
            real = image_size(rec.path)
            if not real:
                self.findings.append(Finding("error", str(rec.path), "unreadable image"))
                continue
            w, h = real
            if parsed["width"] and (parsed["width"], parsed["height"]) != (w, h):
                self.findings.append(
                    Finding(
                        "warn",
                        xp.name,
                        f"XML size {parsed['width']}×{parsed['height']} != image {w}×{h}; "
                        "coordinates rescaled against the real image size",
                    )
                )
            boxes = []
            if w > 0 and h > 0:
                for cls, x1, y1, x2, y2 in parsed["boxes"]:
                    boxes.append(
                        _norm(
                            cls,
                            (x1 + x2) / 2 / w,
                            (y1 + y2) / 2 / h,
                            (x2 - x1) / w,
                            (y2 - y1) / h,
                        )
                    )
            out[rec.stem] = ImageLabels(rec.path, w, h, boxes, [])
        return out


def _norm(cls: int, cx: float, cy: float, w: float, h: float):
    from .labels import Box

    return Box(cls, min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), max(w, 1e-6), max(h, 1e-6))


class CsvSource(LabelSource):
    kind = "flat-csv"

    def __init__(self, layout: Layout, csv_paths: list[Path], img_w: int, img_h: int) -> None:
        super().__init__(layout)
        self.csv_paths = csv_paths
        self.default_w = img_w
        self.default_h = img_h

    def load(self) -> dict[str, ImageLabels]:
        merged: dict[str, list] = defaultdict(list)
        for cp in self.csv_paths:
            rows, findings, notes = index_csv(cp)
            self.findings.extend(findings)
            for n in notes:
                self.findings.append(Finding("info", cp.name, n))
            for name, boxes in rows.items():
                merged[strip_extension(name)].extend(boxes)
        out: dict[str, ImageLabels] = {}
        for rec in self.layout.images:
            entries = merged.get(rec.stem, [])
            real = image_size(rec.path)
            if not real:
                continue
            w, h = real
            boxes = []
            for cls, x1, y1, x2, y2 in entries:
                if w > 1 and h > 1:
                    boxes.append(_norm(cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            out[rec.stem] = ImageLabels(rec.path, w, h, boxes, [])
        return out


def choose_source(layout: Layout) -> tuple[LabelSource | None, str]:
    """Rank candidate label sources by how many images they actually cover."""
    candidates: list[tuple[int, LabelSource, str]] = []
    if any(r.yolo_label for r in layout.images):
        src = YoloFileSource(layout)
        n = sum(1 for r in layout.images if r.yolo_label is not None)
        candidates.append((n, src, f"YOLO .txt files next to/in labels/ for {n} images"))
    for jp in layout.coco_json:
        coco = load_coco(jp) or {}
        n_img = len(coco.get("images", [])) if isinstance(coco, dict) else 0
        covered = min(n_img, len(layout.images)) if n_img else 0
        candidates.append((covered, CocoSource(layout, jp), f"COCO JSON {jp.name} ({n_img} image entries)"))
    if layout.voc_xml:
        stems = {strip_extension(p.name) for p in layout.voc_xml}
        covered = len(stems & layout.image_stems)
        candidates.append((covered, VocSource(layout, layout.voc_xml), f"Pascal VOC XML ({covered} matched)"))
    if layout.csv_files:
        candidates.append((len(layout.images), CsvSource(layout, layout.csv_files, 0, 0), "flat CSV"))
    if not candidates:
        return None, (
            "NO usable label source found. Do not fabricate labels and do not treat dark "
            "circular regions as holes — annotate the images first (CVAT/Label Studio, 2 classes)."
        )
    candidates.sort(key=lambda t: -t[0])
    n, src, why = candidates[0]
    if n == 0:
        return None, f"candidate sources matched 0 images ({'; '.join(c[2] for c in candidates)})"
    runner_up = candidates[1][2] if len(candidates) > 1 else None
    if runner_up:
        why += f" | also present: {runner_up} (ignored)"
    return src, why


# --------------------------------------------------------------- splitting

# Trailing tokens that mean "this is another capture of the same part".
FRAME_WORDS = {
    "frame", "img", "image", "shot", "view", "photo", "pic", "cam", "camera",
    "snap", "snapshot", "capture", "scan", "sample", "t", "f",
}
_NUM_SUFFIX = re.compile(r"^(?P<word>[A-Za-z]+)(?P<num>\d+)$")


def group_key(stem: str, override: str | None) -> str:
    """Key identifying the physical component, so all its captures stay in one split.

    'part012_frame3' and 'part012_4' both -> 'part012'.
    A stem with no recognizable capture suffix is its own group, which over-groups
    (conservative: it can waste data but cannot leak a part across the split).
    Pass --group-regex to override when the naming is not guessable.
    """
    if override:
        m = re.search(override, stem)
        return (m.group(1) if m and m.groups() else stem)
    tokens = [t for t in re.split(r"[_\-.\s]+", stem) if t]
    if len(tokens) < 2:
        return stem
    last = tokens[-1]
    if last.isdigit():                       # part012_3
        return "_".join(tokens[:-1])
    m = _NUM_SUFFIX.match(last)
    if m and m.group("word").lower() in FRAME_WORDS:   # part012_frame3
        return "_".join(tokens[:-1])
    return stem if m is None else "_".join(tokens[:-1] + [f"{m.group('word')}{m.group('num')}"])


def plan_split(
    stems: list[str],
    boxes_per_stem: dict[str, list],
    *,
    val_fraction: float = 0.2,
    val_count: int | None = None,
    seed: int = 42,
    group_override: str | None = None,
    no_grouping: bool = False,
    min_val_groups: int = 1,
) -> SplitPlan:
    """Group-aware, class-stratified split. Whole components move together."""
    plan = SplitPlan()
    groups: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        key = s if no_grouping else group_key(s, group_override)
        groups[key].append(s)
    plan.groups = {k: tuple(sorted(v)) for k, v in groups.items()}

    # A group is "Hole-bearing" etc.; stratify so both classes appear in val.
    def signature(members: list[str]) -> tuple[int, ...]:
        counts = [0] * len(CLASS_NAMES)
        for m in members:
            for b in boxes_per_stem.get(m, []):
                counts[b.cls] += 1
        return tuple(1 if c else 0 for c in counts)

    buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for key, members in groups.items():
        buckets[signature(members)].append(key)

    rng = random.Random(seed)
    all_keys = sorted(k for keys in buckets.values() for k in keys)
    n_groups = len(all_keys)
    if n_groups < 2:
        plan.notes.append(
            f"only {n_groups} component group detected — a group-aware split is impossible. "
            "Every image will go to train and validation will be untrustworthy; collect more parts "
            "or accept a random per-image split with --no-grouping (and say so in your writeup)."
        )
        plan.train, plan.val = all_keys, []
        return plan

    target = val_count if val_count else max(min_val_groups, round(n_groups * val_fraction))
    target = min(target, max(1, n_groups - 1))
    selected: list[str] = []
    for sig, keys in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        keys = sorted(keys)
        rng.shuffle(keys)
        share = max(1 if keys else 0, round(len(keys) * target / n_groups))
        selected.extend(keys[:share])
        if len(selected) >= target:
            break
    if len(selected) < target:  # top up from what is left
        rest = [k for k in all_keys if k not in selected]
        rng.shuffle(rest)
        selected.extend(rest[: target - len(selected)])
    if not selected:
        selected = [rng.choice(all_keys)]

    sel = set(selected)
    for key in all_keys:
        (plan.val if key in sel else plan.train).append(key)
    plan.notes.append(
        f"{n_groups} component groups -> {len(plan.train)} train / {len(plan.val)} val "
        f"(whole components kept together, stratified on class presence)"
    )
    if len(plan.val) == 0:
        plan.notes.append("validation set is empty — refusing to train blind, check --val-fraction")
    return plan


# --------------------------------------------------------------- build

def build_prepared(
    layout: Layout,
    out_root: Path,
    *,
    val_fraction: float = 0.2,
    val_count: int | None = None,
    seed: int = 42,
    group_regex: str | None = None,
    no_grouping: bool = False,
    dry_run: bool = False,
    require_labels: bool = True,
) -> dict:
    """Convert, validate, split, write. Returns a manifest dict."""
    out_root = Path(out_root)
    source, why = choose_source(layout)
    report: dict = {"label_source": why, "notes": list(layout.notes), "errors": [], "warnings": []}

    # A label file whose name matches no image is silent annotation LOSS: those boxes
    # never reach training and nothing else complains. Make it explicit.
    if layout.label_root is not None:
        stems = {rec.stem for rec in layout.images}
        orphans = sorted(
            p.name for p in layout.label_root.rglob("*.txt")
            if strip_extension(p.name) not in stems
        )
        if orphans:
            report["warnings"].append(
                f"{len(orphans)} label file(s) match no image and will be DROPPED "
                f"(check for typos in image/label names): {orphans[:5]}"
            )
            report["notes"].append(
                f"orphan labels: {len(orphans)} — if these are renames of real parts, fix the "
                "filenames; their annotations are not being used"
            )

    if source is None:
        report["errors"].append(why)
        return report
    labels = source.load()
    findings = list(source.findings)
    errors = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warn"]

    # Images with no label artifact at all are the dangerous case: missing vs. truly empty.
    unlabeled = [rec for rec in layout.images if rec.stem not in labels]
    if unlabeled:
        report["notes"].append(
            f"{len(unlabeled)}/{len(layout.images)} images have no matched label artifact. "
            "If the source is per-image files these are MISSING annotations, not negatives."
        )
        if require_labels and source.kind in {"yolo-txt", "pascal-voc", "flat-csv"} and len(unlabeled) > len(layout.images) * 0.02:
            errors.append(
                Finding(
                    "error",
                    "coverage",
                    f"{len(unlabeled)} unlabeled images — refusing to train on a half-annotated set",
                )
            )
            report["errors"].extend(f.format() for f in errors)
            return report

    # A YOLO dataset needs a .txt for every image (empty = intentional negative).
    all_stems = [rec.stem for rec in layout.images if rec.stem in labels or not unlabeled]
    boxes_map = {stem: labels[stem].boxes for stem in all_stems}
    plan = plan_split(
        all_stems,
        boxes_map,
        val_fraction=val_fraction,
        val_count=val_count,
        seed=seed,
        group_override=group_regex,
        no_grouping=no_grouping,
    )
    report["notes"].extend(plan.notes)

    path_map = {rec.stem: rec.path for rec in layout.images}
    if errors:
        report["errors"].extend(f.format() for f in errors)
        if not dry_run:
            report["notes"].append(
                f"{len(errors)} error-level annotation problems. Fix them, then re-run "
                "(or pass a narrower --require-labels exemption). Aborting before any write."
            )
            return report
        report["notes"].append("dry run: errors listed but no files written")

    if dry_run:
        report["notes"].append("dry run: nothing written")
        report["would_write"] = {
            "train": len(plan.train),
            "val": len(plan.val),
            "out_root": str(out_root),
        }
        return report

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    per_class = {i: 0 for i, _ in enumerate(CLASS_NAMES)}
    per_split_class: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    n_empty = 0
    for split, groups in (("train", plan.train), ("val", plan.val)):
        for key in groups:
            for stem in plan.groups[key]:
                rec_labels = labels.get(stem)
                if rec_labels is None:
                    continue
                dst_img = out_root / "images" / split / f"{stem}{path_map[stem].suffix.lower()}"
                dst_lbl = out_root / "labels" / split / f"{stem}.txt"
                if path_map[stem] != dst_img:
                    shutil.copy2(path_map[stem], dst_img)
                lines = []
                bucket = per_split_class[split]
                for b in rec_labels.boxes:
                    lines.append(f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}")
                    per_class[b.cls] += 1
                    name = CLASS_NAMES[b.cls]
                    bucket[name] = bucket.get(name, 0) + 1
                if not lines:
                    n_empty += 1
                dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    count_images = lambda groups: sum(len(plan.groups[g]) for g in groups)  # noqa: E731
    manifest = {
        "label_source": why,
        "kind": source.kind,
        "seed": seed,
        "images": {"train": count_images(plan.train), "val": count_images(plan.val)},
        "groups": {"train": len(plan.train), "val": len(plan.val)},
        "train_groups": list(plan.train),
        "val_groups": list(plan.val),
        "boxes_per_class": {CLASS_NAMES[i]: c for i, c in per_class.items()},
        "boxes_by_split": per_split_class,
        "empty_label_images": n_empty,
        "warnings": [f.format() for f in warns][:200],
        "notes": report["notes"],
        "out_root": str(out_root),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report["manifest"] = manifest
    return report


def write_data_yaml(out_root: Path, path: Path, names: list[str] = CLASS_NAMES) -> Path:
    """Absolute-path dataset config, ready for train/val/predict."""
    lines = [
        f"# generated by prepare_dataset.py — absolute paths so the config survives moving machines",
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:12]
