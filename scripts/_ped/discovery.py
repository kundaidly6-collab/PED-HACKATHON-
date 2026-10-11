"""Dataset discovery: find images, find the label source that actually exists, and
report how images are paired to labels.

This module deliberately does *not* assume a convention. It reports which candidate
layouts matched and with how many files, so a human can confirm the pairing before
any conversion happens. A ``labels/`` tree, a colocated ``.txt``, a COCO JSON and a
VOC XML directory are all plausible; guessing wrong silently corrupts a dataset.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
YOLO_EXTENSIONS = {".txt"}
COCO_EXTENSIONS = {".json"}
VOC_EXTENSIONS = {".xml"}
CSV_EXTENSIONS = {".csv"}

# Sub-directory names commonly used to hold images inside a dataset root.
IMAGE_DIR_CANDIDATES = (
    "images",
    "image",
    "imgs",
    "img",
    "photos",
    "frames",
    "data/images",
)

# Sub-directory names commonly used to hold YOLO-format text labels.
# 'annotations' is included because that is the name this project's agreed layout uses
# (dataset/raw/annotations/*.txt) — even though it can equally hold COCO/VOC files, which
# is why the pairing step only accepts *.txt from it.
LABEL_DIR_CANDIDATES = (
    "labels",
    "label",
    "lbl",
    "annotations",
    "annotation",
    "annotations/yolo",
    "labels_yolo",
    "yolo",
)

# Directories that never contain dataset images.
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "runs",
    "weights",
}


@dataclass
class ImageRecord:
    """One image on disk plus whatever label source we matched to it."""

    path: Path
    stem: str
    yolo_label: Path | None = None
    source: str | None = None  # 'coco' | 'voc' | 'csv' | 'yolo-file' | None
    source_ref: str | None = None  # json/xml path, or csv row id
    duplicate_of: str | None = None


@dataclass
class Layout:
    """The result of discovery: where the images are and which label sources exist."""

    root: Path
    image_root: Path | None = None
    label_root: Path | None = None
    images: list[ImageRecord] = field(default_factory=list)
    coco_json: list[Path] = field(default_factory=list)
    voc_xml: list[Path] = field(default_factory=list)
    csv_files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def image_stems(self) -> set[str]:
        return {rec.stem for rec in self.images}

    def has_any_labels(self) -> bool:
        return bool(
            self.label_root
            or self.coco_json
            or self.voc_xml
            or self.csv_files
            or any(rec.yolo_label for rec in self.images)
        )


def _is_hidden_tree(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def iter_images(root: Path) -> list[Path]:
    """All image files under *root*, sorted, excluding junk directories."""
    out = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith("._")
        and not _is_hidden_tree(p.relative_to(root))
    ]
    return out


def find_image_root(root: Path) -> tuple[Path | None, list[str]]:
    """Locate the directory that holds images.

    Prefers a conventional ``images/`` subtree when it exists and is non-empty;
    otherwise falls back to the *shallowest* directory containing images, which
    handles flat dumps (``dataset/raw/*.jpg``) without pulling in preview renders.
    """
    notes: list[str] = []
    for cand in IMAGE_DIR_CANDIDATES:
        sub = root / cand
        if sub.is_dir() and iter_images(sub):
            return sub, [f"image root: {sub.relative_to(root)} (conventional name)"]

    by_depth: dict[int, list[Path]] = {}
    for p in iter_images(root):
        by_depth.setdefault(len(p.relative_to(root).parts), []).append(p)
    if by_depth:
        depth = min(by_depth)
        dirs = {str(p.parent.relative_to(root)) for p in by_depth[depth]}
        if len(dirs) == 1:
            chosen = root / dirs.pop()
            notes.append(
                f"no images/ directory found; using shallowest image dir "
                f"'{chosen.relative_to(root)}' ({len(by_depth[depth])} images)"
            )
            return chosen, notes
        notes.append(
            f"images are scattered across {len(dirs)} directories at depth {depth}; "
            f"treating the whole root as the image tree"
        )
    return None, notes


def find_label_root(image_root: Path | None, root: Path) -> Path | None:
    """Find a sibling ``labels/`` tree for a given ``images/`` tree."""
    if image_root is not None:
        for sibling in (image_root.parent / "labels", image_root.parent / "label"):
            if sibling.is_dir():
                return sibling
        for cand in LABEL_DIR_CANDIDATES:
            sub = image_root.parent / cand
            if sub.is_dir():
                return sub
    for cand in LABEL_DIR_CANDIDATES:
        sub = root / cand
        if sub.is_dir():
            return sub
    return None


def find_annotation_files(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (coco_json, voc_xml, csv) candidate annotation files under *root*."""
    search_roots = [root / "annotations", root / "annotation", root]
    seen: dict[str, Path] = {}
    for base in search_roots:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or _is_hidden_tree(p.relative_to(root)):
                continue
            key = str(p)
            if key in seen:
                continue
            suffix = p.suffix.lower()
            if suffix in COCO_EXTENSIONS:
                seen[key] = p
            elif suffix in VOC_EXTENSIONS:
                seen[key] = p
            elif suffix in CSV_EXTENSIONS:
                seen[key] = p
    json_files = [p for k, p in seen.items() if p.suffix.lower() in COCO_EXTENSIONS]
    xml_files = [p for k, p in seen.items() if p.suffix.lower() in VOC_EXTENSIONS]
    csv_files = [p for k, p in seen.items() if p.suffix.lower() in CSV_EXTENSIONS]
    # A dataset.yaml / ultralytics cache file is not an annotation file.
    csv_files = [p for p in csv_files if p.name != "labels.csv.cache"]
    json_files = [
        p
        for p in json_files
        if p.name not in {"package.json", "package-lock.json"} and "_meta" not in p.parts
    ]
    return json_files, xml_files, csv_files


def looks_like_coco(path: Path) -> bool:
    """Peek at a JSON file: COCO requires images+annotations (+ ideally categories)."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    return isinstance(data, dict) and {"images", "annotations"} <= set(data.keys())


def strip_extension(name: str) -> str:
    """'component_001.jpg' -> 'component_001'; also handles '.tar.gz'-style stacks."""
    return re.sub(r"\.[A-Za-z0-9]{1,5}(\.[A-Za-z0-9]{1,5})?$", "", name)


def pair_yolo_files(layout: Layout) -> int:
    """Attach a colocated or labels/-tree ``.txt`` to each image. Returns matched count."""
    by_stem: dict[str, list[ImageRecord]] = {}
    for rec in layout.images:
        by_stem.setdefault(rec.stem, []).append(rec)

    matched = 0
    label_stems: set[str] = set()
    if layout.label_root is not None:
        for lp in sorted(layout.label_root.rglob("*")):
            if lp.is_file() and lp.suffix.lower() in YOLO_EXTENSIONS:
                label_stems.add(strip_extension(lp.name))

    for stem, recs in by_stem.items():
        candidates: list[Path] = []
        if layout.label_root is not None:
            # Mirror the image's sub-directory structure inside labels/ if present.
            for rec in recs:
                if layout.image_root is not None:
                    rel = rec.path.parent.relative_to(layout.image_root)
                    candidates.append(layout.label_root / rel / f"{stem}.txt")
                for split_dir in sorted(p for p in layout.label_root.iterdir() if p.is_dir()):
                    candidates.append(split_dir / f"{stem}.txt")
                candidates.append(layout.label_root / f"{stem}.txt")
        for rec in recs:
            candidates.append(rec.path.with_suffix(".txt"))
        found = next((c for c in candidates if c.is_file()), None)
        if found is not None:
            for rec in recs:
                rec.yolo_label = found
                rec.source = "yolo-file"
            matched += 1

    if layout.label_root is not None:
        orphans = label_stems - set(by_stem)
        if orphans:
            layout.notes.append(
                f"{len(orphans)} label files have no matching image "
                f"(e.g. {sorted(orphans)[:3]}) — stem mismatch or extra labels"
            )
    return matched


def detect_component_group(stem: str, override: str | None) -> str:
    """Key used to keep related captures out of both splits.

    Default heuristic: everything before the last ``_``/``-`` separated token that
    looks like a frame index. ``part012_frame3`` -> ``part012``.
    """
    if override:
        m = re.search(override, stem)
        return m.group(1) if m and m.groups() else stem
    parts = re.split(r"[_\-.]+", stem)
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return stem


def build_layout(root: Path, group_regex: str | None = None) -> Layout:
    """Full discovery pass over *root*. Never raises on missing pieces."""
    root = Path(root).expanduser().resolve()
    layout = Layout(root=root)
    if not root.exists():
        layout.notes.append(f"dataset root does not exist: {root}")
        return layout

    layout.image_root, notes = find_image_root(root)
    layout.notes.extend(notes)
    if layout.image_root is not None:
        paths = iter_images(layout.image_root)
    else:
        paths = iter_images(root)

    seen_content: dict[str, str] = {}
    for p in paths:
        rec = ImageRecord(path=p, stem=strip_extension(p.name))
        layout.images.append(rec)
        digest = f"{p.name}:{p.stat().st_size}"
        if digest in seen_content:
            rec.duplicate_of = seen_content[digest]
        else:
            seen_content[digest] = str(p)

    layout.label_root = find_label_root(layout.image_root, root)
    matched = pair_yolo_files(layout)

    coco, xml, csv = find_annotation_files(root)
    layout.coco_json = [p for p in coco if looks_like_coco(p)]
    non_coco = [str(p.relative_to(root)) for p in coco if p not in layout.coco_json]
    if non_coco:
        layout.notes.append(
            f"{len(non_coco)} .json file(s) are not COCO (no images+annotations keys): "
            f"{non_coco[:4]}"
        )
    layout.voc_xml = xml
    layout.csv_files = csv

    layout.notes.insert(
        0,
        f"paired {matched}/{len(layout.images)} images to a YOLO .txt; "
        f"candidates -> yolo-dir:{'yes' if matched else 'no'} "
        f"coco:{len(layout.coco_json)} voc:{len(layout.voc_xml)} csv:{len(layout.csv_files)}",
    )

    for rec in layout.images:
        rec.source_ref = detect_component_group(rec.stem, group_regex)
    return layout


def format_counts(values: Counter, limit: int = 10) -> str:
    if not values:
        return "(none)"
    return ", ".join(f"{k}×{v}" for k, v in values.most_common(limit))
