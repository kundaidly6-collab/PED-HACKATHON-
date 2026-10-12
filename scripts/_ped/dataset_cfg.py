"""Resolve a dataset YAML into concrete (images, labels) directories.

One implementation, used by every script that needs to find the validation split:
train.py, evaluate.py, select_best.py, submit.py.

Why this exists: an earlier version of this project hardcoded `dataset/prepared/...`
in several places. That worked only while the data happened to live there, and produced
the worst kind of failure — "0 images found" — for any other layout. Now that the data
arrives from Kaggle with whatever folder names the organisers chose, reading the YAML
the trainer actually uses is the only safe approach.

A relative `path:` is ambiguous: Ultralytics resolves it against its global
`datasets_dir` setting, which this code cannot see. So the plausible readings are tried
and the one that worked is reported, rather than silently guessing.
"""

from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(data_yaml: Path) -> dict:
    import yaml

    try:
        return yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"could not parse {data_yaml}: {exc}") from exc


def resolve_split_dirs(data_yaml: Path, split: str = "val", root_hint: Path | None = None) -> tuple[Path | None, Path | None, list[str]]:
    """Return (images_dir, labels_dir, notes) for *split* in *data_yaml*.

    labels_dir is None when the standard sibling layout does not exist — callers must
    treat that as "no ground truth" rather than as an empty label set.
    """
    data_yaml = Path(data_yaml)
    notes: list[str] = []
    cfg = load_yaml(data_yaml)
    rel = cfg.get(split) or f"images/{split}"
    base = cfg.get("path")

    if base is None:
        img_dir = (data_yaml.parent / rel).resolve()
        notes.append(f"no 'path:' key; assuming {rel} next to the yaml")
    else:
        base_p = Path(str(base)).expanduser()
        if base_p.is_absolute():
            img_dir = (base_p / rel).resolve()
        else:
            tried = [data_yaml.parent / base_p]
            if root_hint is not None:
                tried.append(root_hint / base_p)
            tried.append(base_p)
            for cand in tried:
                if cand.exists():
                    img_dir = (cand / rel).resolve()
                    notes.append(f"relative 'path:' resolved to {img_dir}")
                    break
            else:
                img_dir = (data_yaml.parent / base_p / rel).resolve()
                notes.append(
                    f"neither {tried[0]} nor {tried[-1]} exists — if the trainer also fails "
                    "to find the data, rewrite 'path:' as an absolute path"
                )
    if not img_dir.is_dir():
        return None, None, notes + [f"image directory does not exist: {img_dir}"]

    split_name = img_dir.name
    candidates: list[Path] = []
    # <root>/images/<split>  ->  <root>/labels/<split>   (the ultralytics convention)
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            swapped = parts[:i] + ["labels"] + parts[i + 1:]
            candidates.append(Path(*swapped))
            break
    candidates += [
        img_dir.parent.parent / "labels" / split_name,  # dataset/images/train -> dataset/labels/train
        img_dir.parent / "labels" / split_name,         # dataset/train/images -> dataset/train/labels
        img_dir.parent / "labels",                      # labels holds the split already
    ]
    lbl_dir = next((c for c in candidates if c.is_dir()), None)
    if lbl_dir is None:
        notes.append(
            "no labels directory found (tried "
            + ", ".join(str(c) for c in candidates[:3])
            + ") — ground-truth-dependent steps will refuse to run"
        )
    return img_dir, lbl_dir, notes


def find_data_yaml(root: Path, explicit: str | None = None) -> Path | None:
    """The dataset config to use: explicit > configs/data.prepared.yaml > configs/data.yaml."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for cand in (root / "configs" / "data.prepared.yaml", root / "configs" / "data.yaml"):
        if cand.is_file():
            return cand
    return None


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
