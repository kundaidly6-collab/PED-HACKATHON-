#!/usr/bin/env python3
"""Phase 1 — inspect the dataset and the hardware before committing to anything.

Answers, in order of what blocks you:
  1. is there data at all, and where?
  2. what label format actually exists, and does it pair to the images?
  3. are the labels trustworthy (class ids, normalized coords, coverage, duplicates)?
  4. how big are the targets -> what imgsz/model is defensible on THIS machine?
  5. what is the single next command to run?

Nothing here writes or converts anything. Run from anywhere:
    python scripts/audit_dataset.py [dataset/raw]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ped.discovery import build_layout, format_counts, iter_images  # noqa: E402
from _ped.hardware import detect_hardware  # noqa: E402
from _ped.labels import CLASS_NAMES, read_yolo_file, summarize  # noqa: E402
from _ped.split import choose_source, image_size  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rule(char: str = "=", n: int = 74) -> None:
    print(char * n)


def section(title: str) -> None:
    print()
    rule("=")
    print(title)
    rule("=")


def audit_hardware() -> None:
    section("1. SYSTEM / HARDWARE")
    prof = detect_hardware()
    for line in prof.summary_lines():
        print(f"  {line}")
    if not prof.on_gpu:
        print(
            "\n  -> No usable CUDA device. A CPU run is only worth it to prove the pipeline.\n"
            "     For real training: Colab/Kaggle free GPU, or a cloud instance with a CUDA\n"
            "     torch build matched to the driver. Do not assume `pip install ultralytics`\n"
            "     gives you a working GPU."
        )
    return None


def audit_discovery(root: Path) -> int:
    section("2. DATASET DISCOVERY")
    print(f"  scanning: {root}")
    if not root.exists():
        print(f"\n  [ERROR] path does not exist.")
        print(f"          create it and drop the original images in, e.g.:")
        print(f"            {root / 'images'}   (any layout: flat, per-split, per-part)")
        print(f"            {root / 'annotations'}  (coco.json / *.xml / *.txt / labels.csv)")
        print("\n  NEXT: put the organizers' files there, then re-run this script.")
        return 1
    layout = build_layout(root)
    print(f"  images found: {len(layout.images)}")
    if layout.image_root:
        print(f"  image root:   {layout.image_root.relative_to(root)}")
    for note in layout.notes:
        print(f"  · {note}")
    if not layout.images:
        print("\n  [ERROR] the folder exists but contains no readable image files.")
        print("          supported:", ", ".join(sorted({'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'})))
        print("          If the dataset is a zip/tar, unpack it first — scanners do not read archives.")
        return 1

    print("\n  directory sample:")
    for rec in layout.images[:5]:
        lbl = " + " + rec.yolo_label.name if rec.yolo_label else ""
        print(f"    {rec.path.relative_to(root)}{lbl}")
    dups = [r for r in layout.images if r.duplicate_of]
    if dups:
        print(f"\n  [WARN] {len(dups)} images share a filename+size with another (possible duplicates):")
        for r in dups[:5]:
            print(f"    {r.path.name}  ~  {Path(r.duplicate_of).name}")
    return 0


def audit_images(root: Path) -> Counter:
    section("3. IMAGE PROPERTIES")
    layout = build_layout(root)
    dims, modes, unreadable, sizes = Counter(), Counter(), [], []
    for i, rec in enumerate(layout.images):
        size = image_size(rec.path)
        if size is None:
            unreadable.append(rec.path)
            continue
        dims[size] += 1
        try:
            from PIL import Image

            with Image.open(rec.path) as im:
                modes[im.mode] += 1
        except Exception:
            pass
        sizes.append(rec.path.stat().st_size)
    print(f"  readable images: {len(layout.images) - len(unreadable)}/{len(layout.images)}")
    print("  most common dimensions (W,H):")
    for size, count in dims.most_common(10):
        print(f"    {size[0]}×{size[1]}  →  {count} images")
    print(f"  modes: {format_counts(modes)}")
    if sizes:
        total_gib = sum(sizes) / 1024**3
        print(
            f"  file size: median {Counter(sizes).most_common(1)[0][0] // 1024} KB, "
            f"total {total_gib:.2f} GiB"
        )
        if total_gib > 4:
            print("  [WARN] >4 GiB of images: keep cache='disk', not 'ram', and consider tiling.")
    if len(dims) > 1:
        print(
            f"  [WARN] {len(dims)} distinct resolutions. Ultralytics letterboxes every image to "
            f"imgsz, so a mix of sizes is fine for training but your submission coordinates must\n"
            f"         be scaled back to each ORIGINAL image size — check predict.py handles this."
        )
    for p in unreadable[:5]:
        print(f"  [ERROR] unreadable: {p.relative_to(root)}")
    return dims


def audit_labels(root: Path) -> dict:
    section("4. ANNOTATIONS")
    layout = build_layout(root)
    source, why = choose_source(layout)
    print(f"  candidate label source: {why}")
    if source is None:
        print("\n  [BLOCKER] no usable labels. Supervised training cannot start.")
        print("            Do NOT 'auto-label' every dark circle as a Hole; a detector trained on")
        print("            invented boxes scores worse than one trained on fewer true boxes.")
        print("\n  NEXT: annotate Spot + Hole boxes (CVAT / Label Studio / Roboflow), or ask the")
        print("        organizers for the label file. Then re-run this audit.")
        return {"blocked": True, "stats": None, "source": None}

    labels = source.load()
    findings = list(source.findings)
    records = [(l.path, None, l.width, l.height, l.boxes) for l in labels.values()]
    stats = summarize(records, findings)
    print(f"  parsed format: {source.kind}")
    print(f"  images with a parsed label artifact: {len(labels)}/{len(layout.images)}")
    print(f"  total boxes: {stats.boxes}")
    print(f"  per class:   " + ", ".join(f"{CLASS_NAMES[c]}={stats.per_class.get(c, 0)}" for c in range(len(CLASS_NAMES))))
    print(f"  {stats.class_balance_note()}")
    if stats.boxes:
        med_area = sorted(stats.areas_pct)[len(stats.areas_pct) // 2]
        print(f"  box area:    median {med_area:.2f}% of image area, mean {stats.mean_area_pct():.2f}%")
        print(f"  box height:  median {sorted(stats.heights_px)[len(stats.heights_px)//2]:.1f}px "
              f"on images up to {stats.max_side()}px")
        print(f"  shapes:      {format_counts(stats.aspect)}")
    empty = sum(1 for l in labels.values() if not l.boxes)
    if empty:
        print(f"  [INFO] {empty} images parsed with ZERO boxes. Confirm these are intentional")
        print(f"         negatives (part has no spot/hole) and not an image-name mismatch:")
        zero = [l.path.name for l in labels.values() if not l.boxes][:5]
        print(f"         e.g. {zero}")
    if stats.tiny:
        print(f"  [WARN] {dict(stats.tiny)} box(es) smaller than {3}px — likely mislabels; "
              "a 2px 'box' teaches the model to hallucinate")

    errors = [f for f in stats.findings if f.severity == "error"]
    warns = [f for f in stats.findings if f.severity == "warn"]
    if errors:
        print(f"\n  ERRORS ({len(errors)}) — these MUST be fixed before training:")
        for f in errors[:15]:
            print(f"    {f.format()}")
        if len(errors) > 15:
            print(f"    ... {len(errors) - 15} more")
    if warns:
        print(f"\n  WARNINGS ({len(warns)}):")
        for f in warns[:10]:
            print(f"    {f.format()}")
        if len(warns) > 10:
            print(f"    ... {len(warns) - 10} more")
    if not errors and not warns and stats.boxes:
        print("\n  no structural errors found in the parsed sample")
    return {"blocked": False, "stats": stats, "source": source.kind, "errors": len(errors)}


def sample_raw_label(root: Path) -> None:
    """Show literal file contents — the fastest way to catch a wrong format guess."""
    section("5. RAW FILE PEEK — verify the format by eye before trusting any converter")
    layout = build_layout(root)
    txt = [r.yolo_label for r in layout.images if r.yolo_label][:1]
    if not (layout.coco_json or layout.voc_xml or layout.csv_files or txt):
        print("  (nothing to peek — no annotation files were found)")
        return
    for jp in layout.coco_json[:1]:
        print(f"\n  --- peek {jp.relative_to(root)} ---")
        head = jp.read_text(encoding="utf-8", errors="replace")[:400].replace("\n", " ")
        print(f"  {head}...")
        try:
            import json

            data = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
            print(f"  keys: {list(data)[:12]}")
            cats = data.get("categories", [])
            if cats:
                print(f"  categories: {[(c.get('id'), c.get('name')) for c in cats[:8]]}")
                print(f"  >>> class ids in the file are {sorted({c.get('id') for c in cats})}, "
                      "which are NOT necessarily your YOLO class ids 0/1 — mapping is explicit in prepare_dataset.py")
        except Exception as exc:
            print(f"  (could not parse for summary: {exc})")
    if txt and txt[0]:
        p = txt[0]
        print(f"\n  --- peek {p.relative_to(root)} ---")
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:6]:
            print(f"  {line}")
        print("  ^ expect: <class_id> <cx> <cy> <w> <h>, all of the last four in [0,1]")
    if layout.voc_xml:
        p = layout.voc_xml[0]
        print(f"\n  --- peek {p.relative_to(root)} ---")
        print("  " + " ".join(p.read_text(encoding="utf-8", errors="replace").split())[:300] + "...")
    if layout.csv_files:
        p = layout.csv_files[0]
        print(f"\n  --- peek {p.relative_to(root)} ---")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[:4]
        for line in lines:
            print(f"  {line[:160]}")


def audit_plan(root: Path, stats) -> None:
    section("6. HARDWARE-ADAPTIVE PLAN (this is the point of the audit)")
    prof = detect_hardware()
    if stats is None or stats.boxes == 0:
        print("  cannot propose a training config: no verified boxes yet. Phase 2 first.")
        return
    from _ped.hardware import plan_training

    plan = plan_training(prof, stats)
    print(f"  median target height drives imgsz -> {plan.image_size}")
    for line in plan.lines()[1:]:
        print(f" {line}")
    n_img = stats.images
    if n_img < 40:
        print(f"\n  [RISK] {n_img} labeled images is very few. mAP@45 will be dominated by annotation\n"
              f"         quality, not architecture. Priorities: more boxes, correct class mapping,\n"
              f"         no part leaking between splits. Use pretrained weights and light augmentation.")
    if stats.max_side() >= 1600 and plan.image_size < stats.max_side():
        print(f"\n  [NOTE] images are up to {stats.max_side()}px but training at {plan.image_size}px downsamples\n"
              f"         ~{stats.max_side() // plan.image_size}×. Tiny welds can vanish. Benchmark\n"
              f"         --imgsz 1024/1280 and/or tiled inference before concluding the model is bad.")
    if "IMBALANCED" in stats.class_balance_note():
        print("\n  [NOTE] class imbalance detected above: oversample the minority class or add")
        print("         boxes for it before touching augmentation strength.")


def verdict(root: Path, blocked: bool, errors: int) -> None:
    section("NEXT STEP")
    if blocked:
        print("  python scripts/audit_dataset.py   # after you have added label files")
        print("  (training is not the next step — labeling or label access is)")
    elif errors:
        print("  fix the error-level findings above, then:")
        print("  python scripts/prepare_dataset.py --dry-run")
    else:
        print("  labels look structurally sound. Build the working dataset:")
        print(f"  python scripts/prepare_dataset.py --raw {root} --out {ROOT / 'dataset/prepared'}")
        print("  then, on the machine that will actually train:")
        print("  python scripts/train.py --dry-run      # show the chosen config")
        print("  python scripts/train.py                # baseline run")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the PED dataset and this machine.")
    ap.add_argument(
        "root",
        nargs="?",
        default=None,
        help="dataset folder to inspect (default: dataset/raw, then dataset/, then .)",
    )
    ap.add_argument("--no-images-detail", action="store_true", help="skip per-image size stats (faster on huge sets)")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = None
        primary = ROOT / "dataset" / "raw"
        if primary.exists() and any(iter_images(primary)):
            root = primary
        if root is None:
            # Do NOT fall back to scanning dataset/ wholesale: it can hold generated
            # copies (prepared/, demo sets) and silently double-count every image.
            candidates = sorted(
                d for d in (ROOT / "dataset").glob("*")
                if d.is_dir() and d.name != "raw" and any(iter_images(d))
            ) if (ROOT / "dataset").exists() else []
            if candidates:
                print("No images in dataset/raw. Other folders in dataset/ do contain images:")
                for c in candidates:
                    n = len(iter_images(c))
                    print(f"  {c.relative_to(ROOT)}  ({n} images)")
                print("\nPass one explicitly, e.g.  python scripts/audit_dataset.py dataset/_synthetic_demo/raw")
                print("Drop the organizers' data into dataset/raw/ to audit that instead.\n")
                root = candidates[0]
                print(f"(auditing {root.relative_to(ROOT)} — regenerate with --root to change)\n")
            else:
                root = primary

    rule("#")
    print("# PED HACKATHON · dataset & hardware audit")
    print(f"# repo: {ROOT}")
    rule("#")
    audit_hardware()
    code = audit_discovery(root)
    if code == 0:
        if not args.no_images_detail:
            audit_images(root)
        res = audit_labels(root)
        sample_raw_label(root)
        audit_plan(root, res.get("stats"))
        verdict(root, res.get("blocked", False), res.get("errors", 0))
        return 0 if not res.get("errors") else 0
    verdict(root, True, 0)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
