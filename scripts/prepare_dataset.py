#!/usr/bin/env python3
"""Phase 2 — normalize annotations into a canonical YOLO dataset, leakage-safe.

Read-only on dataset/raw. Writes dataset/prepared/{images,labels}/{train,val} plus
configs/data.prepared.yaml with absolute paths. Refuses to write if any error-level
annotation problem exists (pixel coords, unknown class ids, half-covered images,
inverted boxes) — a converter that "fixes" a wrong format silently is how you end up
training on garbage with a confident-looking loss curve.

    python scripts/prepare_dataset.py --dry-run         # see what it would do
    python scripts/prepare_dataset.py                   # build it
    python scripts/prepare_dataset.py --group-regex '(part\\d+)'   # your own part key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ped.discovery import build_layout  # noqa: E402
from _ped.split import build_prepared, write_data_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert + split annotations into YOLO layout.")
    ap.add_argument("--raw", default=str(ROOT / "dataset" / "raw"), help="original dataset folder (never modified)")
    ap.add_argument("--out", default=str(ROOT / "dataset" / "prepared"), help="working dataset to create")
    ap.add_argument("--data-yaml", default=str(ROOT / "configs" / "data.prepared.yaml"))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-count", type=int, default=None, help="exact number of validation COMPONENTS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--group-regex",
        default=None,
        help="regex with one capture group identifying the physical part, e.g. '^(.*)_[0-9]+$'. "
        "All matches of the same key stay in the same split.",
    )
    ap.add_argument("--no-grouping", action="store_true", help="split per image (only if parts are truly independent)")
    ap.add_argument("--overwrite", action="store_true", help="wipe an existing --out first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="permit images without any label artifact to become empty (negative) labels",
    )
    args = ap.parse_args()

    raw, out = Path(args.raw).expanduser().resolve(), Path(args.out).expanduser().resolve()
    if not raw.exists():
        print(f"[ERROR] raw dataset not found: {raw}", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()):
        if not args.overwrite and not args.dry_run:
            print(
                f"[ERROR] refusing to write into a non-empty directory: {out}\n"
                f"        re-run with --overwrite to rebuild it, or point --out elsewhere.",
                file=sys.stderr,
            )
            return 2
        if args.overwrite and not args.dry_run:
            import shutil

            shutil.rmtree(out)
            print(f"  removed previous working dataset: {out}")

    print(f"reading  : {raw}")
    print(f"writing  : {out}")
    layout = build_layout(raw, group_regex=args.group_regex)
    print(f"images   : {len(layout.images)}")

    report = build_prepared(
        layout,
        out,
        val_fraction=args.val_fraction,
        val_count=args.val_count,
        seed=args.seed,
        group_regex=args.group_regex,
        no_grouping=args.no_grouping,
        dry_run=args.dry_run,
        require_labels=not args.allow_unlabeled,
    )

    if "manifest" not in report:
        print("\nABORTED — nothing was written.")
        for e in report.get("errors", []):
            print(f"  ERROR: {e}")
        for w in report.get("warnings", []):
            print(f"  WARN:  {w}")
        for n in report.get("notes", []):
            print(f"  note:  {n}")
        print(
            "\nThis is the intended behaviour: the plan says stop and fix labels rather than\n"
            "train on guessed annotations. If you believe the check is wrong, run\n"
            "  python scripts/audit_dataset.py\n"
            "and read the format peek before overriding."
        )
        return 3

    man = report["manifest"]
    print(f"\nlabel source : {man['label_source']} ({man['kind']})")
    print(f"images       : train {man['images']['train']} / val {man['images']['val']}")
    print(f"components   : train {man['groups']['train']} / val {man['groups']['val']}")
    print(f"boxes per class: {man['boxes_per_class']}")
    print(f"empty labels   : {man['empty_label_images']} (intentional negatives)")
    if man["warnings"]:
        print(f"warnings kept: {len(man['warnings'])} (see manifest.json)")
        for w in man["warnings"][:5]:
            print(f"  [WARN] {w}")
    for n in man["notes"]:
        print(f"  · {n}")

    if args.dry_run:
        print(f"\ndry run — would write: {report.get('would_write')}")
        return 0

    yaml_path = write_data_yaml(out, Path(args.data_yaml))
    print(f"\nwrote {yaml_path}")
    print(yaml_path.read_text())
    print("Next:")
    print("  python scripts/validate_labels.py     # visual + numeric label check")
    print("  python scripts/train.py --dry-run     # chosen config for this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
