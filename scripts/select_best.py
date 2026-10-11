#!/usr/bin/env python3
"""Choose the checkpoint that is actually best on the COMPETITION metric.

Why this exists: Ultralytics decides `best.pt` (and when early stopping fires) using
its own fitness, which for detection is

    stats["fitness"] = stats["metrics/mAP50-95(B)"]   # always use box mAP50-95

i.e. mAP@50:95 — not mAP@45, not even mAP@50. So the checkpoint the trainer calls
"best" is chosen by a metric you are not scored on, and on a small dataset the two can
disagree. (train.py saves an epoch checkpoint every `save_period`, so the candidates
already exist on disk; they just were not being compared on the right yardstick.)

This script scores every saved checkpoint on the validation split with the same mAP@45
implementation used everywhere else in this repo, prints the comparison, and copies the
winner next to the run's weights.

    python scripts/select_best.py --run ped_baseline
    python scripts/select_best.py --run ped_baseline --imgsz 1024 --copy-to runs/ped_baseline/weights/chosen.pt
    python scripts/select_best.py --weights runs/a/weights/best.pt runs/b/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]

# Two checkpoints within this much mAP@45 are indistinguishable given one validation
# split this small; reporting a winner from noise would be false precision.
NOISE = 0.005
# Below this, the run has not learned anything worth submitting.
MIN_MEANINGFUL = 0.02

from _ped.labels import CLASS_NAMES  # noqa: E402
from _ped.metrics import best_confidence, score  # noqa: E402


def discover(data_yaml: Path, split: str = "val") -> tuple[Path, Path]:
    """Find the (images, labels) directories for a split, via the dataset yaml."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = Path(str(cfg.get("path", "."))).expanduser()
    if not base.is_absolute():
        base = (data_yaml.parent / base)
    img_rel = cfg.get(split) or f"images/{split}"
    img_dir = (base / img_rel).resolve()
    lbl_dir = Path(str(img_dir).replace(f"{Path(img_rel).parts[0]}", "labels", 1))
    if not lbl_dir.is_dir():
        alt = img_dir.parent.parent / "labels" / img_dir.name
        lbl_dir = alt if alt.is_dir() else lbl_dir
    return img_dir, lbl_dir


def score_checkpoint(weights: Path, img_dir: Path, lbl_dir: Path, sizes, gt, imgsz: int,
                     conf: float, device: str | None) -> dict:
    import map45  # reuse the verified predictor + GT loader

    preds = map45.run_model(weights, img_dir, sizes, imgsz, conf, device)
    card = score(gt, preds, iou_threshold=0.45,
                 class_names={i: n for i, n in enumerate(CLASS_NAMES)}, conf_steps=0)
    return {
        "weights": str(weights),
        "name": weights.name,
        "map45": round(card.map, 4),
        "per_class": {s.name: (None if s.n_gt == 0 else round(s.ap, 4)) for s in card.per_class},
        "n_pred": card.n_pred,
        "n_fn": sum(s.n_fn for s in card.per_class),
        "n_fp": sum(s.n_fp for s in card.per_class),
        "size_mib": round(weights.stat().st_size / 1024 / 1024, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", default=None, help="run name under runs/ (scores all its checkpoints)")
    src.add_argument("--weights", nargs="+", default=None, help="explicit checkpoint paths to compare")
    ap.add_argument("--data", default=str(ROOT / "configs" / "data.prepared.yaml"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.01, help="inference floor for a fair comparison")
    ap.add_argument("--device", default=None)
    ap.add_argument("--copy-to", default=None, help="copy the winner here (e.g. .../weights/chosen.pt)")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    if args.weights:
        candidates = [Path(w) for w in args.weights]
    else:
        wdir = ROOT / "runs" / args.run / "weights"
        if not wdir.is_dir():
            print(f"[ERROR] no weights directory at {wdir}", file=sys.stderr)
            return 2
        # epochN.pt first so the table reads chronologically, best/last last
        candidates = sorted(wdir.glob("epoch*.pt")) + [p for p in (wdir / "best.pt", wdir / "last.pt") if p.is_file()]
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        print("[ERROR] no checkpoints found", file=sys.stderr)
        return 2

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        print(f"[ERROR] dataset yaml not found: {data_yaml}\n        run scripts/prepare_dataset.py first", file=sys.stderr)
        return 2
    img_dir, lbl_dir = discover(data_yaml, args.split)
    if not img_dir.is_dir():
        print(f"[ERROR] {args.split} images not found at {img_dir}", file=sys.stderr)
        return 2

    import map45

    sizes = map45.read_sizes(img_dir)
    gt = map45.gt_from_labels(lbl_dir, sizes)
    n_gt = sum(len(v) for v in gt.values())
    if n_gt == 0:
        print(f"[ERROR] no ground truth in {lbl_dir}", file=sys.stderr)
        return 2

    print(f"=== checkpoint selection on mAP@45 ===")
    print(f"  split     : {img_dir}  ({len(sizes)} images, {n_gt} GT boxes)")
    print(f"  imgsz     : {args.imgsz}   conf floor: {args.conf}")
    print(f"  candidates: {len(candidates)}\n")

    rows = []
    for w in candidates:
        try:
            rows.append(score_checkpoint(w, img_dir, lbl_dir, sizes, gt, args.imgsz, args.conf, args.device))
        except Exception as exc:
            print(f"  [skip] {w.name}: {type(exc).__name__}: {str(exc)[:110]}")

    if not rows:
        print("[ERROR] every checkpoint failed to score", file=sys.stderr)
        return 2

    head = f"{'checkpoint':<16}{'mAP@45':>9}" + "".join(f"{n:>9}" for n in CLASS_NAMES) + f"{'missed':>8}{'FP':>7}{'MiB':>7}"
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = "".join(f"{(r['per_class'].get(n) if r['per_class'].get(n) is not None else float('nan')):>9.4f}" for n in CLASS_NAMES)
        print(f"{r['name']:<16}{r['map45']:>9.4f}{cells}{r['n_fn']:>8}{r['n_fp']:>7}{r['size_mib']:>7.1f}")

    top = max(r["map45"] for r in rows)
    tied = [r for r in rows if top - r["map45"] <= NOISE]
    # Prefer the checkpoint the trainer maintained (best.pt) when there is no evidence
    # to override it: an epoch dump still carries optimizer state and is not the
    # intended product. Ties are reported, not silently broken.
    preferred = [r for r in tied if r["name"] in ("best.pt", "last.pt")] or tied
    winner = max(preferred, key=lambda r: r["map45"])

    # A run that scored nothing has not produced a winner, and saying otherwise is the
    # exact false confidence this repo tries to avoid.
    if top < MIN_MEANINGFUL:
        print(f"\nNO USABLE CHECKPOINT: the best score across {len(rows)} checkpoint(s) is "
              f"{top:.4f} (below {MIN_MEANINGFUL}).")
        print("  Every checkpoint missed or mislocated essentially all ground-truth boxes, so there")
        print("  is nothing here worth submitting — this is a training problem, not a selection one.")
        print("  Check, in this order: enough epochs (a few epochs from scratch learns nothing),")
        print("  imgsz vs target size, label correctness (validate_labels.py), and class balance.")
        print("  Not copying anything.")
        if args.report_json:
            Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report_json).write_text(
                json.dumps({"winner": None, "reason": f"all checkpoints <= {MIN_MEANINGFUL}",
                            "all": rows, "imgsz": args.imgsz, "split": args.split}, indent=2), encoding="utf-8")
            print(f"  report → {args.report_json}")
        return 1

    print(f"\nBEST mAP@45: {winner['name']}  ({winner['map45']:.4f})")
    if len(tied) > 1:
        names = ", ".join(f"{r['name']} {r['map45']:.4f}" for r in tied)
        print(f"  {len(tied)} checkpoints are within {NOISE} of each other ({names}).")
        print(f"  Reporting a tie rather than a winner; defaulting to {winner['name']}.")
        print("  With scores this close, re-run with more epochs — the ranking is not yet meaningful.")

    ultralytics_best = next((r for r in rows if r["name"] == "best.pt"), None)
    if ultralytics_best:
        delta = winner["map45"] - ultralytics_best["map45"]
        if winner["name"] == "best.pt":
            print("  Ultralytics' own choice agrees with mAP@45 here — use best.pt.")
        elif delta <= NOISE:
            print(f"  Ultralytics' best.pt ties on mAP@45 ({ultralytics_best['map45']:.4f}); keeping best.pt.")
        else:
            print(f"  NOTE: Ultralytics' best.pt scores {ultralytics_best['map45']:.4f} on mAP@45 "
                  f"({delta:+.4f} vs the winner).")
            print("        ultralytics picks best.pt by mAP@50:95, which is not the competition metric.")
            print(f"        Submit {winner['name']} unless you have a reason not to.")

    target = Path(args.copy_to) if args.copy_to else Path(winner["weights"]).parent / "best_map45.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["weights"], target)
    print(f"\n  copied → {target}")
    print(f"  use it:  python scripts/submit.py --weights {target} --source <test dir> --format <fmt>")

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(
            json.dumps({"winner": winner, "all": rows, "imgsz": args.imgsz, "split": args.split}, indent=2),
            encoding="utf-8")
        print(f"  report → {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
