"""mAP@45 — the metric this competition is actually scored on.

Ultralytics reports mAP@50 and mAP@50-95. Neither is mAP@45, and the ``iou`` kwarg on
predict/NMS is *not* an evaluation threshold. So we score predictions ourselves using
COCO conventions: one-to-one greedy matching by descending confidence, 101-point
interpolated AP over a recall grid, averaged over the classes that have ground truth.

Confidence is swept as well, so the submission threshold is chosen from validation
data instead of guessed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RECALL_GRID = np.linspace(0.0, 1.0, 101)


@dataclass
class GTBox:
    cls: int
    xyxy: np.ndarray  # absolute pixels, (4,)


@dataclass
class PredBox:
    cls: int
    xyxy: np.ndarray
    conf: float


@dataclass
class Detection:
    """A prediction tagged with the image it came from, so matching stays per-image."""

    img: str
    box: PredBox


@dataclass
class ClassScore:
    cls: int
    name: str
    ap: float
    n_gt: int
    n_pred: int
    precision: float = 0.0
    recall: float = 0.0
    n_fp: int = 0
    n_fn: int = 0

@dataclass
class ScoreCard:
    iou_threshold: float
    map: float
    per_class: list[ClassScore] = field(default_factory=list)
    conf_curve: list[tuple[float, float]] = field(default_factory=list)
    n_images: int = 0
    n_gt: int = 0
    n_pred: int = 0
    notes: list[str] = field(default_factory=list)

    def table(self) -> str:
        head = f"{'class':<8}{f'AP@{self.iou_threshold:.2f}':>10}{'GT':>7}{'pred':>7}{'missed':>8}{'FP':>6}{'P':>7}{'R':>7}"
        lines = [head, "-" * len(head)]
        for sc in self.per_class:
            if sc.n_gt == 0:
                lines.append(
                    f"{sc.name:<8}{'n/a':>10}{sc.n_gt:>7}{sc.n_pred:>7}"
                    f"{sc.n_fn:>8}{sc.n_fp:>6}{'-':>7}{'-':>7}"
                )
            else:
                lines.append(
                    f"{sc.name:<8}{sc.ap:>10.4f}{sc.n_gt:>7}{sc.n_pred:>7}"
                    f"{sc.n_fn:>8}{sc.n_fp:>6}{sc.precision:>7.3f}{sc.recall:>7.3f}"
                )
        lines.append("-" * len(head))
        lines.append(f"{'mAP':<8}{self.map:>10.4f}{self.n_gt:>7}{self.n_pred:>7}")
        return "\n".join(lines)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) xyxy arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    inter_w = np.clip(np.minimum(a[:, 2:3], b[None, :, 2]) - np.maximum(a[:, 0:1], b[None, :, 0]), 0, None)
    inter_h = np.clip(np.minimum(a[:, 3:4], b[None, :, 3]) - np.maximum(a[:, 1:2], b[None, :, 1]), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def interpolate_ap(tp_flags: np.ndarray, n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """COCO 101-point interpolated AP.

    tp_flags must be sorted by descending confidence. Returns (AP, precision[], recall[]).
    """
    if tp_flags.size == 0 or n_gt == 0:
        return (0.0 if n_gt else -1.0), np.zeros(0), np.zeros(0)
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(~tp_flags)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    ap = 0.0
    for t in RECALL_GRID:
        mask = rec >= t
        ap += (prec[mask].max() if mask.any() else 0.0) / len(RECALL_GRID)
    return float(ap), prec, rec


def _match_one_class(
    dets: list[Detection],
    gt_by_img: dict[str, np.ndarray],
    iou_thr: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Greedy one-to-one matching. Returns (tp flags by conf order, missed-per-image)."""
    order = sorted(dets, key=lambda d: -d.box.conf)
    flags = np.zeros(len(order), dtype=bool)
    assigned: dict[str, set[int]] = defaultdict(set)
    for i, det in enumerate(order):
        gmat = gt_by_img.get(det.img)
        if gmat is None or gmat.size == 0:
            continue
        ious = iou_matrix(det.box.xyxy[None, :], gmat)[0]
        for gi in np.argsort(-ious):
            if ious[gi] < iou_thr:
                break
            if int(gi) not in assigned[det.img]:
                assigned[det.img].add(int(gi))
                flags[i] = True
                break
    missed = {
        img: int(max(0, (len(m) if m.size else 0) - len(assigned.get(img, ()))))
        for img, m in gt_by_img.items()
    }
    return flags, missed


def score(
    gt: dict[str, list[GTBox]],
    preds: dict[str, list[PredBox]],
    *,
    iou_threshold: float = 0.45,
    class_names: dict[int, str] | None = None,
    conf_steps: int = 40,
) -> ScoreCard:
    """Score *preds* against *gt*, both keyed by image id (file stem)."""
    names = class_names or {}
    gt_by_cls: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    gt_counts: dict[int, int] = defaultdict(int)
    for img, boxes in gt.items():
        by_cls: dict[int, list[GTBox]] = defaultdict(list)
        for b in boxes:
            by_cls[b.cls].append(b)
        for cls, lst in by_cls.items():
            gt_by_cls[cls][img] = np.stack([g.xyxy for g in lst]) if lst else np.zeros((0, 4))
            gt_counts[cls] += len(lst)
    for cls, per_img in list(gt_by_cls.items()):
        for img in gt:
            per_img.setdefault(img, np.zeros((0, 4)))

    dets_by_cls: dict[int, list[Detection]] = defaultdict(list)
    pred_counts: dict[int, int] = defaultdict(int)
    for img, boxes in preds.items():
        for b in boxes:
            dets_by_cls[b.cls].append(Detection(img=img, box=b))
            pred_counts[b.cls] += 1

    card = ScoreCard(
        iou_threshold=iou_threshold,
        map=0.0,
        n_images=len(set(gt) | set(preds)),
        n_gt=sum(len(v) for v in gt.values()),
        n_pred=sum(len(v) for v in preds.values()),
    )

    classes = sorted(set(gt_counts) | set(pred_counts))
    aps: list[float] = []
    for cls in classes:
        n_gt = gt_counts.get(cls, 0)
        dets = dets_by_cls.get(cls, [])
        gt_img = gt_by_cls.get(cls, {})
        if n_gt == 0:
            card.per_class.append(
                ClassScore(cls, names.get(cls, str(cls)), -1.0, 0, len(dets), n_fn=0, n_fp=len(dets))
            )
            if dets:
                card.notes.append(
                    f"class {names.get(cls, cls)}: {len(dets)} predictions, 0 ground-truth boxes. "
                    "Either the class was never annotated (fix the labels) or all of these are false positives."
                )
            continue

        flags, missed = _match_one_class(dets, gt_img, iou_threshold)
        order = sorted(dets, key=lambda d: -d.box.conf)
        confs = np.array([d.box.conf for d in order], dtype=float) if order else np.zeros(0)
        ap, prec, rec = interpolate_ap(flags, n_gt)
        aps.append(ap)
        n_fp = int((~flags).sum())
        n_tp = int(flags.sum())
        best_i = int(np.argmax(rec)) if rec.size else 0
        card.per_class.append(
            ClassScore(
                cls=cls,
                name=names.get(cls, str(cls)),
                ap=ap,
                n_gt=n_gt,
                n_pred=len(dets),
                precision=float(prec[best_i]) if prec.size else 0.0,
                recall=float(rec[best_i]) if rec.size else 0.0,
                n_fp=n_fp,
                n_fn=n_gt - n_tp,
            )
        )

        if conf_steps and order:
            curve = []
            for thr in np.round(np.linspace(0.02, 0.95, conf_steps), 3):
                n = int((confs >= thr).sum())
                if n == 0:
                    curve.append((float(thr), 0.0))
                    continue
                a, _, _ = interpolate_ap(flags[:n], n_gt)
                curve.append((float(thr), a))
            if not card.conf_curve:
                card.conf_curve = curve
            else:
                card.conf_curve = [(t, (a + b) / 2) for (t, a), (_, b) in zip(card.conf_curve, curve)]

    card.map = float(np.mean(aps)) if aps else 0.0
    return card


def best_confidence(card: ScoreCard) -> tuple[float, float]:
    """(threshold, mAP) at the best swept confidence threshold."""
    if not card.conf_curve:
        return 0.001, card.map
    conf, ap = max(card.conf_curve, key=lambda t: t[1])
    return float(conf), float(ap)


def load_gt_from_yolo(label_root: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[GTBox]]:
    """Ground truth for a prepared YOLO split: {stem: [GTBox]} (empty list = negative image)."""
    gt: dict[str, list[GTBox]] = {}
    for stem, (w, h) in sizes.items():
        boxes: list[GTBox] = []
        lp = label_root / f"{stem}.txt"
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                boxes.append(
                    GTBox(
                        cls=cls,
                        xyxy=np.array(
                            [
                                max(0.0, min(x1, w)),
                                max(0.0, min(y1, h)),
                                max(0.0, min(x2, w)),
                                max(0.0, min(y2, h)),
                            ],
                            dtype=float,
                        ),
                    )
                )
        gt[stem] = boxes
    return gt
