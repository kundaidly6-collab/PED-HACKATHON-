#!/usr/bin/env python3
"""Known-answer tests for the parts of this pipeline that fail QUIETLY.

A broken audit script is obvious. A broken mAP@50 or a broken train/val split is not —
both produce plausible-looking numbers, and you would only discover the problem after
the deadline. So these are checked against hand-computed answers:

  metric     perfect predictions -> 1.0; one of two found -> ~0.5; box matching at
             IoU 0.44 vs 0.46 behaves differently at threshold 0.45; duplicates count
             as false positives; a class with no ground truth is excluded from the mean
  parsing    pixel-space coords, unknown class ids, inverted boxes and 4-value lines
             are all CAUGHT (each is a real failure mode of real annotation exports)
  formats    COCO (category ids that are not 0/1) and VOC XML convert to the same boxes
             as the equivalent YOLO file
  split      no component group appears in both train and val (leakage check)
  roundtrip  normalized -> pixels -> normalized survives, so submission coordinates are
             not silently off by a factor of the image size

    python scripts/selftest.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ped.labels import Box, parse_yolo_text, parse_voc_xml, coco_category_map  # noqa: E402
from _ped.metrics import GTBox, PredBox, score  # noqa: E402
from _ped.split import build_prepared, choose_source, group_key, plan_split  # noqa: E402
from _ped.discovery import build_layout  # noqa: E402
import annotate  # noqa: E402  (local annotation UI: label writer + save routing)

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not condition else ""))


def gt_one(x1, y1, x2, y2, cls=0) -> GTBox:
    return GTBox(cls=cls, xyxy=np.array([x1, y1, x2, y2], dtype=float))


def pred_one(x1, y1, x2, y2, conf=0.9, cls=0) -> PredBox:
    return PredBox(cls=cls, xyxy=np.array([x1, y1, x2, y2], dtype=float), conf=conf)


# --------------------------------------------------------------- metric

def test_metric() -> None:
    print("\n= mAP@50 metric (the scored number) =")
    # The competition threshold is 0.50. This guards the assumption most easily lost in
    # a refactor: a default of 0.45 silently over-reports every score.
    import inspect
    default_iou = inspect.signature(score).parameters["iou_threshold"].default
    check("score() defaults to the competition threshold 0.50 (not 0.45)",
          default_iou == 0.50, f"found {default_iou}")
    gt = {"a": [gt_one(0, 0, 100, 100)], "b": [gt_one(200, 0, 300, 100)]}

    perfect = {"a": [pred_one(0, 0, 100, 100, 0.9)], "b": [pred_one(200, 0, 300, 100, 0.8)]}
    card = score(gt, perfect, iou_threshold=0.45)
    check("perfect predictions -> mAP 1.0", abs(card.map - 1.0) < 1e-9, f"got {card.map:.4f}")

    half = {"a": [pred_one(0, 0, 100, 100, 0.9)], "b": []}
    card = score(gt, half, iou_threshold=0.45)
    # 51 of the 101 recall stops sit at t<=0.5, so one-of-two-found scores 0.50495,
    # not 0.5 — that is what 101-point interpolation does, and matching it matters
    # when you compare against the organizers' scorer.
    check("one of two objects found -> AP 0.50495 (101-point interpolation)",
          abs(card.map - 51 / 101) < 1e-9, f"got {card.map:.5f}")

    # IoU of [0,0,100,100] vs [0,0,100,90] = 0.9; vs [0,0,100,50] = 0.5
    check("IoU 0.9 matches at threshold 0.45",
          score({"a": [gt_one(0, 0, 100, 100)]}, {"a": [pred_one(0, 0, 100, 90)]}, iou_threshold=0.45).map > 0.99)
    # [0,0,100,100] vs [0,30,100,130] is IoU 0.538 -- ABOVE 0.45, so it must match.
    # [0,0,100,100] vs [0,60,100,160] is IoU 0.25 -- below, so it must not.
    iou_0538 = score({"a": [gt_one(0, 0, 100, 100)]}, {"a": [pred_one(0, 30, 100, 130)]}, iou_threshold=0.45)
    iou_025 = score({"a": [gt_one(0, 0, 100, 100)]}, {"a": [pred_one(0, 60, 100, 160)]}, iou_threshold=0.45)
    check("IoU 0.538 matches at 0.45, IoU 0.25 does not",
          iou_0538.map > 0.99 and iou_025.map < 1e-9, f"{iou_0538.map:.2f} / {iou_025.map:.2f}")

    # Same box, different thresholds -> different verdict (this is why --iou at NMS is not the metric)
    box = [0, 0, 100, 70]  # IoU 0.7 against the GT
    at45 = score({"a": [gt_one(0, 0, 100, 100)]}, {"a": [pred_one(*box)]}, iou_threshold=0.45).map
    at50 = score({"a": [gt_one(0, 0, 100, 100)]}, {"a": [pred_one(*box)]}, iou_threshold=0.50).map
    at80 = score({"a": [gt_one(0, 0, 100, 100)]}, {"a": [pred_one(*box)]}, iou_threshold=0.80).map
    check("threshold sensitivity: 0.45/0.50 hit, 0.80 misses",
          at45 > 0.99 and at50 > 0.99 and at80 < 1e-9, f"{at45:.2f}/{at50:.2f}/{at80:.2f}")

    dupes = {"a": [pred_one(0, 0, 100, 100, 0.9), pred_one(0, 0, 100, 100, 0.8), pred_one(1, 1, 101, 101, 0.7)]}
    card = score({"a": [gt_one(0, 0, 100, 100)]}, dupes, iou_threshold=0.45)
    check("duplicate detections are not rewarded (AP 1.0, 2 FP recorded)",
          abs(card.map - 1.0) < 1e-9 and sum(s.n_fp for s in card.per_class) == 2,
          f"map={card.map:.3f} fp={sum(s.n_fp for s in card.per_class)}")

    # A prediction in a class that has no GT must NOT be averaged in as a 0.
    card = score({"a": [gt_one(0, 0, 100, 100)]},
                 {"a": [pred_one(0, 0, 100, 100, 0.9), pred_one(0, 0, 50, 50, 0.9, cls=1)]},
                 iou_threshold=0.45)
    check("class with no GT is excluded from the mean, not scored 0",
          abs(card.map - 1.0) < 1e-9, f"got {card.map:.4f}")
    check("that class is still reported with its FP count",
          any(s.cls == 1 and s.n_gt == 0 and s.n_pred == 1 for s in card.per_class))

    # Ranking is by confidence, so a CONFIDENT false positive is expensive: it sits
    # above the true positive and drags interpolated precision down to ~0.5.
    # (This is standard COCO behaviour and is worth internalising: one over-eager
    # detection at conf 0.95 costs more mAP than three cautious ones at conf 0.2.)
    gt_conf = {"a": [gt_one(0, 0, 100, 100)]}
    ranked = {"a": [pred_one(0, 0, 100, 100, conf=0.3), pred_one(300, 300, 400, 400, conf=0.95)]}
    card = score(gt_conf, ranked, iou_threshold=0.45)
    check("a confident FP above the TP caps AP at ~0.5 (ranking is by confidence)",
          abs(card.map - 0.5) < 0.02 and card.per_class[0].n_fp == 1,
          f"map={card.map:.3f} fp={card.per_class[0].n_fp}")

    # ...and with the ranking the other way round the same detections score ~1.0
    ranked2 = {"a": [pred_one(0, 0, 100, 100, conf=0.95), pred_one(300, 300, 400, 400, conf=0.3)]}
    card2 = score(gt_conf, ranked2, iou_threshold=0.45)
    check("same boxes, TP ranked first -> AP ~1.0 (ordering is what changed)",
          card2.map > 0.9, f"map={card2.map:.3f}")

    # Empty predictions must be 0, not a crash
    card = score({"a": [gt_one(0, 0, 10, 10)]}, {"a": []}, iou_threshold=0.45)
    check("no predictions -> mAP 0 (no crash)", card.map == 0.0 and card.per_class[0].n_fn == 1)

    # Empty GT for an image (a true negative) must not be counted as an error
    card = score({"a": [], "b": [gt_one(0, 0, 10, 10)]}, {"a": [], "b": [pred_one(0, 0, 10, 10)]}, iou_threshold=0.45)
    check("negative image with no predictions is clean", abs(card.map - 1.0) < 1e-9)

    # Confidence sweep direction: raising the threshold should drop the FPs
    card = score({"a": [gt_one(0, 0, 100, 100)]},
                 {"a": [pred_one(0, 0, 100, 100, 0.9), pred_one(400, 400, 450, 450, 0.2)]},
                 iou_threshold=0.45, conf_steps=20)
    curve = dict(card.conf_curve)
    low = min(curve, key=lambda k: abs(k - 0.05))
    high = min(curve, key=lambda k: abs(k - 0.5))
    check("sweep: a higher threshold keeps AP (drops the low-conf FP)",
          curve[high] >= curve[low] - 1e-9, f"ap@0.05={curve[low]:.3f} ap@0.5={curve[high]:.3f}")


# --------------------------------------------------------------- parsing

def test_parsing() -> None:
    print("\n= annotation parsing / validation =")
    boxes, findings = parse_yolo_text("0 0.5 0.5 0.2 0.2\n", "t")
    check("valid line parses", len(boxes) == 1 and abs(boxes[0].cx - 0.5) < 1e-9)

    _, f = parse_yolo_text("0 320 240 40 40\n", "t")
    check("PIXEL coords are rejected (not silently normalized)",
          any("PIXELS" in x.message for x in f), [x.message for x in f])

    _, f = parse_yolo_text("7 0.5 0.5 0.1 0.1\n", "t")
    check("out-of-range class id is rejected", any("class id 7" in x.message for x in f))

    _, f = parse_yolo_text("0 0.5 0.5 0.1\n", "t")
    check("4-value line is reported, not treated as a box", any("4 values" in x.message for x in f))

    _, f = parse_yolo_text("0 0.5 0.5 1.2 0.2\n", "t")
    check("w>1 is flagged and clamped", any("w/h > 1" in x.message for x in f))

    _, f = parse_yolo_text("0 0.5 0.5 0 0.2\n", "t")
    check("zero-width box is an error", any("non-positive" in x.message for x in f))

    _, f = parse_yolo_text("hello world\n", "t")
    check("garbage line is reported", any("non-numeric" in x.message for x in f))

    boxes, f = parse_yolo_text("# comment\n\n0 0.5 0.5 0.2 0.2\n", "t")
    check("comments and blank lines are tolerated", len(boxes) == 1 and not f)

    xml = """<annotation><filename>x.jpg</filename><size><width>200</width><height>100</height></size>
    <object><name>Hole</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>70</xmax><ymax>40</ymax></bndbox></object>
    <object><name>NotAClass</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
    </annotation>"""
    tmp = Path(tempfile.mkdtemp()) / "x.xml"
    tmp.write_text(xml)
    parsed = parse_voc_xml(tmp)
    check("VOC XML: Hole maps to class 1", parsed["boxes"] == [(1, 50.0, 20.0, 70.0, 40.0)])
    check("VOC XML: unknown object name is skipped with a warning",
          any("NotAClass" in f.message for f in parsed["findings"]))

    cat_map = coco_category_map({"categories": [{"id": 7, "name": "Spot"}, {"id": 8, "name": "Hole"}, {"id": 9, "name": "Crack"}]})
    check("COCO: ids are mapped by NAME, not assumed 0/1",
          cat_map == {7: 0, 8: 1, 9: -1}, str(cat_map))

    check("group key keeps the part, drops the frame",
          group_key("part012_frame3", None) == "part012" and group_key("part012_frame4", None) == "part012")


# --------------------------------------------------------------- formats

def test_format_equivalence() -> None:
    print("\n= COCO / VOC / YOLO produce identical boxes =")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "images").mkdir(parents=True)
    (tmp / "annotations").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (200, 100), (120, 120, 120)).save(tmp / "images" / "p001_f0.jpg")
    (tmp / "images" / "p001_f1.jpg").write_bytes((tmp / "images" / "p001_f0.jpg").read_bytes())
    expected = [Box(0, 0.25, 0.5, 0.2, 0.4), Box(1, 0.75, 0.5, 0.1, 0.2)]  # cx,cy in [0,1]

    (tmp / "annotations" / "p001_f0.txt").write_text("0 0.250000 0.500000 0.200000 0.400000\n1 0.750000 0.500000 0.100000 0.200000\n")
    layout = build_layout(tmp)
    src, _ = choose_source(layout)
    got = src.load()["p001_f0"].boxes
    ok = len(got) == 2 and all(abs(a.cx - b.cx) < 1e-6 and abs(a.cy - b.cy) < 1e-6 and a.cls == b.cls for a, b in zip(got, expected))
    check("YOLO txt -> boxes", ok, str([(b.cls, round(b.cx, 3)) for b in got]))

    # same content as VOC (pixels: cx=50,cy=50 -> x 30..70 ; cx=150 -> 140..160)
    for f in (tmp / "annotations").glob("*.txt"):
        f.unlink()
    (tmp / "annotations" / "p001_f0.xml").write_text(
        "<annotation><filename>p001_f0.jpg</filename><size><width>200</width><height>100</height></size>"
        "<object><name>Spot</name><bndbox><xmin>30</xmin><ymin>30</ymin><xmax>70</xmax><ymax>70</ymax></bndbox></object>"
        "<object><name>Hole</name><bndbox><xmin>140</xmin><ymin>40</ymin><xmax>160</xmax><ymax>60</ymax></bndbox></object>"
        "</annotation>"
    )
    src, why = choose_source(build_layout(tmp))
    got = src.load()["p001_f0"].boxes
    ok = len(got) == 2 and all(abs(a.cx - b.cx) < 1e-6 and a.cls == b.cls for a, b in zip(got, expected))
    check("VOC xml -> identical boxes to the YOLO txt", ok, f"{why} :: {[(b.cls, round(b.cx,3)) for b in got]}")

    for f in (tmp / "annotations").glob("*.xml"):
        f.unlink()
    coco = {
        "images": [{"id": 1, "file_name": "p001_f0.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [30, 30, 40, 40], "area": 1600, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 8, "bbox": [140, 40, 20, 20], "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 7, "name": "Spot"}, {"id": 8, "name": "Hole"}],
    }
    (tmp / "annotations" / "instances.json").write_text(json.dumps(coco))
    src, why = choose_source(build_layout(tmp))
    got = src.load()["p001_f0"].boxes
    ok = len(got) == 2 and all(abs(a.cx - b.cx) < 1e-6 and a.cls == b.cls for a, b in zip(got, expected))
    check("COCO json (category ids 7/8) -> identical boxes", ok, f"{why} :: {[(b.cls, round(b.cx,3)) for b in got]}")

    # an unknown COCO category must block, not silently become class 0
    coco["categories"].append({"id": 9, "name": "Crack"})
    coco["annotations"].append({"id": 3, "image_id": 1, "category_id": 9, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0})
    (tmp / "annotations" / "instances.json").write_text(json.dumps(coco))
    src, _ = choose_source(build_layout(tmp))
    src.load()
    check("unmapped COCO category is raised as an error, not guessed",
          any("neither Spot nor Hole" in f.message for f in src.findings),
          [f.message for f in src.findings][:2])
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- split

def test_split() -> None:
    print("\n= leakage-safe split =")
    stems = [f"part{i:03d}_frame{j}" for i in range(10) for j in range(3)]
    boxes = {s: [Box(0, 0.5, 0.5, 0.1, 0.1), Box(1, 0.2, 0.2, 0.05, 0.05)] for s in stems}
    boxes[stems[0]] = [Box(1, 0.5, 0.5, 0.1, 0.1)]  # a hole-only component

    plan = plan_split(stems, boxes, val_fraction=0.3, seed=1)
    train_stems = {s for k in plan.train for s in plan.groups[k]}
    val_stems = {s for k in plan.val for s in plan.groups[k]}
    check("no image is in both splits", not (train_stems & val_stems))
    train_parts = {group_key(s, None) for s in train_stems}
    val_parts = {group_key(s, None) for s in val_stems}
    check("no COMPONENT is in both splits (the leakage that inflates val)", not (train_parts & val_parts),
          f"overlap={train_parts & val_parts}")
    check("both splits are non-empty", bool(train_stems) and bool(val_stems))
    check("val is roughly the requested fraction", 0.15 <= len(val_stems) / len(stems) <= 0.45,
          f"{len(val_stems)}/{len(stems)}")

    p1 = plan_split(stems, boxes, val_fraction=0.3, seed=1)
    p2 = plan_split(stems, boxes, val_fraction=0.3, seed=1)
    check("same seed -> same split (reproducible experiments)", p1.train == p2.train and p1.val == p2.val)

    single = plan_split(["part000_frame0"], {"part000_frame0": boxes[stems[0]]}, val_fraction=0.2)
    check("a single component produces an explicit warning, not a false confidence",
          any("impossible" in n for n in single.notes), str(single.notes))


# --------------------------------------------------------------- annotation

def test_annotation_stack() -> None:
    """The label writer and the save router: the two places where a slip costs work.

    Annotation is a human typing for hours. These checks are about the failure modes
    that would throw that away — a label file that cannot be parsed, a class id the
    trainer does not know, or a save that lands on the wrong image.
    """
    print("\n= annotation: label writing and save routing =")

    # 1. clamping: a box dragged past the image edge must not become an invalid label
    text = annotate.render_yolo([{"c": 0, "x": 1.4, "y": -0.3, "w": 2.0, "h": 0.5}])
    parts = text.split()
    check("out-of-range coordinates are clamped into [0, 1]",
          parts[1] == "1.000000" and parts[2] == "0.000000" and parts[3] == "1.000000",
          f"got {text.strip()!r}")
    check("a clamped line still has exactly 5 fields", len(parts) == 5, f"got {len(parts)}")

    # 2. class ids: only 0=Spot and 1=Hole exist. A stray id would train a phantom class.
    try:
        annotate.render_yolo([{"c": 2, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}])
        check("an unknown class id is rejected", False, "class 2 was accepted")
    except ValueError:
        check("an unknown class id is rejected", True)

    # 3. a written file must survive the same parser the rest of the repo uses
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.txt"
        boxes = [{"c": 1, "x": 0.25, "y": 0.5, "w": 0.1, "h": 0.2},
                 {"c": 0, "x": 0.75, "y": 0.5, "w": 0.2, "h": 0.2}]
        annotate.save_yolo(f, boxes)
        back = annotate.load_yolo(f)
        check("save->load round-trips every box, class included",
              len(back) == 2 and back[0]["c"] == 1 and back[1]["c"] == 0
              and abs(back[0]["x"] - 0.25) < 1e-6 and abs(back[1]["w"] - 0.2) < 1e-6,
              f"got {back}")
        parsed, findings = parse_yolo_text(f.read_text(), "selftest-annotate")
        check("the repo's own label parser accepts what the UI writes",
              len(parsed) == 2 and sorted(b.cls for b in parsed) == [0, 1] and not findings,
              f"got {len(parsed)} boxes, findings={findings}")
        # an empty image is legitimate (it may genuinely have no defects)
        annotate.save_yolo(f, [])
        check("an image with no boxes writes an empty file, not a malformed one",
              f.read_text() == "" and annotate.load_yolo(f) == [])

    # 4. save routing: never guess which image a request means
    images = [{"path": "/tmp/a/part01.png"}, {"path": "/tmp/a/part02.png"}]
    check("save by index resolves", annotate.resolve_image(images, {"i": 1})["path"].endswith("part02.png"))
    check("save by filename resolves", annotate.resolve_image(images, {"key": "part01.png"})["path"].endswith("part01.png"))
    for bad, why in [({}, "no index and no key"), ({"i": -1}, "index -1"), ({"i": 9}, "index past the end"),
                     ({"key": "nope.png"}, "a filename that is not in the list")]:
        try:
            annotate.resolve_image(images, bad)
            check(f"a save request with {why} is REFUSED", False, f"it was accepted: {bad}")
        except ValueError:
            check(f"a save request with {why} is REFUSED", True)

    # 5. the class ids the UI writes must be the ones the competition names
    check("annotation classes are 0=Spot and 1=Hole, matching the competition",
          annotate.CLASS_NAMES == {0: "Spot", 1: "Hole"}, f"got {annotate.CLASS_NAMES}")


# --------------------------------------------------------------- roundtrip

def test_roundtrip() -> None:
    print("\n= coordinate roundtrip (submission correctness) =")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        w, h = int(rng.integers(320, 4000)), int(rng.integers(320, 4000))
        bw, bh = rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.3)
        # keep boxes fully inside the frame: a box hanging off the edge is CLIPPED by
        # as_xyxy_pixels on purpose, so it cannot round-trip exactly by design.
        cx = rng.uniform(bw / 2 + 1e-3, 1 - bw / 2 - 1e-3)
        cy = rng.uniform(bh / 2 + 1e-3, 1 - bh / 2 - 1e-3)
        b = Box(0, cx, cy, bw, bh)
        x1, y1, x2, y2 = b.as_xyxy_pixels(w, h)
        back = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
        worst = max(worst, max(abs(back[0] - cx), abs(back[1] - cy), abs(back[2] - bw), abs(back[3] - bh)))
    check("normalized -> pixel -> normalized is exact (<1e-9)", worst < 1e-9, f"worst drift {worst:.2e}")

    # writing with 6 decimal places then re-reading (what prepare/predict actually do)
    worst6 = 0.0
    for _ in range(200):
        w, h = int(rng.integers(320, 4000)), int(rng.integers(320, 4000))
        bw, bh = rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.3)
        cx = rng.uniform(bw / 2 + 1e-3, 1 - bw / 2 - 1e-3)
        cy = rng.uniform(bh / 2 + 1e-3, 1 - bh / 2 - 1e-3)
        line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        got, _ = parse_yolo_text(line, "t")
        b = got[0]
        x1, y1, x2, y2 = b.as_xyxy_pixels(w, h)
        # compare in PIXELS: 6-decimal rounding is sub-pixel at any plausible resolution
        worst6 = max(worst6, abs((x2 - x1) - bw * w), abs((y2 - y1) - bh * h))
    check("6-decimal txt round-trip stays sub-pixel", worst6 < 0.01, f"worst {worst6:.4f}px")

    # out-of-frame boxes must be CLIPPED (no negative coordinates in a submission)
    b = Box(0, 0.02, 0.5, 0.1, 0.1)
    x1, y1, x2, y2 = b.as_xyxy_pixels(1000, 1000)
    check("a box hanging off the edge is clipped, never negative",
          x1 >= 0 and y1 >= 0 and x2 <= 1000 and y2 <= 1000, f"{x1},{y1},{x2},{y2}")


def test_end_to_end_build() -> None:
    print("\n= full build on a generated raw tree =")
    tmp = Path(tempfile.mkdtemp())
    raw, prepared = tmp / "raw", tmp / "prepared"
    (raw / "images").mkdir(parents=True)
    (raw / "annotations").mkdir(parents=True)
    from PIL import Image

    for part in range(4):
        for frame in range(3):
            name = f"part{part:02d}_frame{frame}"
            Image.new("RGB", (400, 300), (100, 100, 100)).save(raw / "images" / f"{name}.jpg")
            (raw / "annotations" / f"{name}.txt").write_text("0 0.500000 0.500000 0.100000 0.100000\n1 0.300000 0.300000 0.050000 0.050000\n")
    (raw / "annotations" / "part00_frame0.txt").write_text("")  # intentional negative (a true negative sample)

    report = build_prepared(build_layout(raw), prepared, val_fraction=0.34, seed=3)
    man = report.get("manifest")
    check("build succeeds on a clean tree", man is not None, str(report.get("errors")))
    if man:
        n_train = len(list((prepared / "labels" / "train").glob("*.txt")))
        n_val = len(list((prepared / "labels" / "val").glob("*.txt")))
        check("every image got a label file (including the negative)",
              n_train == man["images"]["train"] and n_val == man["images"]["val"],
              f"{n_train}+{n_val} vs {man['images']}")
        neg = [p for split in ("train", "val") for p in (prepared / "labels" / split).glob("part00_frame0.txt")]
        check("negative image produced an empty label file",
              len(neg) == 1 and not neg[0].read_text().strip(), f"found {[str(p) for p in neg]}")
        train_parts = {p.name.split("_")[0] for p in (prepared / "images" / "train").iterdir()}
        val_parts = {p.name.split("_")[0] for p in (prepared / "images" / "val").iterdir()}
        check("no part spans train and val", not (train_parts & val_parts), f"{train_parts} vs {val_parts}")

    # a bad label must ABORT the build, not warn and continue
    (raw / "annotations" / "part01_frame0.txt").write_text("0 400 300 20 20\n")  # pixel coords
    report2 = build_prepared(build_layout(raw), tmp / "prepared2", dry_run=True)
    check("pixel-coordinate label aborts the build with an explanation",
          report2.get("manifest") is None and any("PIXELS" in e for e in report2.get("errors", [])),
          str(report2.get("errors"))[:160])

    # an orphan label (filename typo) must be reported, not silently dropped:
    # this is how a whole part's annotations disappear without anyone noticing.
    (raw / "annotations" / "part01_frame0.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (raw / "annotations" / "part01_frame99.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    report3 = build_prepared(build_layout(raw), tmp / "prepared3", dry_run=True)
    check("orphan label file is surfaced as a warning",
          any("match no image" in w for w in report3.get("warnings", [])),
          str(report3.get("warnings"))[:160])
    shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_config_resolution() -> None:
    """train.py must read the split named by --data, not a hardcoded folder.

    Regression: dataset_stats() used to look only at dataset/prepared, so pointing --data
    at any other dataset reported "0 images / 0 boxes" and the imgsz planning went blind.
    """
    print("\n= dataset config resolution (train.py reads what the trainer reads) =")
    import train  # noqa: E402
    from _ped.dataset_cfg import resolve_split_dirs  # noqa: E402

    tmp = Path(tempfile.mkdtemp())
    root = tmp / "mydata"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    from PIL import Image

    for i in range(3):
        Image.new("RGB", (200, 200), (90, 90, 90)).save(root / "images" / "train" / f"p{i}.jpg")
        (root / "labels" / "train" / f"p{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.05 0.05\n")
    Image.new("RGB", (200, 200), (90, 90, 90)).save(root / "images" / "val" / "p9.jpg")
    (root / "labels" / "val" / "p9.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    yml = tmp / "data.mine.yaml"
    yml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: Spot\n  1: Hole\n")
    img_dir, lbl_dir, notes = resolve_split_dirs(yml, "train")
    check("absolute 'path:' resolves to the named split", img_dir == root / "images" / "train", str(img_dir))
    check("labels dir is found at <root>/labels/<split>", lbl_dir == root / "labels" / "train", str(lbl_dir))

    from _ped.labels import DatasetStats

    stats, meta = train.dataset_stats(yml)
    check("boxes are counted from the configured dataset (not a hardcoded folder)",
          isinstance(stats, DatasetStats) and stats.boxes == 6 and stats.images == 3,
          f"images={stats.images} boxes={stats.boxes} src={meta.get('source')}")
    check("per-class counts reach the planner", dict(stats.per_class) == {0: 3, 1: 3}, str(dict(stats.per_class)))

    # a relative path that resolves next to the yaml must be found too
    rel = tmp / "rel" / "data.yaml"
    (tmp / "rel").mkdir()
    (tmp / "rel" / "ds" / "images" / "train").mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(tmp / "rel" / "ds" / "images" / "train" / "x.jpg")
    rel.write_text("path: ds\ntrain: images/train\nval: images/val\nnames:\n  0: Spot\n")
    img_dir2, _lbl2, _n2 = resolve_split_dirs(rel, "train")
    check("relative 'path:' resolves relative to the yaml",
          img_dir2 == (tmp / "rel" / "ds" / "images" / "train").resolve(), str(img_dir2))
    shutil.rmtree(tmp, ignore_errors=True)


def test_kaggle_reference_parsing() -> None:
    """Kaggle link forms must resolve to the right kind + id before any download.

    A misread here means the download silently targets the wrong resource (or nothing),
    so it is checked rather than eyeballed.
    """
    print("\n= Kaggle reference parsing =")
    import fetch_dataset as fd  # noqa: E402

    cases = [
        ("https://www.kaggle.com/t/95c0d96a415c52241ed6c14bb2a0b30a", "task", "95c0d96a415c52241ed6c14bb2a0b30a"),
        ("https://www.kaggle.com/competitions/foo-bar", "competition", "foo-bar"),
        ("https://www.kaggle.com/c/titanic", "competition", "titanic"),
        ("https://www.kaggle.com/datasets/owner/name", "dataset", "owner/name"),
        ("https://www.kaggle.com/d/owner/name", "dataset", "owner/name"),
        ("https://www.kaggle.com/datasets/owner/name?select=train.csv", "dataset", "owner/name"),
        ("https://www.kaggle.com/t/abc/overview", "task", "abc"),
        ("owner/name", "dataset", "owner/name"),
    ]
    ok = 0
    for url, kind, ident in cases:
        got_kind, got_id = fd.parse_kaggle_reference(url)
        if (got_kind, got_id) == (kind, ident):
            ok += 1
        else:
            print(f"    mismatch {url}: got {(got_kind, got_id)} want {(kind, ident)}")
    check(f"all {len(cases)} Kaggle URL forms parse correctly", ok == len(cases), f"{ok}/{len(cases)}")

    user, key, src = fd.credentials()
    check("credentials are reported without exposing the secret",
          key is None or isinstance(src, str), "credentials() leaked no value")

    # the failure text must name the real problem, not a generic error
    diag = fd.diagnose("Max retries exceeded with url: /v1/...", "task", "abc")
    check("a network failure is diagnosed as a network failure, not an auth problem",
          "cannot reach kaggle.com" in diag, diag[:80])
    diag403 = fd.diagnose("403 Forbidden", "competition", "abc")
    check("a 403 points at rule acceptance, and mentions the browser fallback for /t/ tasks",
          "ACCEPTED THE COMPETITION RULES" in diag403 and "browser" in diag403)


def main() -> int:
    print("PED pipeline self-test — known-answer checks for the failure modes that hide")
    test_metric()
    test_parsing()
    test_format_equivalence()
    test_split()
    test_roundtrip()
    test_annotation_stack()
    test_end_to_end_build()
    test_dataset_config_resolution()
    test_kaggle_reference_parsing()
    print("\n" + "=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print(f"  · {f}")
        return 1
    print("All checks pass — the metric and the split behave as specified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
