# PED Hackathon — visual inspection of Spot welds and Holes

Detect two classes on metal components and maximise **mAP@45**:

| id | class  | what it is |
|----|--------|------------|
| 0  | `Spot` | spot weld nugget |
| 1  | `Hole` | hole through the material |

This repo is built around one idea: **the dataset and the machine decide the training
config, not the other way round.** Run `audit_dataset.py` first; it prints the single
next command for *your* data instead of assumptions about it.

Nothing here invents labels. If the annotations are missing or malformed, the pipeline
stops and says so — a detector trained on guessed boxes (e.g. "every dark circle is a
Hole") scores worse than one trained on fewer, correct boxes, and you find out too late.

---

## Quick start

```bash
python -m pip install -r requirements.txt          # add opencv-python-headless on any headless box
python scripts/audit_dataset.py                    # 1. where do we stand?  ← START HERE
python scripts/prepare_dataset.py --dry-run        # 2. what would the conversion do?
python scripts/prepare_dataset.py                  #    build dataset/prepared + configs/data.prepared.yaml
python scripts/validate_labels.py                  # 3. label gate + contact sheet you must look at
python scripts/train.py --dry-run                  # 4. the config chosen for THIS machine
python scripts/train.py                            #    train
python scripts/select_best.py --run ped_baseline    # 5. pick the checkpoint by mAP@45, not by mAP@50:95
python scripts/evaluate.py --name ped_baseline      #    Ultralytics metrics + the real mAP@45
python scripts/submit.py --weights runs/ped_baseline/weights/best.pt --source <test dir> --format <?>
```

No data yet? Prove the plumbing on generated images first:

```bash
python scripts/selftest.py                                                    # 44 known-answer checks, no data needed
python scripts/make_synthetic_data.py --out dataset/_synthetic_demo/raw       # fake parts, looks structurally real
python scripts/prepare_dataset.py --raw dataset/_synthetic_demo/raw --out dataset/_synthetic_demo/prepared
```

### Status of the 9-point plan

| # | plan item | script | state |
|---|---|---|---|
| 0 | put originals in `dataset/raw/` | — | your action; drop-zone is ready and gitignored |
| 1 | install dependencies | `requirements.txt` | done, incl. the headless-opencv trap |
| 2 | run the audit | `audit_dataset.py` | done — images, labels, GPU, imgsz, next command |
| 3 | confirm format + class mapping | `audit_dataset.py` §4–5 | done — peeks the raw file by eye, maps COCO/VOC names explicitly |
| 4 | leakage-safe train/val split | `prepare_dataset.py` | done — grouped by component, stratified, seeded |
| 5 | train the baseline | `train.py` | done — config derived from the audit, recorded per run |
| 6 | inspect predictions / per-class errors | `map45.py`, `evaluate.py` | done — per-class AP, worst-image list, confusion matrix |
| 7 | evaluate with the official mAP@45 | `map45.py`, `select_best.py` | done — own COCO-style implementation + unit tests (`selftest.py`); checkpoint selection corrected |
| 8 | test the submission format | `submit.py` | done — 5 formats, coordinates in original pixels |

Verified end-to-end on generated data in this sandbox (no GPU, 2 CPU cores, 31 train images,
40 epochs from scratch — see the note on the blocked weight CDN below):

```
HEADLINE mAP@45 = 0.4271     Ultralytics mAP@50 = 0.4096
per class:  Spot AP@45 0.854 (53 GT)   Hole AP@45 0.000 (16 GT)   ← the lesson
```

Spot is found; **Hole is not found at all**. On a from-scratch model the minority class is
exactly what a single mAP number hides — and it is what the next iteration has to attack
(more Hole annotations, weight the loss, or raise resolution for the small objects).

---

## The one metric that matters

Ultralytics reports **mAP@50** and **mAP@50:95**. Neither is the competition number, and
`model.predict(iou=...)` is NMS, not an evaluation threshold. `scripts/map45.py`
computes **mAP@45** itself: one-to-one greedy matching by descending confidence, 101-point
interpolated AP, mean over classes that have ground truth (COCO conventions), and a
confidence sweep so the submission threshold is *measured* rather than defaulted to 0.25.

```
class      AP@0.45     GT   pred  missed    FP      P      R
------------------------------------------------------------
Spot        0.8700     53    ...
Hole        0.6100     16    ...
------------------------------------------------------------
mAP         0.7400     69   ...
```

The per-class split is the useful part: with `Spot` far above `Hole`, your score is gated
by the minority class, and no amount of extra epochs on the majority class will fix it.

**Full explanation of the metric, how the IoU threshold changes what counts as correct,
and why mAP@45 is an evaluation metric rather than a training objective:
[`docs/metric_guide.md`](docs/metric_guide.md).**

### One trap worth knowing up front

Ultralytics chooses `best.pt` — and decides when to stop early — using
`mAP@50:95`, not your competition metric (verified in
`ultralytics/models/yolo/detect/val.py`). On a small dataset those rankings disagree.
`select_best.py` re-scores every saved checkpoint on mAP@45; in a 24-epoch test run the
gap between checkpoints was `0.1073` vs `0.2198`, i.e. **11 points of score** depending
on which file you submit.

---

## Pipeline

| script | what it does | when it fails loudly |
|---|---|---|
| `audit_dataset.py` | discovers images/labels, validates coordinates and class ids, profiles the GPU, proposes an imgsz/model, prints the next command | no images, no usable labels, pixel-space coords, unknown class ids, orphan labels |
| `prepare_dataset.py` | converts YOLO/COCO/VOC/CSV → canonical YOLO, group-aware split, absolute-path `data.prepared.yaml` | any error-level label problem; it refuses to write |
| `validate_labels.py` | per-image/per-class gate on the *built* dataset + contact sheet with boxes drawn | missing/orphan/duplicate labels, degenerate boxes, manifest disagreement |
| `train.py` | hardware-adaptive training, records `config.json` per run | no data, no GPU (unless `--allow-cpu`/`--smoke`) |
| `map45.py` | the scored metric + confidence sweep + worst-image report | missing GT, unreadable predictions |
| `select_best.py` | scores every saved checkpoint on mAP@45 and copies the winner | refuses to name a winner from noise or from an untrained run |
| `evaluate.py` | Ultralytics val for PR curves/confusion matrix, then calls `map45.py` | missing weights |
| `predict.py` | inference, optional `--tile` for tiny targets on large frames | unreadable images |
| `submit.py` | yolo / kaggle / coco / csv / xml output in **original image pixels** | — |
| `selftest.py` | known-answer tests for metric, parsing, formats, split, coordinate roundtrip | — |

### Detail worth reading

**The split is by component, not by image.** `part012_frame3` and `part012_frame7` are
the same physical part, three frames apart. Random per-image splitting puts them on both
sides and your validation mAP becomes a memorisation score. Override the heuristic with
`--group-regex '(part\d+)'` when the naming is not guessable.

**`data.prepared.yaml` uses absolute paths on purpose.** Ultralytics resolves a relative
`path:` in a dataset YAML against its global `datasets_dir`, *not* against the YAML's own
location, so a relative path that looks right in the file can silently point elsewhere —
the classic "Dataset not found" on a second machine.

**Empty label files are kept, missing ones are not.** An empty `.txt` = "this part
genuinely has no spot or hole" (a useful negative). A *missing* `.txt` = probably an
annotator who did not finish, which is why coverage below ~98% aborts the build.

**Small targets need a decision, not a default.** `audit_dataset.py` reports the median
and the smallest quartile of box heights, then picks the cheapest imgsz that keeps the
small quartile above ~12px (the stride-32 floor). If even 1280 cannot, it says so and
points at `predict.py --tile`.

---

## Hardware notes

- `pip install ultralytics` does **not** guarantee GPU support. Verify:
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
  `False` on a GPU box means a CPU-only wheel or a CUDA/driver mismatch.
- **Headless machines**: ultralytics pulls `opencv-python` (GUI build), which dies with
  `ImportError: libGL.so.1: cannot open shared object file`. Install
  `opencv-python-headless` instead of installing libGL, and remove the GUI package if
  both are present — it shadows the headless one.
- Batch size is derived from **free** VRAM (not total) minus ~1.2 GiB for the CUDA context,
  capped conservatively. Raise it only after one clean epoch.
- The sandbox this repo was built in has no GPU and cannot reach the pretrained-weight CDN,
  so `train.py` falls back to building the architecture from YAML and prints a warning.
  From-scratch training is a fallback, not a plan: on a normal machine `yolo11n.pt`
  downloads automatically and gives a large head start on a small dataset.

---

## What actually moves mAP@45, in order

1. **Label correctness and coverage** — swapped class ids or a half-annotated part cost
   more than any hyperparameter. `validate_labels.py` exists to catch this; look at the sheet.
2. **Leakage-free validation** — a val score that is too good means the split leaked.
3. **The minority class** — check the per-class table before and after every change.
4. **Input resolution / tiling** — only if targets are small relative to the frame.
5. **Model size and epochs** — usually last, and only once 1–4 are settled.

---

## Repo layout

```
configs/data.prepared.yaml   generated, absolute paths, this is what training reads
configs/data.yaml            hand-written fallback
dataset/raw/                 originals — treated as read-only by every script
dataset/prepared/            canonical YOLO output + manifest.json (seed, groups, counts)
dataset/prepared/manifest.json  what was converted, how it was split, box counts per split
runs/                        training runs, reports, label sheets, submissions
scripts/_ped/                discovery, label parsing, hardware planning, metrics, split
```

`manifest.json` is the audit trail: seed, component groups per split, and box counts per
class and per split. `validate_labels.py` re-checks the built files against it, which is
how a converter bug is caught before it becomes a training run.
