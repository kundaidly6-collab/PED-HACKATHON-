# mAP@50 — what it is, and where it belongs in the workflow

## The short version

**mAP@50 is an evaluation metric, not a training objective.** You do not train *with* it.
You train by minimising a loss, then you *measure* mAP@50 on held-out images to decide
which model to submit and at what confidence threshold. Those are different jobs and it
matters that they stay separate.

**@50 is the competition's number.** Every default in this repo is 0.50. Other thresholds
(0.45, 0.75) are available through `--iou` as diagnostics, and the scorer always prints
which one it used — a number quoted without its threshold is meaningless.

## What "mAP" means, built up one piece at a time

**1. IoU — how well one box overlaps another**

```
IoU = (area of overlap) / (area of union)
```

`1.0` = identical boxes, `0.0` = no overlap. It is the only thing that decides whether a
prediction counts as correct.

**2. TP vs FP — the matching rule**

For one image and one class, each predicted box is compared against the unclaimed
ground-truth boxes:

- IoU ≥ 0.50 with an unmatched ground-truth box → **true positive**, and that GT box is now claimed
- otherwise → **false positive**
- any ground-truth box never claimed → **false negative** (a miss)

Matching is done highest-confidence-first and is strictly one-to-one, so a second box on
the same object is a false positive, not a second hit.

**3. Precision and recall**

- Precision = TP / (TP + FP) — of what I flagged, how much was real
- Recall = TP / (TP + FN) — of what was real, how much I found

Both depend on where you set the confidence threshold, so you cannot quote one number.

**4. AP — the whole precision/recall curve in one number**

Rank every detection by confidence, walk down the list, and track precision and recall as
you go. AP is the area under that curve (this repo uses COCO's 101-point interpolation).
AP answers: *across all confidence settings, how good is this class?*

**5. mAP — average over classes**

`mAP = mean(AP)` across classes that have ground truth. Here that is
`(AP_Spot + AP_Hole) / 2`.

**6. The @50 — the IoU threshold used for matching**

Every "TP" above required IoU ≥ 0.50. Score the same predictions at a different threshold
and you get a different number. That is the whole meaning of the suffix, and why the
scorer prints it beside every result.

## Why 0.50, and when to look at 0.45 or 0.75

0.50 is the PASCAL VOC convention and the threshold this competition scores at: a
prediction must overlap its ground-truth box by at least half of their combined area to
count. The threshold sets **how precisely a box must be placed to count**. Comparison,
for a square box shifted along one axis:

| object size | max offset allowed @0.50 | @0.45 (looser) | @0.75 (stricter) |
|---|---|---|---|
| 10 px | 3.3 px | 3.8 px | 1.4 px |
| 20 px | 6.7 px | 7.6 px | 2.9 px |
| **48 px** (the measured holes) | **16.0 px** | 18.2 px | 6.9 px |
| 100 px | 33.3 px | 37.9 px | 14.3 px |
| 300 px | 100.1 px | 113.8 px | 42.9 px |

Read the left column against the middle: **a fixed pixel error means something different
at every scale.** 3 px of slop is harmless on a 48 px hole (IoU ≈ 0.88) and fatal on a
10 px one (IoU ≈ 0.54 at 0.45, failing outright at 0.75).

Two consequences worth internalising:

- **@50 is stricter than @45, and it is the scored one.** A box that scores at one can
  score zero at the other, because the match decision is a step function. Measured on
  this repo's own scorer: a single box at IoU 0.493 gives **mAP@50 = 0.000 but
  mAP@45 = 1.000**. A model tuned by reading @45 will look better than it scores.
- **@75 is a different question entirely** — "is the box tight?" rather than "is the
  object found?". Useful for spotting a localisation problem behind a healthy @50, never
  a substitute for the scored number.

For this dataset: the ~48 px mounting holes allow ~16 px of offset at @50, so the score
will be driven by whether the *small* features (the weld/bolt spot, and any small holes in
other images) are detected at all — not by sub-pixel box precision. **Recall on the
minority class is the lever, not box tightness.**

## Where mAP@50 does and does not act in this repo

| stage | used? | why |
|---|---|---|
| gradient descent | **no** | mAP is not differentiable: it contains a discrete match/no-match decision and a sort by confidence, so there is no usable gradient. YOLO trains on a loss (box regression + classification + DFL). |
| early stopping | **not by default** | Ultralytics stops on its own fitness = mAP@50:95 (see below) |
| `best.pt` selection | **not by default** | same fitness; `scripts/select_best.py` fixes this |
| model comparison | **yes** | `scripts/eval_map.py` — same metric for every experiment |
| confidence threshold | **yes** | the sweep picks the threshold that maximises mAP@50 on validation |
| submission | **yes, indirectly** | you submit boxes; their mAP@50 is your score |

## The trap: Ultralytics' "best.pt" is not chosen by this metric

Verified in `ultralytics/models/yolo/detect/val.py`:

```python
stats["fitness"] = stats["metrics/mAP50-95(B)"]   # always use box mAP50-95 for fitness
```

So the checkpoint the trainer calls `best.pt`, and the point at which early stopping
fires, are chosen by **mAP@50:95** — a stricter metric that averages AP over IoU
thresholds from 0.50 to 0.95. Note carefully what that is *not*: it is not mAP@50 either,
even though it contains 0.50 in its name. On a small dataset the two rankings can
disagree, and the disagreement costs score.

`scripts/select_best.py` resolves this by scoring every checkpoint the run saved
(`save_period`) with the same mAP@50 implementation used everywhere else:

```bash
python scripts/select_best.py --run ped_baseline        # table + copy winner to best_metric.pt
```

It reports a **tie** rather than a winner when checkpoints are within 0.005 of each
other, and reports **no usable checkpoint** rather than picking one when every
checkpoint scores near zero — because naming a winner out of noise is exactly the false
confidence that costs a hackathon.

## Why you should always read the per-class columns

A single mAP number hides the thing that decides your next action. From this repo's
end-to-end synthetic demo:

```
checkpoint    metric     Spot     Hole   missed    FP
best.pt       0.4271   0.8540   0.0000       ...     ...   <- measured at IoU 0.45
```

> **Threshold warning on that row.** It was recorded at **@0.45**, before the competition
> threshold was confirmed as @50, so it is *not* a competition score and it is optimistic:
> the same predictions score **≤ 0.4271 at @50**, and the exact figure has not been
> re-measured. `scripts/select_best.py` re-scores every checkpoint at the current default,
> so any table you generate from here on is @50. Never quote a number without its
> threshold — that is precisely the mistake this row would cause if it went unlabelled.

The `Hole = 0.0000` half of it *is* threshold-independent: no Hole prediction was matched
at any threshold, so its AP is zero regardless. Mean of 0.85 and 0.00 is 0.43. The
average says "roughly half marks"; the split says
**"Spot works, Hole is completely absent"** — a data and imbalance problem where extra
epochs on the majority class will do nothing. Always look at both columns, and note that
the minority class is usually what gates the mean.
