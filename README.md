# PED Hackathon — Spot & Hole detection on pressed metal components

**Goal: train the best possible detector for the two target classes, measured by
mAP@50. A higher mAP@50 means better detection performance — that is the number this
project optimises for, and the only number that decides the ranking.**

> Training images are supplied **without labels**, so annotation is part of the task, not
> a given: `scripts/propose_labels.py` proposes boxes and `scripts/annotate.py` is the UI
> where a human confirms them. Nothing proposed is a training label until it is reviewed.

| id | class | what it is |
|----|-------|-----------|
| 0 | `Spot` | bolt, spot-weld nugget or weld feature on the component |
| 1 | `Hole` | nut, or any physical hole / opening in the component |

### The organizers' annotation rules, and what this repo does with them

The rulebook maps words to classes, and the mapping is not the one your instinct
suggests — **a nut is a `Hole`, and a bolt is a `Spot`**:

| you see | you annotate |
|---|---|
| Nut | `Hole` |
| physical hole / opening | `Hole` |
| Bolt | `Spot` |
| weld spot / spot-welding feature | `Spot` |

There is **no `Punch` class** and no third class. Everything in this repo uses those two.

The test-set annotations are hidden and submissions are scored automatically. Two
sentences in the rules decide how you are scored, and both are worth reading twice
because they disagree with each other and with the obvious reading.

### The metric sentence contradicts itself

> "Submissions are evaluated using **mAP@50 (mean Average Precision at IoU 0.45)**."
> "The Intersection over Union (IoU) ... is at least **0.45**."

The name says 50. The definition says 0.45. Matching is a **step function**, not a
slope, so the gap is not cosmetic — measured on this repo's own scorer, one prediction
at IoU 0.4925 scores:

```
IoU >= 0.50 : mAP 0.0000
IoU >= 0.45 : mAP 1.0000
```

Same boxes, same confidence, opposite conclusion. So `eval_map.py` **reports both
readings from a single run** and `--iou` stays a first-class parameter (default 0.50,
the stricter one). Tuning against the stricter threshold is safe under either reading,
because AP at 0.45 is never *lower* than AP at 0.50 — the 0.45 number is the optimistic
end of a range, never a surprise. This is the single best question to put to the
organizers; until it is answered, quote both numbers, never one.

### The submission carries class NAMES, so the id order cannot hurt you

The rules fix the class **names** (`Hole`, `Spot`) and never publish a numeric order.
`Spot`=0, `Hole`=1 is **this repo's** convention — consistent across `data.yaml`, the
labels, the annotator and the trainer, which is all training needs. And because the
submission format carries the class as a **string** (`"class":"Hole"`), that internal
convention cannot leak into the scored file. `submit.py` still prints the mapping it
uses and can emit a legend or flip the order (`--class-order hole-first`) for the
auxiliary id-based exports, but for the real submission this risk is retired.

**Team entries** (nothing in this repo can do it for you): 2–5 members per team, at least
one Section Head, each member joins the competition with their own Kaggle account and
accepts the team invitation, and a participant may belong to only one team.

### The submission file, exactly

One CSV, two columns, predictions as a JSON list per image:

```
ID,predictions
image001.jpg,"[{""class"":""Hole"",""xcenter"":0.52,""ycenter"":0.43,""width"":0.18,""height"":0.21,""confidence"":0.91}]"
image_003.jpg,"[]"
```

| rule | how this repo honours it |
|---|---|
| exactly two columns, `ID` and `predictions` | `--format ped` (the default) writes only this |
| ID = the **exact** test filename | the ID is taken from the file on disk, extension included |
| every test image exactly once | `--test-images <folder>` drives the rows, so coverage is total |
| no detections → `[]` | empty images are written as `[]`, never omitted |
| class is `Hole` or `Spot` | the class **name** is written, never the numeric id |
| coords normalized 0–1 | converted back from each image's own pixel size |
| width, height **not zero** | a box that rounds to zero is dropped and counted |
| confidence 0–1 | clamped, and validated |

```bash
python scripts/submit.py --weights runs/ped_baseline/weights/best_metric.pt     --source dataset/raw/test/images --test-images dataset/raw/test/images     --out runs/submission.csv
python scripts/validate_submission.py --csv runs/submission.csv     --test-images dataset/raw/test/images     # exit 0 only if a grader would accept it
```

`validate_submission.py` checks every rule above and refuses the file otherwise: a
missing image row, a JSON field that was never quoted (so the commas split the row into
seven columns), `"hole"` in the wrong case, a class given as a training id, a width of
zero, pixel coordinates where normalized ones belong, a confidence of 1.4, a duplicated
ID, the wrong header, a box hanging off the edge (legal, but flagged). Each of those
scores zero, and none of them is visible by looking at the file.

**One ambiguity remains in the JSON keys.** The field list says `x_center` / `y_center`;
the worked example and the sentence *"xcenter and ycenter represent the center"* say the
plain form. `--keys plain` (default) or `--keys underscore` picks one, and the validator
reports which style a file uses. Whichever you choose, confirm it before the deadline —
a strict reader keying on the other spelling would find no coordinates at all. A cheap
way to settle both this and the metric question: submit one of each early and compare
leaderboard scores, rather than discovering it at the end.

---

## 1. The project in one page

### The problem

Given photographs of pressed sheet-metal components, draw a bounding box around every
**spot weld** and every **hole**. It is an object-detection task with a small number of
classes but a large number of small instances per image, on shiny, low-contrast metal
where the same physical feature can look completely different from one lighting angle to
the next.

### The metric, and why it is not just "accuracy"

mAP@50 is **mean Average Precision at IoU threshold 0.50**. Reading it from the inside out:

1. **IoU** (intersection over union) measures how well a predicted box overlaps the true box. `1.0` = identical, `0.0` = no overlap.
2. A prediction counts as a **true positive** only if its IoU with an unmatched ground-truth box is **≥ 0.50**. Otherwise it is a **false positive**. Any ground-truth box never matched is a **miss**.
3. Matching is done **highest-confidence-first** and is strictly one-to-one, so a second box on the same object is a false positive, not a second hit.
4. **AP** for a class is the area under the precision–recall curve obtained by sweeping the confidence threshold — it summarises every possible threshold in one number.
5. **mAP** is the mean of AP over the classes that have ground truth. Here: `(AP_Spot + AP_Hole) / 2`.

So **mAP@50 rewards finding objects and penalises only genuinely sloppy boxes.** It is
the classic PASCAL VOC threshold and the one this competition scores at. Because a fixed
pixel error means something different at every object scale — 3 px of slop is harmless on
a 48 px hole (IoU 0.88) and fatal on a 10 px one (IoU 0.54) — the threshold materially
changes what counts as correct, and **@50 is stricter than @45**: a box sitting at IoU
0.493 scores 1.000 at @45 and **0.000** at @50. Every default in this repo is 0.50;
`--iou 0.45` and `--iou 0.75` are diagnostics, and the scorer prints the threshold it used
with every result.

> **Full walkthrough, including the localisation-tolerance table and why mAP@50 is an
> evaluation metric rather than a training objective: [`docs/metric_guide.md`](docs/metric_guide.md).**

### The one distinction that matters most

**You do not train *with* mAP@50.** It is not differentiable — it contains a discrete
match/no-match decision and a sort by confidence, so there is no gradient to descend.
Training minimises a **loss** (box regression + classification + DFL). mAP@50 is then
*measured* on held-out images to decide:

1. **which checkpoint to submit** — and Ultralytics' own `best.pt` is chosen by mAP@50:95, *not* your metric (see §5),
2. **what confidence threshold to submit at** — chosen by a sweep, not guessed,
3. **whether a change actually helped** — the same yardstick for every experiment.

### The organising principle

**The dataset and the machine decide the training configuration, not the other way round.**
Every script measures before it acts, and refuses to continue when the inputs are not
trustworthy — no guessed labels, no silent format conversion, no hardcoded image size.
A detector trained on invented boxes scores worse than one trained on fewer correct
boxes, and you find that out after the deadline. So the pipeline stops and says so instead.

---

## 2. Getting the data (Kaggle)

The project data comes from Kaggle:

```
https://www.kaggle.com/t/95c0d96a415c52241ed6c14bb2a0b30a
```

That link is a **`/t/` task link**, which matters — see the note below.

### Route A — automated download (needs credentials)

```bash
pip install kagglehub                       # already in requirements.txt
python scripts/fetch_dataset.py --check     # reports credentials + reachability, downloads nothing
python scripts/fetch_dataset.py --url https://www.kaggle.com/t/95c0d96a415c52241ed6c14bb2a0b30a
```

Credentials are read from the standard locations and **never printed**:

- `~/.kaggle/kaggle.json` — from Kaggle → *your avatar* → *Settings* → *API* → *Create New Token*
  (Windows: `C:\Users\<you>\.kaggle\kaggle.json`; `chmod 600` it on Linux/macOS)
- or the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`

The script accepts any Kaggle reference form and resolves it itself:
`/t/<id>`, `/competitions/<slug>`, `/c/<slug>`, `/datasets/<owner>/<name>`, `/d/<owner>/<name>`,
a bare slug, or `owner/name`. It tries the API, falls back to the `kaggle` CLI, and — most
importantly — **tells the three failure modes apart**, because they have different fixes:

| what Kaggle says | what it actually means | the fix |
|---|---|---|
| 403 / permission denied | you have not accepted the competition rules | open the page, click **Join / I Accept**, retry |
| 401 / authentication | token missing or expired | create a fresh `kaggle.json` |
| 404 / not found | wrong slug, or a `/t/` id used as a slug | check the link form |
| connection / SSL error | this machine cannot reach Kaggle | use Route B — no token will fix it |

> **About the `/t/` form:** `/t/<hex>` links are Kaggle **tasks**, and they are frequently
> private — they require an invitation and may expose **no public download endpoint at all**.
> If the API keeps refusing while the browser shows you the data, that is expected, not a bug
> in your setup. Use Route B.

### Route B — manual download (always works, no token)

1. Open the task/competition page in a browser and download the data files.
2. Put the archive(s) — `.zip`, `.tar`, `.7z`, `.rar` — into `dataset/raw/`.
3. Unpack and register them:

```bash
python scripts/fetch_dataset.py --unpack-only
```

This performs no network calls whatsoever, unpacks nested archives, and reports how many
images and label files it found. **Every other script in the project runs fully offline**,
so Route B is a complete substitute for Route A.

### What happens next, automatically

Whatever route you used, the next command is the same — and it is the one that decides
everything downstream:

```bash
python scripts/audit_dataset.py
```

---

## 3. End-to-end workflow

```bash
# 0 — environment
python -m pip install -r requirements.txt        # headless boxes: see the opencv note in the file

# 1 — data in
python scripts/fetch_dataset.py --check                                    # verify access
python scripts/fetch_dataset.py --url https://www.kaggle.com/t/<id>        # or --unpack-only
#             └─ or drop images straight into dataset/raw/

# 2 — audit: what do we actually have, and what is the next command?
python scripts/audit_dataset.py

# 3 — normalize annotations + build a leakage-safe split
python scripts/prepare_dataset.py --dry-run       # see what it would do
python scripts/prepare_dataset.py                  # writes dataset/prepared + configs/data.prepared.yaml
python scripts/validate_labels.py                  # label gate + a contact sheet you must LOOK at

# 4 — train
python scripts/train.py --dry-run                  # the config chosen for THIS machine
python scripts/train.py                            # baseline run

# 5 — choose and score
python scripts/select_best.py --run ped_baseline    # pick the checkpoint by mAP@50, not mAP@50:95
python scripts/evaluate.py --name ped_baseline      # Ultralytics metrics + the real mAP@50 + sweep

# 6 — submit in the organizers' format, then validate it
python scripts/submit.py --weights runs/ped_baseline/weights/best_metric.pt \
                         --source dataset/raw/test/images \
                         --test-images dataset/raw/test/images \
                         --conf <swept value> --out runs/submission.csv
python scripts/validate_submission.py --csv runs/submission.csv \
                         --test-images dataset/raw/test/images
```

### What each step does, and why it exists

| # | script | what it decides | failure it prevents |
|---|---|---|---|
| 2 | `audit_dataset.py` | Is there data? What label format is really there? Are the boxes in the right coordinate space? What `imgsz`/model/batch does **this** machine support? | training on the wrong folder, or on pixel coordinates mistaken for normalized ones |
| 2 | `recon_image.py` | what a single real component looks like: target sizes, how many appearances each class has | annotating or aug‑menting for an appearance the model never sees |
| 3 | `prepare_dataset.py` | converts YOLO/COCO/VOC/CSV → canonical YOLO; splits train/val **by component** | a part appearing in both splits, which inflates validation mAP into a memorisation score |
| 3 | `validate_labels.py` | are the labels, as written to disk, internally consistent — and do they *look* right? | a swapped Spot/Hole mapping, which is invisible in a table of numbers |
| 4 | `train.py` | batch size from **free** VRAM, image size from measured target size, model from dataset scale | OOM crashes and a default `imgsz=640` that silently destroys small targets |
| 5 | `eval_map.py` | the competition metric itself (`--iou`, default 0.50), plus the confidence sweep | optimising mAP@50 because that is what the trainer prints |
| 5 | `select_best.py` | which checkpoint to actually ship | submitting a checkpoint the trainer picked by the wrong metric |
| 6 | `predict.py` / `submit.py` | tiled inference for tiny targets; the organizers' `ID,predictions` CSV in **original image pixels**, plus auxiliary exports | coordinates in letterboxed space, which loses the whole submission |
| 6 | `validate_submission.py` | whether a grader would accept the file at all: columns, IDs, coverage, key names, class names, ranges, `[]` rows | a zero-width box, an unquoted JSON field, or a missing image row — each of which scores zero and none of which is visible in the CSV |
| — | `selftest.py` | 106 known-answer checks on the metric, parsing, formats, split, coordinate round-trip, the annotation stack, the class mapping and the submission file | the silent failures that only show up as a disappointing score |

---

## 4. Why the details are the way they are

**The split is by component, not by image.** `part012_frame3` and `part012_frame7` are the
same physical part, captured seconds apart. A random per-image split puts one in train and
the other in validation, and validation mAP becomes a memorisation score that does not
transfer to the test set. Override the grouping heuristic with
`--group-regex '(part\d+)'` when the naming is not guessable.

**`data.prepared.yaml` uses absolute paths on purpose.** Ultralytics resolves a relative
`path:` in a dataset YAML against its global `datasets_dir` setting, *not* against the
YAML's own location — so a relative path that looks correct in the file can silently point
somewhere else. This project reads the YAML back with one shared resolver
(`_ped/dataset_cfg.py`) instead of hardcoding `dataset/prepared/...`, because Kaggle
archives rarely use the folder names you expect.

**Empty labels are kept; missing labels are not.** An empty `.txt` means "this component
genuinely has no spot or hole" — a useful negative example. A *missing* `.txt` usually
means an annotator did not finish, so coverage below ~98% aborts the build rather than
quietly training on half a dataset.

**Small targets need a decision, not a default.** The audit reports the median **and the
smallest quartile** of box heights and picks the cheapest `imgsz` that keeps the small
quartile above ~12 px (the stride-32 floor). If even 1280 cannot, it says so and points at
tiled inference rather than pretending a bigger number solves it.

**Colour augmentation is capped deliberately.** For this component, `Hole` appears as a
near-white through-hole, a near-black shadowed bore, and a saturated warm bore in the same
photograph. Aggressive hue/saturation augmentation can move one appearance toward another
and teach the model that the *appearance* is noise. Geometric augmentation is safe and
useful; colour augmentation is kept moderate.

---

## 5. The trap this project exists to avoid

Verified in `ultralytics/models/yolo/detect/val.py`:

```python
stats["fitness"] = stats["metrics/mAP50-95(B)"]   # always use box mAP50-95 for fitness
```

Ultralytics chooses `best.pt` — and decides when early stopping fires — using
**mAP@50:95**. That is not your competition metric, and not even mAP@50. On a small
dataset the two rankings disagree, and the disagreement costs score. In a 24-epoch test
run here, two checkpoints from the same run scored **0.1073** and **0.2198** mAP@45 (that
test predates the correct threshold; the *swing* is the point, and it exceeds 11 points) —
choosing the wrong file is expensive whatever the threshold.

`scripts/select_best.py` re-scores every saved checkpoint on mAP@50 and copies the winner.
It refuses to overstate what it knows: checkpoints within 0.005 are reported as a **tie**
rather than a winner, and a run where everything scores near zero reports **no usable
checkpoint** instead of naming a winner out of noise.

**Always read the per-class columns, not just the mean.** From this repo's end-to-end run
(recorded at IoU 0.45, before the competition threshold was confirmed as 0.50 — so treat
the absolute figure as an upper bound, not a score):

```
checkpoint    IoU      metric     Spot     Hole
best.pt       0.45     0.4271   0.8540   0.0000
```

The mean of 0.85 and 0.00 is 0.43 — "roughly half marks". The split says **"Spot works,
Hole is completely absent"**, which is a data and class-balance problem where extra epochs
on the majority class will do nothing. That distinction is the difference between a useful
next action and a wasted day.

---

## 6. What actually raises mAP@50, in order

1. **Label correctness and coverage.** Swapped class ids or a half-annotated component cost
   more than any hyperparameter. `validate_labels.py` exists to catch this; look at the sheet.
2. **A leakage-free validation split.** If validation scores look too good, the split leaked.
3. **The minority class.** Check the per-class table before and after every change; the
   mean hides the class that is actually gating the score.
4. **Input resolution and tiling** — only if targets are small relative to the frame.
5. **Model size and epoch count** — usually last, and only once 1–4 are settled.
6. **Test-time confidence threshold** — free score, from `eval_map.py --sweep`. Choose it on
   validation and pass it to `submit.py --conf`.

---

## 7. Hardware

- `pip install ultralytics` does **not** guarantee GPU support. Check:
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
  `False` on a GPU box means a CPU-only wheel or a CUDA/driver mismatch.
- **Headless machines** (Colab, Kaggle notebooks, cloud VMs): ultralytics pulls
  `opencv-python` (the GUI build), which dies with `ImportError: libGL.so.1: cannot open
  shared object file`. Install `opencv-python-headless` instead of installing libGL, and
  remove the GUI package if both are present — it shadows the headless one.
- Batch size is derived from **free** VRAM minus ~1.2 GiB for the CUDA context, then capped
  conservatively. Raise it only after one clean epoch.
- Without a GPU, `train.py` refuses to start a long run. Use `--smoke` to prove the plumbing
  or `--allow-cpu` to accept the slowness deliberately.
- If the pretrained-weight download is blocked, `train.py` falls back to building the
  architecture from YAML **and says so loudly** — from-scratch training is a fallback, never
  the plan. On a normal machine `yolo11n.pt` downloads automatically and is a large head start.

---

## 8. Status against the plan

| # | plan item | script | state |
|---|---|---|---|
| 0 | get the originals into `dataset/raw/` | `fetch_dataset.py` | done — Kaggle API + browser fallback, offline unpack |
| 1 | install dependencies | `requirements.txt` | done, incl. the headless-opencv trap |
| 2 | audit dataset + hardware | `audit_dataset.py` | done |
| 3 | confirm format and class mapping | `audit_dataset.py` §4–5, `recon_image.py` | done |
| 4 | leakage-safe train/val split | `prepare_dataset.py` | done — grouped by component, stratified, seeded |
| 5 | train the baseline | `train.py` | done — config derived from the audit, recorded per run |
| 6 | inspect predictions and per-class errors | `eval_map.py`, `evaluate.py` | done |
| 7 | evaluate | `eval_map.py`, `select_best.py` | done — own implementation, both threshold readings reported, checkpoint selection corrected |
| 8 | test the submission format | `submit.py`, `validate_submission.py` | done — the organizers' exact CSV, original-pixel coordinates, and a validator that refuses a malformed file |

Verified end to end on generated data in a GPU-less sandbox (2 CPU cores, 31 training
images, 40 epochs from scratch): `Spot 0.854 / Hole 0.000`, mean 0.4271 — measured at the
old IoU 0.45 default, so re-run with the current @50 default before quoting a number. The
per-class split is the finding that matters, and it is exactly what a single number hides.

---

## 9. Repository layout

```
configs/data.prepared.yaml        generated by prepare_dataset.py, absolute paths, what training reads
configs/data.yaml                 hand-written fallback
dataset/raw/                      originals — every script treats this as read-only
dataset/prepared/                 canonical YOLO output + manifest.json (seed, groups, per-split counts)
docs/metric_guide.md              what mAP@50 is, in detail
docs/recon_findings.md            what the first real component looks like, measured
scripts/_ped/                     discovery, label parsing, dataset config, hardware planning, metrics, split
scripts/*.py                      the pipeline, listed in §3
runs/                             training runs, plots, reports, submissions (gitignored)
```

`manifest.json` is the audit trail: seed, component groups per split, and box counts per
class and per split. `validate_labels.py` re-checks the built files against it, which is how
a converter bug gets caught before it becomes a training run.

**Nothing generated is committed.** `.gitignore` excludes `dataset/`, `runs/`, `*.pt`,
`*.npy` and the venv — all regenerable from the scripts and pointless in history.

---

## 10. Common problems

| symptom | cause | fix |
|---|---|---|
| `Dataset not found` from the trainer | relative `path:` in the yaml | re-run `prepare_dataset.py` (writes absolute paths) |
| trainer reports **0 images** | `--data` points at a dataset whose layout the yaml does not match | check the printed resolution notes; they name every path tried |
| Kaggle **403** | competition rules not accepted | open the page, click *Join / I Accept*, retry |
| Kaggle fails but the browser works | `/t/` task link, private or API-less | `fetch_dataset.py --unpack-only` |
| `libGL.so.1: cannot open shared object file` | GUI opencv on a headless box | `pip install opencv-python-headless`, remove `opencv-python` |
| `torch.cuda.is_available()` is `False` on a GPU box | CPU wheel or CUDA/driver mismatch | install the matching torch build from pytorch.org |
| validation score looks too good | component leaked across the split | check `manifest.json` groups; use `--group-regex` |
| mAP@50 near zero on both classes | from-scratch weights on too little data, or labels wrong | get the pretrained download working; run `validate_labels.py` |
