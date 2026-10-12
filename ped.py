#!/usr/bin/env python3
"""ped.py — ONE file that runs the whole project, end to end.

Everything this competition needs, from downloading the images to a validated
submission, starts here:

    python ped.py steps            the whole pipeline in order, and who does what
    python ped.py doctor           can this machine and this data actually run it?
    python ped.py all              run every automated stage, in order, and stop on the first failure

    python ped.py <stage> [args]   run one stage — arguments pass straight through
    python ped.py <stage> --help   that stage's own options, unchanged

Stages (all of them, in the order the pipeline uses them):

    fetch     download the competition data        (needs Kaggle login or a browser)
    audit     what is really in dataset/raw: images, labels, sizes, GPU
    recon     study ONE real part before annotating anything
    propose   draft bounding boxes for the unlabelled training images
    annotate  review them in a local UI          <-- a HUMAN does this step
    prepare   convert to canonical YOLO + component-grouped train/val split
    labels    gate the labels + render a contact sheet you must LOOK at
    train     train the detector (config derived from this machine)
    select    choose the checkpoint by the competition metric, not the trainer's
    evaluate  Ultralytics numbers + the real metric at BOTH threshold readings
    metric    the metric alone, on saved predictions
    predict   tiled inference for small targets, dumped to files
    submit    write the organizers' ID,predictions CSV
    check     validate that CSV against every submission rule
    doctor    environment + data readiness report
    steps     the pipeline map
    selftest  the known-answer test suite

WHY THIS FILE IS A DISPATCHER AND NOT A 5,000-LINE SCRIPT
---------------------------------------------------------
The stages are ~5,000 lines of tested code under scripts/, and they exist as
separate files for a reason. Folding them into one file would create a SECOND copy
of rules like the class mapping, the metric's IoU threshold, or the split logic —
and two copies drift. That is precisely the failure this project spends its effort
catching: an earlier revision had the scorer defaulting to IoU 0.45 while every CLI
passed 0.50, so the command line looked right and any library caller was silently
wrong. So: one entry point, one implementation of each rule, and this file owns the
*order* and the *checks between stages* — which is what is actually missing when you
have sixteen scripts and one deadline.

Everything prints what it is about to do. Nothing here trains on guessed labels:
if the training images have no annotations yet, `all` stops and tells you to run
`propose` and then `annotate`, because those labels are the participant's job.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --------------------------------------------------------------------------------------
# The stage table: this is the single source of truth for "what can be run".
# group is used for ordering and for the gates in `all`; needs_data says whether the
# stage is meaningless without the competition images.
# --------------------------------------------------------------------------------------
STAGES: dict[str, dict] = {
    "fetch": {
        "script": "fetch_dataset.py", "group": "setup", "needs_data": False,
        "what": "download the competition data (Kaggle login, or the browser fallback)",
    },
    "make-data": {
        "script": "make_synthetic_data.py", "group": "setup", "needs_data": False,
        "what": "generate a FAKE dataset that looks structurally real, to test the plumbing",
    },
    "audit": {
        "script": "audit_dataset.py", "group": "setup", "needs_data": True,
        "what": "report what is really in the data: layout, label format, sizes, GPU, imgsz",
    },
    "recon": {
        "script": "recon_image.py", "group": "setup", "needs_data": True,
        "what": "measure ONE real component: target sizes and how many appearances per class",
    },
    "propose": {
        "script": "propose_labels.py", "group": "annotate", "needs_data": True,
        "what": "draft bounding boxes + a review sheet (proposals, never labels)",
    },
    "annotate": {
        "script": "annotate.py", "group": "annotate", "needs_data": True,
        "what": "open the review UI where a human confirms every box",
    },
    "prepare": {
        "script": "prepare_dataset.py", "group": "data", "needs_data": True,
        "what": "convert to canonical YOLO and split train/val BY COMPONENT",
    },
    "labels": {
        "script": "validate_labels.py", "group": "data", "needs_data": True,
        "what": "gate the label files and render a contact sheet you must look at",
    },
    "train": {
        "script": "train.py", "group": "train", "needs_data": True,
        "what": "train the detector with a config derived from this machine",
    },
    "select": {
        "script": "select_best.py", "group": "train", "needs_data": True,
        "what": "score every saved checkpoint and keep the best on the real metric",
    },
    "evaluate": {
        "script": "evaluate.py", "group": "train", "needs_data": True,
        "what": "Ultralytics metrics + the competition metric + the confidence sweep",
    },
    "metric": {
        "script": "eval_map.py", "group": "train", "needs_data": True,
        "what": "the competition metric on its own, on saved predictions",
    },
    "predict": {
        "script": "predict.py", "group": "submit", "needs_data": True,
        "what": "run inference (optionally tiled) and dump predictions to files",
    },
    "submit": {
        "script": "submit.py", "group": "submit", "needs_data": True,
        "what": "write the organizers' ID,predictions CSV, one row per test image",
    },
    "check": {
        "script": "validate_submission.py", "group": "submit", "needs_data": False,
        "what": "validate a submission against every rule, and refuse it if any fails",
    },
    "doctor": {
        "script": None, "group": "meta", "needs_data": False,
        "what": "environment and data readiness report with the exact next command",
    },
    "steps": {
        "script": None, "group": "meta", "needs_data": False,
        "what": "print this pipeline map",
    },
    "selftest": {
        "script": "selftest.py", "group": "meta", "needs_data": False,
        "what": "run the known-answer tests for the parts that fail quietly",
    },
}

# The automated chain `all` runs, in order. Human steps (propose/annotate) are
# deliberately absent: `all` stops and hands over rather than guessing a class.
ALL_PLAN = ["audit", "prepare", "labels", "train", "select", "evaluate", "submit", "check"]

# What the trainer and the metric need that the 30-second data-prep install does not.
TRAIN_DEPS = [("ultralytics", "ultralytics"), ("torch", "torch")]
BASE_DEPS = [("numpy", "numpy"), ("PIL", "pillow"), ("yaml", "pyyaml")]
OPT_DEPS = [("cv2", "opencv-python-headless"), ("kagglehub", "kagglehub")]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _banner(text: str, ch: str = "=") -> None:
    _say("\n" + ch * 74)
    _say(text)
    _say(ch * 74)


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(module: str, dist: str | None = None) -> str:
    try:
        from importlib.metadata import version

        return version(dist or module)
    except Exception:
        return "?"


def _run(script: str, argv: list[str]) -> int:
    """Run one stage as a child process: the same command you would type yourself.

    A child process rather than an import, so a stage behaves identically whether you
    call it through ped.py or directly, and a crash inside torch cannot take the runner
    with it.
    """
    path = SCRIPTS / script
    if not path.is_file():
        _say(f"[ERROR] missing {path.relative_to(ROOT)} — is this the repository root?")
        return 2
    return subprocess.call([sys.executable, str(path), *argv], cwd=str(ROOT))


def _count(directory: Path, exts: set[str]) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_weights() -> Path | None:
    """The best checkpoint on disk, preferring the one chosen by the real metric.

    `select` copies its winner to best_metric.pt because Ultralytics' own best.pt is
    chosen by mAP@50-95 — a different metric. So if both exist, best_metric.pt is the one
    that was picked for the competition, and it wins here too.
    """
    runs = ROOT / "runs"
    if not runs.is_dir():
        return None
    for pattern in ("*/weights/best_metric.pt", "*/weights/best.pt"):
        hits = sorted(runs.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None


def data_summary() -> dict:
    """What is actually on disk, discovered rather than assumed.

    This deliberately calls the repository's own _ped.discovery instead of looking for
    "dataset/raw/train/images" by hand. A first version of this function did exactly that
    and reported "0 training images" for a perfectly good dataset sitting in
    raw/images + raw/annotations — which then blocked `all` with a message telling the
    user to download data they already had. The competition ships train/images and
    test/images, but every tool here accepts any common layout, so the entry point must
    not be the one place that hardcodes the folder names.
    """
    raw = ROOT / "dataset" / "raw"
    summary = {
        "raw": raw, "train_images": 0, "test_images": 0,
        "labeled": 0, "boxes": 0, "labels": 0,
        "prepared": (ROOT / "configs" / "data.prepared.yaml").is_file(),
        "test_dir": None, "layout": "none",
    }

    if raw.is_dir():
        try:
            sys.path.insert(0, str(SCRIPTS))
            from _ped.discovery import build_layout, iter_images  # noqa: PLC0415

            layout = build_layout(raw)
            summary["train_images"] = len(layout.images)
            if layout.image_root:
                summary["layout"] = str(layout.image_root.relative_to(raw))
                summary["layout"] += (f" + {layout.label_root.relative_to(raw)}"
                                      if layout.label_root else " (no label folder yet)")

            # Annotations = boxes a human produced. A label file that exists but is empty
            # is not an annotation, and neither is a draft proposal, so count the boxes.
            for rec in layout.images:
                if rec.yolo_label and rec.yolo_label.is_file():
                    lines = [ln for ln in rec.yolo_label.read_text(
                        encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                    if lines:
                        summary["labeled"] += 1
                        summary["boxes"] += len(lines)
            if layout.coco_json or layout.voc_xml or layout.csv_files:
                summary["labels"] = len(layout.coco_json) + len(layout.voc_xml) + len(layout.csv_files)
                summary["labeled"] = summary["labeled"] or len(layout.images)

            test = raw / "test"
            if test.is_dir() and iter_images(test):
                summary["test_dir"] = test / "images" if (test / "images").is_dir() else test
                summary["test_images"] = len(iter_images(test))
        except Exception as exc:   # a report must never crash the runner
            summary["layout"] = f"unreadable ({type(exc).__name__}: {exc})"
    return summary


# --------------------------------------------------------------------------------------
# meta commands
# --------------------------------------------------------------------------------------
def cmd_steps(_argv: list[str]) -> int:
    """The pipeline map: every stage, in order, and who performs it."""
    s = data_summary()
    _banner("PED pipeline — the order that matters")
    rows = [
        ("1", "fetch", "download the data", "you, once (Kaggle login)"),
        ("2", "audit", "what is actually in dataset/raw", "automatic"),
        ("3", "recon", "measure one real part before annotating", "automatic"),
        ("4", "propose", "draft boxes for the unlabelled images", "automatic"),
        ("5", "annotate", "confirm every box and its class", "HUMAN — the blocking step"),
        ("6", "prepare", "canonical YOLO + component-grouped split", "automatic"),
        ("7", "labels", "gate the labels, then LOOK at the sheet", "automatic + your eyes"),
        ("8", "train", "train, configured from this machine", "automatic"),
        ("9", "select", "keep the best checkpoint on the real metric", "automatic"),
        ("10", "evaluate", "metric at both threshold readings + sweep", "automatic"),
        ("11", "submit", "the organizers' ID,predictions CSV", "automatic"),
        ("12", "check", "validate every submission rule", "automatic"),
    ]
    for n, stage, what, who in rows:
        _say(f"  {n:>2}. {stage:<9} {what:<46} {who}")

    _say("\n  `python ped.py all` runs the automatic ones: "
         + " → ".join(ALL_PLAN))
    _say("  It stops before train if the images have no annotations yet, because those")
    _say("  labels are the participant's job, not the tool's.")

    _banner("Where this project stands right now", "-")
    _say(f"  layout          : {s['layout']}")
    _say(f"  training images : {s['train_images']}")
    _say(f"  test images     : {s['test_images']}")
    _say(f"  annotations     : {s['labeled']} image(s) labelled, {s['boxes']} box(es)")
    _say(f"  prepared split  : {'yes' if s['prepared'] else 'no'}")
    if s["train_images"] == 0:
        nxt = "python ped.py fetch        (or drop the images into dataset/raw/ by hand)"
    elif s["labeled"] == 0:
        nxt = "python ped.py propose      then  python ped.py annotate"
    elif not s["prepared"]:
        nxt = "python ped.py prepare"
    elif s["test_images"] == 0:
        nxt = "python ped.py train        (no test images yet, so `submit` will wait)"
    else:
        nxt = "python ped.py all"
    _say(f"\n  NEXT: {nxt}")
    return 0


def cmd_doctor(_argv: list[str]) -> int:
    """Environment + data report. Checks only; it changes nothing."""
    _banner("PED doctor — can this machine and this data run the pipeline?")
    _say(f"  python          {sys.version.split()[0]}  ({sys.executable})")
    _say(f"  platform        {sys.platform}, {os.cpu_count()} CPU core(s)")
    _say(f"  project root    {ROOT}")

    _banner("Dependencies", "-")
    ok = True
    for mod, dist in BASE_DEPS:
        have = _have(mod)
        ok &= have
        _say(f"  [{'ok' if have else 'MISSING'}] {mod:<12} {_version(mod, dist) if have else 'pip install ' + dist}")
    for mod, dist in OPT_DEPS:
        have = _have(mod)
        _say(f"  [{'ok' if have else '--'}] {mod:<12} "
             f"{_version(mod, dist) if have else 'optional: pip install ' + dist}")
    train_missing = []
    for mod, dist in TRAIN_DEPS:
        have = _have(mod)
        train_missing += [] if have else [dist]
        _say(f"  [{'ok' if have else 'MISSING'}] {mod:<12} "
             f"{_version(mod, dist) if have else 'pip install ' + dist}")

    cuda = None
    if _have("torch"):
        try:
            import torch  # noqa: PLC0415  (slow import; only when present)

            cuda = torch.cuda.is_available()
            _say(f"  GPU             {'available: ' + torch.cuda.get_device_name(0) if cuda else 'NONE — training will be CPU-slow'}")
            if cuda:
                free, total = torch.cuda.mem_get_info(0)
                _say(f"  VRAM            {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
        except Exception as exc:
            _say(f"  [warn] torch present but unusable: {exc}")

    _banner("Data", "-")
    s = data_summary()
    _say(f"  dataset/raw                 {'found' if s['raw'].is_dir() else 'MISSING'}")
    _say(f"  training images             {s['train_images']}")
    _say(f"  test images                 {s['test_images']}")
    _say(f"  layout discovered           {s['layout']}")
    _say(f"  annotations                 {s['labeled']} image(s) labelled, {s['boxes']} box(es)")
    _say(f"  configs/data.prepared.yaml  {'present' if s['prepared'] else 'not built yet'}")

    # Blockers stop the pipeline outright; warnings only stop the stages that need them.
    # Splitting them keeps the exit code meaningful: a fresh clone with no data exits 1
    # ("not ready"), while a data-prep-only machine with no torch still exits 0 ("prep is
    # ready, training is not").
    blockers: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    if not ok:
        blockers.append(("pip install -r requirements.txt", "the data-prep dependencies"))
    if s["train_images"] == 0:
        blockers.append(("python ped.py fetch", "no training images — or place them in dataset/raw/"))
    if train_missing:
        warnings.append(("pip install " + " ".join(train_missing), "needed by train / select / evaluate"))
    if s["train_images"] and s["labeled"] == 0:
        warnings.append(("python ped.py propose   then   python ped.py annotate",
                         "images exist but carry no annotations yet — this is the human step"))

    _banner("Verdict", "-")
    for why, what in blockers:
        _say(f"  [BLOCKER] {why}")
        _say(f"            {what}")
    for why, what in warnings:
        _say(f"  [warn]    {why}")
        _say(f"            {what}")
    if blockers:
        _say(f"\n  {len(blockers)} blocker(s): the pipeline cannot run until they are fixed.")
        return 1
    if warnings:
        _say(f"\n  {len(warnings)} warning(s); nothing blocks the data stages.")
        _say("  Ready for: python ped.py all")
        return 0
    _say("  All checks passed — python ped.py all")
    return 0


# --------------------------------------------------------------------------------------
# `all`: the automated chain, with the gates that matter
# --------------------------------------------------------------------------------------
def cmd_all(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ped.py all", description=__doc__.split("\n")[0])
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit a (slow) CPU training run — forwarded to train.py")
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip, e.g. train,select")
    ap.add_argument("--only", default="", help="comma-separated stages to run, e.g. prepare,labels")
    ap.add_argument("--submission", default=str(ROOT / "runs" / "submission.csv"),
                    help="where the submission is written, and what `check` validates")
    ap.add_argument("--preds", default=None,
                    help="a directory of saved YOLO+conf predictions: skips training for `submit`")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate dataset/prepared even though it exists (overwrites it)")
    a = ap.parse_args(argv)

    plan = list(ALL_PLAN)
    if a.only:
        wanted = [x.strip() for x in a.only.split(",") if x.strip()]
        unknown = [w for w in wanted if w not in STAGES]
        if unknown:
            _say(f"[ERROR] unknown stage(s) in --only: {unknown}")
            return 2
        plan = [p for p in plan if p in wanted]
    if a.skip:
        drop = {x.strip() for x in a.skip.split(",")}
        plan = [p for p in plan if p not in drop]

    _banner("ped.py all — the automated chain")
    for i, stage in enumerate(plan, 1):
        _say(f"  {i}. {stage:<9} {STAGES[stage]['what']}")
    if a.dry_run:
        _say("\n[dry-run] nothing executed.")
        return 0

    # Gate 1: the data. Without images every later stage is a confusing failure, so say
    # it here rather than 40 seconds into an audit.
    s = data_summary()
    if s["train_images"] == 0:
        _banner("STOPPED: no training images", "-")
        _say("  Nothing under dataset/raw/ that looks like a training image.")
        _say("  Fix with:   python ped.py fetch          (needs a Kaggle login)")
        _say("         or:  place train/images/ under dataset/raw/ yourself")
        return 2

    # Gate 2: annotations. THE human gate. The competition ships unlabelled images, so
    # this is the expected stopping point on a fresh checkout — and it must stop, because
    # training here would mean training on the tool's own guesses.
    if s["labeled"] == 0 and ("prepare" in plan or "train" in plan):
        _banner("STOPPED: the training images have no annotations yet", "-")
        _say("  The competition provides images WITHOUT labels, so this step is yours:")
        _say("    1.  python ped.py propose     # draft boxes + a review sheet")
        _say("    2.  python ped.py annotate    # confirm each box and set its class")
        _say("    3.  python ped.py all         # then this runs to the submission")
        _say("\n  Nothing is trained on proposed boxes here: a wrong class in the training")
        _say("  set is indistinguishable from a hard example, and costs you the score twice.")
        return 2

    # Gate 3: dependencies, before spending time on the data stages.
    missing = [dist for mod, dist in TRAIN_DEPS if not _have(mod)]
    if missing and ("train" in plan or "select" in plan or "evaluate" in plan):
        _banner("STOPPED: training dependencies are missing", "-")
        _say("  Not installed: " + ", ".join(missing))
        _say("  Install with:  pip install -r requirements.txt")
        _say("  Then re-run:   python ped.py all")
        return 2

    results: list[tuple[str, int]] = []
    skipped: dict[str, str] = {}
    for i, stage in enumerate(plan, 1):
        _banner(f"[{i}/{len(plan)}] {stage} — {STAGES[stage]['what']}")
        extra: list[str] = []

        # dataset/prepared is derived data, and prepare_dataset.py refuses to write into a
        # non-empty directory (correctly: it will not silently destroy a folder). Without
        # this branch, `all` could never be run twice — the second run would die at stage
        # two, which is exactly when someone is re-running it after fixing their labels.
        if stage == "prepare" and s["prepared"] and not a.rebuild:
            _say(f"  {ROOT / 'dataset' / 'prepared'} already exists — skipping.")
            _say("  Your labels are newer than it? Rebuild with:  python ped.py all --rebuild")
            results.append((stage, 0))
            continue
        if stage == "prepare" and a.rebuild:
            extra.append("--overwrite")
        if stage == "train":
            if a.allow_cpu:
                extra.append("--allow-cpu")
            if a.epochs:
                extra += ["--epochs", str(a.epochs)]

        # The tail of the chain needs something to submit. Without weights (or saved
        # predictions) submit.py exits with an argparse error about --weights/--preds,
        # which reads like a bug in ped.py rather than "you have not trained yet".
        if stage == "submit":
            weights = find_weights()
            if a.preds:
                extra += ["--preds", str(a.preds)]
            elif weights:
                extra += ["--weights", str(weights)]
                _say(f"  weights: {weights.relative_to(ROOT)}")
            else:
                _say("  no trained weights on disk and no --preds — skipping submit.")
                _say("  Train first (python ped.py train), or pass --preds <dir>.")
                skipped[stage] = "nothing to submit yet"
                continue
        if stage == "check":
            target = Path(a.submission)
            if not target.is_file():
                _say(f"  {target} does not exist — skipping the check.")
                skipped[stage] = "no submission file to validate"
                continue
            extra += ["--csv", str(target)]
            if s["test_dir"]:
                extra += ["--test-images", str(s["test_dir"])]
        if stage == "submit":
            # Point submit at the test images so EVERY image gets a row (the rules
            # require it) and at a known output path so `check` can find it.
            if s["test_dir"]:
                extra += ["--source", str(s["test_dir"]), "--test-images", str(s["test_dir"])]
            extra += ["--out", str(Path(a.submission))]
        rc = _run(STAGES[stage]["script"], extra)
        results.append((stage, rc))
        if rc != 0:
            _banner(f"FAILED at '{stage}' (exit {rc}) — stopping", "-")
            _say("  Fix the error above, then re-run just that stage:")
            _say(f"    python ped.py {stage} --help")
            _say("  ...and continue with:")
            nxt = plan[i] if i < len(plan) else None
            _say(f"    python ped.py {nxt}" if nxt else "    python ped.py all")
            return rc

    _banner("Done — the chain completed")
    for stage, rc in results:
        _say(f"  [ok]   {stage}")
    for stage, why in skipped.items():
        _say(f"  [skip] {stage} — {why}")
    sub = Path(a.submission)
    if sub.is_file():
        _say(f"\n  submission: {sub}")
        _say(f"  validated by the `check` stage above — upload only if it printed")
        _say(f"  'All required checks passed.'")
    return 0


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    meta = {"steps": cmd_steps, "doctor": cmd_doctor, "all": cmd_all}

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        _say("\nEveryday commands:")
        _say("  python ped.py all              run the whole automated pipeline")
        _say("  python ped.py doctor           is this machine and data ready?")
        _say("  python ped.py steps            the pipeline, in order\n")
        _say("Stages: " + "  ".join(sorted(STAGES)))
        _say("Run `python ped.py <stage> --help` for any stage's own options.")
        return 0

    if argv[0] in ("-V", "--version", "commands", "list"):
        _say("meta:")
        for name in ("all", "doctor", "steps"):
            _say(f"  {name:<9} " + {
                "all": "run every automated stage, in order, stopping on the first failure",
                "doctor": "environment and data readiness report",
                "steps": "the pipeline map",
            }[name])
        _say("stages:")
        for name, info in STAGES.items():
            _say(f"  {name:<9} {info['what']}")
        return 0

    stage, rest = argv[0], argv[1:]

    # Meta commands first: `all`, `steps` and `doctor` are not pipeline stages and have
    # no entry in STAGES. Checking the table before this list made `all` un-runnable.
    if stage in meta:
        return meta[stage](rest)

    if stage not in STAGES:
        close = [k for k in (*STAGES, *meta) if k.startswith(stage[:3])]
        _say(f"[ERROR] unknown stage {stage!r}.")
        if close:
            _say(f"        did you mean: {', '.join(close)}?")
        _say("        run `python ped.py steps` for the full list.")
        return 2

    return _run(STAGES[stage]["script"], rest)


if __name__ == "__main__":
    raise SystemExit(main())
