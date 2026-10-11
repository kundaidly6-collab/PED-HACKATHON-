#!/usr/bin/env python3
"""Fetch the competition data from Kaggle into dataset/raw/ and hand off to the audit.

Kaggle is deliberately awkward about this, so the script:
  * accepts a full Kaggle URL, a slug, or a competition id in any of Kaggle's forms
  * tries the REST API first (clear errors, no interactive prompts) and falls back to
    the `kaggle` CLI if the API route is unavailable
  * distinguishes "not authenticated", "rules not accepted", and "no such resource",
    because those three have completely different fixes
  * unpacks archives and then reports where the images/labels actually landed, rather
    than assuming a layout

Credentials are read from the standard locations and are never printed:
    ~/.kaggle/kaggle.json                     {"username": "...", "key": "..."}
    KAGGLE_USERNAME + KAGGLE_KEY              (environment variables)

    python scripts/fetch_dataset.py --check
    python scripts/fetch_dataset.py --url https://www.kaggle.com/t/95c0d96a415c52241ed6c14bb2a0b30a
    python scripts/fetch_dataset.py --dataset owner/name
    python scripts/fetch_dataset.py --list-files
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "dataset" / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOT_EXTS = {".txt", ".json", ".xml", ".csv"}

# The link supplied for this project. A /t/<hex> path is Kaggle's task/competition form.
DEFAULT_URL = "https://www.kaggle.com/t/95c0d96a415c52241ed6c14bb2a0b30a"
DEFAULT_ID = "95c0d96a415c52241ed6c14bb2a0b30a"


# ----------------------------------------------------------------- URL parsing

def parse_kaggle_reference(text: str) -> tuple[str, str]:
    """('competition'|'dataset'|'task', identifier) from a URL or slug.

    Handles the shapes Kaggle actually uses:
      https://www.kaggle.com/competitions/<slug>
      https://www.kaggle.com/c/<slug>
      https://www.kaggle.com/datasets/<owner>/<name>
      https://www.kaggle.com/d/<owner>/<name>
      https://www.kaggle.com/t/<hex-id>          (task / private link)
      <slug>  |  <owner>/<name>
    """
    text = text.strip().strip('"').strip("'")
    url = text.split("?")[0].rstrip("/")

    m = re.search(r"kaggle\.com/(?:competitions|c)/([^/\s]+)", url)
    if m:
        return "competition", m.group(1)
    m = re.search(r"kaggle\.com/(?:datasets|d)/([^/\s]+)/([^/\s]+)", url)
    if m:
        return "dataset", f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"kaggle\.com/(?:t|tasks)/([^/\s]+)", url)
    if m:
        return "task", m.group(1)
    if "kaggle.com" in url:
        tail = url.split("kaggle.com", 1)[1].strip("/")
        if tail:
            parts = tail.split("/")
            if len(parts) >= 2 and parts[0] in {"datasets", "d"}:
                return "dataset", f"{parts[1]}/{parts[2]}" if len(parts) > 2 else parts[1]
            return "competition", parts[-1]
    if "/" in text:
        return "dataset", text
    return "competition", text


# ----------------------------------------------------------------- credentials

def credentials() -> tuple[str | None, str | None, str]:
    """(username, key, source description). Never returns or logs secret values."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key, "environment (KAGGLE_USERNAME / KAGGLE_KEY)"
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, None, f"~/.kaggle/kaggle.json is unreadable ({exc})"
        u, k = data.get("username"), data.get("key")
        if u and k:
            return u, k, "~/.kaggle/kaggle.json"
        return None, None, "~/.kaggle/kaggle.json exists but lacks username/key"
    return None, None, "nowhere found"


def auth_help() -> str:
    return (
        "How to authenticate (2 minutes, once):\n"
        "  1. kaggle.com -> your avatar -> Settings -> API -> 'Create New Token'\n"
        "     (this downloads kaggle.json)\n"
        "  2. put it at ~/.kaggle/kaggle.json  and  chmod 600 ~/.kaggle/kaggle.json\n"
        "     on Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json\n"
        "  3. or set the environment variables KAGGLE_USERNAME and KAGGLE_KEY\n"
        "Then re-run this script. The token is personal — never commit it, and never paste\n"
        "it into a chat or a notebook cell you might share."
    )


# ----------------------------------------------------------------- API access

def try_kagglehub(kind: str, ident: str, dest: Path) -> tuple[bool, str]:
    """Preferred route: kagglehub handles caching and returns a real path."""
    try:
        import kagglehub
    except ImportError:
        return False, "kagglehub not installed (pip install kagglehub)"
    try:
        if kind == "dataset":
            path = kagglehub.dataset_download(ident)
        else:
            # kagglehub exposes competitions as '<owner>/<slug>' style handles; the raw
            # numeric id from a /t/ link is not one of them, so let it raise and fall back.
            path = kagglehub.competition_download(ident)  # type: ignore[attr-defined]
        return True, str(path)
    except AttributeError:
        return False, "this kagglehub build has no competition downloader"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:400]}"


def try_cli(kind: str, ident: str, dest: Path) -> tuple[bool, str]:
    """Fallback: the official CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = (["kaggle", "datasets", "download", "-d", ident, "-p", str(dest), "--force"]
           if kind == "dataset" else
           ["kaggle", "competitions", "download", "-c", ident, "-p", str(dest), "--force"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        return False, "kaggle CLI not on PATH (pip install kaggle)"
    except subprocess.TimeoutExpired:
        return False, "kaggle CLI timed out after 30 min"
    if proc.returncode == 0:
        return True, str(dest)
    out = (proc.stdout + proc.stderr).strip()
    return False, out[:400] if out else f"exit code {proc.returncode}"


def diagnose(message: str, kind: str, ident: str) -> str:
    """Turn a raw API error into the actual fix."""
    low = message.lower()
    if any(s in low for s in ("403", "forbidden", "permission", "unauthorized", "401")):
        lines = [
            "Kaggle says this account may not download it. Two usual causes:",
            "  (a) you have not ACCEPTED THE COMPETITION RULES on the website — open the",
            "      competition page, click 'Join Competition' / 'I Understand and Accept', then retry;",
            "  (b) your token is missing, expired, or belongs to another account.",
        ]
        if kind == "task":
            lines.append(
                f"  Note: '{ident}' came from a /t/ link. /t/ links are Kaggle TASKS and are often\n"
                "  private — they need an invitation, and there may be no public download endpoint at all."
            )
        # The browser route works whenever the API does not, for every resource kind, so
        # it belongs in every permission failure rather than only the /t/ case.
        lines.append(
            "  Either way, you can bypass the API entirely: download the files in a browser\n"
            "  and run  python scripts/fetch_dataset.py --unpack-only  — no token needed."
        )
        return "\n".join(lines)
    if "404" in low or "not found" in low:
        return (
            f"Kaggle has no {kind} matching '{ident}'. If the link was a /t/ task id, the id is not\n"
            "a competition slug and cannot be fetched by slug — use the browser download instead."
        )
    if any(s in low for s in ("authenticat", "credential", "401", "token", "api key")):
        return auth_help()
    if any(s in low for s in ("max retries", "connection", "timeout", "timed out", "ssl", "resolve", "unreachable", "name or service")):
        return (
            "This machine cannot reach kaggle.com. That is a network restriction, not a\n"
            "Kaggle or credentials problem — no token will fix it. Options:\n"
            "  · run this script on a machine with normal internet access, or\n"
            "  · download in a browser and use:  python scripts/fetch_dataset.py --unpack-only\n"
            "The rest of the pipeline never needs the network at all."
        )
    if "not on path" in low or "not installed" in low:
        return (
            "Neither download route is available on this machine. Install one of them:\n"
            "  pip install kagglehub        (preferred)\n"
            "  pip install kaggle           (provides the CLI)\n"
            "or skip the API entirely: download in a browser, then --unpack-only."
        )
    return "None of the attempted routes worked. The messages above are the raw output."


# ----------------------------------------------------------------- unpacking

def unpack(directory: Path, verbose: bool = True) -> list[Path]:
    """Extract archives in place, depth-first, and return the extracted roots."""
    done: list[Path] = []
    for _ in range(4):  # nested archives are common (zip inside zip)
        archives = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".zip", ".7z", ".tar", ".gz", ".tgz", ".rar"}]
        if not archives:
            break
        for arc in archives:
            if verbose:
                print(f"  unpacking {arc.name} ({arc.stat().st_size / 1e6:.1f} MB)")
            if arc.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(arc) as zf:
                        zf.extractall(arc.parent)
                    arc.unlink()
                    done.append(arc.parent)
                except Exception as exc:
                    print(f"    [warn] could not unzip {arc.name}: {exc}")
            else:
                try:
                    subprocess.run(["tar", "-xf", str(arc), "-C", str(arc.parent)], check=True, capture_output=True)
                    arc.unlink()
                    done.append(arc.parent)
                except Exception as exc:
                    print(f"    [warn] could not extract {arc.name}: {exc} (7z/rar need extra tools)")
    return done


def describe(root: Path) -> dict:
    """What did we actually get? Counts by type, so the next step is obvious."""
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    others: dict[str, int] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() not in IMG_EXTS:
            others[p.suffix.lower() or "(none)"] = others.get(p.suffix.lower() or "(none)", 0) + 1
    return {"images": len(images), "by_ext": others, "sample": [str(p.relative_to(root)) for p in images[:5]]}


def link_into_raw(source: Path, raw: Path, verbose: bool = True) -> None:
    """Put the fetched tree where every other script expects to find it.

    Symlink rather than copy when possible: competition archives are often several GB
    and copying them doubles disk use for no benefit.
    """
    raw.mkdir(parents=True, exist_ok=True)
    if source.resolve() == raw.resolve():
        return
    entries = [p for p in source.iterdir()]
    for entry in entries:
        target = raw / entry.name
        if target.exists():
            if verbose:
                print(f"  (exists, left alone) {entry.name}")
            continue
        try:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
            if verbose:
                print(f"  linked {entry.name}")
        except (OSError, NotImplementedError):
            import shutil
            (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, target)
            if verbose:
                print(f"  copied {entry.name}")


# ----------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Download the Kaggle data into dataset/raw/.")
    ap.add_argument("--url", default=None, help=f"Kaggle URL or slug (default: {DEFAULT_URL})")
    ap.add_argument("--dataset", default=None, help="Kaggle dataset as owner/name")
    ap.add_argument("--competition", default=None, help="Kaggle competition slug")
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="destination (default: dataset/raw)")
    ap.add_argument("--check", action="store_true", help="report credentials and reachability, download nothing")
    ap.add_argument("--list-files", action="store_true", help="list what the remote holds, download nothing")
    ap.add_argument("--no-link", action="store_true", help="leave the download in its cache, do not populate dataset/raw")
    ap.add_argument(
        "--unpack-only",
        action="store_true",
        help="skip Kaggle entirely: unpack whatever is already sitting in dataset/raw "
        "(the path to use after a manual browser download, which needs no API or token)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser().resolve()
    if args.dataset:
        kind, ident = "dataset", args.dataset
    elif args.competition:
        kind, ident = "competition", args.competition
    else:
        kind, ident = parse_kaggle_reference(args.url or DEFAULT_URL)

    if args.unpack_only:
        # The offline path: a human downloaded the files. No token, no API, no network.
        raw.mkdir(parents=True, exist_ok=True)
        print("=== offline mode: unpacking what is already in dataset/raw ===")
        print(f"  directory: {raw}")
        before = describe(raw)
        print(f"  found now: {before['images']} images, {sum(before['by_ext'].values())} other files")
        print("\n  unpacking:")
        unpack(raw, verbose=True)
        info = describe(raw)
        print(f"\n=== after unpacking ===\n  images: {info['images']}")
        if info["by_ext"]:
            print("  other files: " + ", ".join(f"{k}×{v}" for k, v in sorted(info["by_ext"].items(), key=lambda t: -t[1])[:8]))
        for s0 in info["sample"]:
            print(f"    {s0}")
        if info["images"] == 0:
            print(
                "\n[WARN] still no images. Either the archive is in a format this cannot open\n"
                "       (7z/rar need p7zip / unrar installed), or the files are elsewhere.\n"
                f"       Put the image files somewhere under {raw} and re-run."
            )
            return 4
        print("\nNEXT:  python scripts/audit_dataset.py")
        return 0

    user, key, source = credentials()
    print("=== Kaggle access ===")
    print(f"  reference : {args.url or args.dataset or args.competition or DEFAULT_URL}")
    print(f"  resolved  : kind={kind} id={ident}")
    print(f"  token     : {'found in ' + source if key else 'MISSING — ' + source}")

    if args.check:
        print(f"  kagglehub : {'installed' if _has('kagglehub') else 'not installed'}")
        print(f"  kaggle CLI: {'on PATH' if _which('kaggle') else 'not on PATH'}")
        print(f"  destination: {raw}  ({'exists' if raw.exists() else 'will be created'})")
        if not key:
            print("\n" + auth_help())
            return 2
        print("\nCredentials present. Try the real download with:")
        print(f"  python scripts/fetch_dataset.py --url {args.url or DEFAULT_URL}")
        return 0

    if not key:
        print("\n[BLOCKED] no Kaggle credentials found.")
        print(auth_help())
        print("\nNothing was downloaded, so no partial state to clean up.")
        return 2

    if kind == "task":
        print(
            "\n[NOTE] this link is a Kaggle /t/ TASK id, not a competition slug or dataset handle.\n"
            "       Task links are frequently private and may have no public API download path.\n"
            "       Attempting anyway; if it fails, use the browser download (see --check output)."
        )

    # Try every route, in order of how clean their errors are, and keep ALL messages:
    # a single blank error tells the user nothing about what to fix.
    attempts: list[str] = []
    ok, result = False, ""
    routes = (
        [("kagglehub", "dataset"), ("cli", "dataset")]
        if kind == "dataset"
        else [("kagglehub", "competition"), ("cli", "competition")]
    )
    for route, kind_for_route in routes:
        if route == "kagglehub":
            done, msg = try_kagglehub(kind_for_route, ident, raw)
        else:
            done, msg = try_cli(kind_for_route, ident, raw)
        label = f"{route}({kind_for_route})"
        if done:
            ok, result = True, msg
            break
        attempts.append(f"{label}: {msg or 'no detail returned'}")

    # For a /t/ task id there is no dataset handle to try, but a competition attempt can
    # still succeed when the id doubles as one -- so make that attempt too.
    if not ok and kind == "task":
        done, msg = try_kagglehub("competition", ident, raw)
        if done:
            ok, result = True, msg
        else:
            attempts.append(f"kagglehub(competition, task id): {msg or 'no detail returned'}")

    if not ok:
        if args.verbose:
            print(f"  [kagglehub] {result}")
        ok2, result2 = try_cli(kind, ident, raw)
        if ok2:
            ok, result = ok2, result2
        elif kind == "competition":
            # last resort: kagglehub sometimes handles competitions even when the CLI fails
            ok3, result3 = try_kagglehub("competition", ident, raw)
            if ok3:
                ok, result = ok3, result3
            else:
                result = f"{result}\n{result2}\n{result3}"
        else:
            result = f"{result}\n{result2}"

    if not ok:
        joined = "\n".join(attempts)
        print("\n[FAILED] every download route was tried and none worked:")
        for a in attempts:
            print(f"  · {a}")
        print("\n" + diagnose(joined, kind, ident))
        print("\nFallback that always works, with no API involved:")
        print("  1. open the competition page in a browser and download the data files")
        print(f"  2. put the archive(s) in {raw}")
        print("  3. python scripts/fetch_dataset.py --unpack-only      (or just unzip them)")
        print("  4. python scripts/audit_dataset.py")
        return 3

    print(f"\n  downloaded to: {result}")
    source_path = Path(result)
    if not args.no_link and source_path.is_dir() and not (kind == "dataset" and source_path == raw):
        print("\n=== placing data where the pipeline expects it ===")
        link_into_raw(source_path, raw, verbose=True)

    target = raw if (raw.exists() and any(raw.rglob("*"))) else source_path
    print(f"\n=== unpacking ({target}) ===")
    unpack(target if target.is_dir() else target.parent, verbose=args.verbose)

    info = describe(raw if raw.exists() else target)
    print(f"\n=== what we got ===\n  images: {info['images']}")
    if info["by_ext"]:
        print("  other files: " + ", ".join(f"{k}×{v}" for k, v in sorted(info["by_ext"].items(), key=lambda t: -t[1])[:8]))
    for s in info["sample"]:
        print(f"    {s}")
    if info["images"] == 0:
        print(
            "\n[WARN] no images found after unpacking. The archive may hold only labels, or a\n"
            "       nested archive in a format needing 7z/unrar. Inspect the folder by hand."
        )
        return 4

    print(
        "\nNEXT:  python scripts/audit_dataset.py\n"
        "       The audit decides whether the labels came with it, what format they are in,\n"
        "       and prints the single next command for THIS data. If labels are absent, stop\n"
        "       there and annotate — do not train on guesses."
    )
    return 0


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def _which(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


if __name__ == "__main__":
    raise SystemExit(main())
