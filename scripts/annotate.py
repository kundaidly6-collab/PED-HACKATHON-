#!/usr/bin/env python3
"""A local annotation UI: review, correct and save boxes for the unlabelled images.

This competition ships TRAINING IMAGES WITH NO LABELS, so this script is the gate
between "a folder of photographs" and "a dataset". It is deliberately small and
dependency-free (Python stdlib only, no build step, no node, no cloud account):

  * loads draft proposals from propose_labels.py as a starting point
  * you confirm, move, reclass or delete each box
  * writes standard YOLO label files, one .txt per image

Keeping the data local matters: the images stay on this machine, labels cannot leak into
a service, and there is no upload step to get wrong.

    python scripts/annotate.py --images dataset/raw/train/images
    python scripts/annotate.py --images dataset/raw/train/images --drafts dataset/drafts --port 8000

Then open the printed URL. Controls are shown in the page.
Classes:  Spot = 0 (bolt, weld spot)      Hole = 1 (nut, opening)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# The class mapping lives in ONE place: _ped/labels.py, which is stdlib-only, so this
# server stays dependency-free. A second copy here would be a mapping that can silently
# drift from the one the labels and data.yaml are written with.
from _ped.labels import CLASS_NAMES  # noqa: E402

STATE: dict = {"images": [], "labels_dir": None, "drafts_dir": None, "mime": {}}


def load_yolo(path: Path) -> list[dict]:
    boxes = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            try:
                boxes.append({"c": int(float(p[0])), "x": float(p[1]), "y": float(p[2]),
                              "w": float(p[3]), "h": float(p[4])})
            except ValueError:
                continue
    return boxes


def resolve_image(images: list[dict], data: dict) -> dict:
    """Find the image a save request refers to.

    Accepts an integer index ``i`` or a filename ``key``. A missing or
    out-of-range index is an ERROR: it used to fall back to -1 and quietly
    overwrite the LAST image's labels, which is exactly the kind of silent data
    loss an annotator cannot afford.
    """
    if "i" in data and data["i"] is not None:
        try:
            k = int(data["i"])
        except (TypeError, ValueError):
            raise ValueError(f"bad index: {data['i']!r}")
        if not 0 <= k < len(images):
            raise ValueError(f"index {k} out of range (0..{len(images) - 1})")
        return images[k]
    key = str(data.get("key") or data.get("stem") or data.get("name") or "")
    if key:
        stem = Path(key).stem
        for rec in images:
            if Path(rec["path"]).stem == stem or rec["path"] == key:
                return rec
        raise ValueError(f"no image named {key!r}")
    raise ValueError("save request needs an image index (i) or a filename (key)")


def render_yolo(boxes: list[dict]) -> str:
    """YOLO detection lines for these boxes. Coordinates are clamped to [0, 1] so a
    box dragged past the edge of the image can never produce an invalid label."""
    lines = []
    for b in boxes:
        c = int(b["c"])
        if c not in (0, 1):
            raise ValueError(f"class id {c} is not part of this competition (only 0=Spot, 1=Hole)")
        lines.append(
            f"{c} {min(max(float(b['x']), 0.0), 1.0):.6f} {min(max(float(b['y']), 0.0), 1.0):.6f} "
            f"{min(float(b['w']), 1.0):.6f} {min(float(b['h']), 1.0):.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def save_yolo(path: Path, boxes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_yolo(boxes), encoding="utf-8")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>PED annotation</title>
<style>
  :root { --bg:#14161a; --panel:#1c1f26; --line:#2c313c; --fg:#e8eaed; --dim:#9aa3b2;
          --spot:#3b7cff; --hole:#00c8d7; --warn:#e0489a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; position:sticky; top:0; z-index:5; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .muted { color:var(--dim); }
  .spacer { flex:1; }
  button { background:#262b34; color:var(--fg); border:1px solid var(--line); border-radius:6px;
           padding:6px 11px; cursor:pointer; font-size:13px; }
  button:hover { background:#313846; }
  button.on { background:var(--spot); border-color:var(--spot); color:#fff; }
  button.hole.on { background:var(--hole); border-color:var(--hole); color:#06222a; }
  main { display:flex; gap:14px; padding:14px; align-items:flex-start; flex-wrap:wrap; }
  #stage { position:relative; background:#0b0d10; border:1px solid var(--line); border-radius:8px;
           overflow:hidden; flex:1 1 720px; min-width:320px; }
  #wrap { position:relative; display:inline-block; }
  canvas { display:block; max-width:100%; height:auto; cursor:crosshair; }
  #list { width:290px; background:var(--panel); border:1px solid var(--line); border-radius:8px;
          max-height:78vh; overflow:auto; }
  .row { padding:5px 10px; display:flex; justify-content:space-between; gap:8px; cursor:pointer;
         border-bottom:1px solid #22262e; font-size:12.5px; }
  .row:hover { background:#242833; }
  .row.active { background:#2d3442; }
  .row.done { color:#7fd18a; }
  .badge { font-size:11px; padding:1px 6px; border-radius:10px; background:#2b3140; color:var(--dim); }
  .badge.amb { background:var(--warn); color:#fff; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  #hint { margin:0 14px 20px; color:var(--dim); font-size:12.5px; }
  kbd { background:#22262e; border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:11.5px; }
</style></head>
<body>
<header>
  <h1>PED annotator</h1>
  <span class="muted" id="counter">…</span>
  <button id="bSpot" class="on">Spot <span class="muted">[1]</span> bolt / weld</button>
  <button id="bHole" class="hole">Hole <span class="muted">[2]</span> nut / opening</button>
  <span class="spacer"></span>
  <span class="muted" id="status"></span>
</header>
<main>
  <div id="stage"><div id="wrap"><canvas id="cv"></canvas></div></div>
  <div>
    <div id="list"></div>
  </div>
</main>
<div id="hint">
  <b>Drag</b> on empty space to draw a box · <b>click</b> a box to select · <b>drag its body</b> to move ·
  <b>drag a corner</b> to resize · <kbd>Del</kbd> remove selected · <kbd>1</kbd>/<kbd>2</kbd> set class (also reclass the selection) ·
  <kbd>A</kbd>/<kbd>←</kbd> prev · <kbd>D</kbd>/<kbd>→</kbd> next · <kbd>S</kbd> save · <kbd>E</kbd> mark reviewed &amp; next.
  <span class="badge amb">magenta</span> = a round feature inside the panel: the tool will not guess
  Nut/Hole vs weld/Spot, so that call is yours. Saving writes YOLO .txt next to nothing else — your originals are untouched.
</div>

<script>
const CLS = {0:{name:'Spot', color:'#3b7cff'}, 1:{name:'Hole', color:'#00c8d7'}};
const AMB = '#e0489a';
let state = { images: [], i: 0, cur: null, boxes: [], sel: -1, cls: 0, dirty: false };
let img = new Image(), scale = 1;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');

async function boot() {
  const r = await fetch('api/state'); const j = await r.json();
  state.images = j.images;
  document.getElementById('counter').textContent = j.images.length + ' images';
  render();
  buildList();
  load(0);
}

function buildList() {
  const el = document.getElementById('list'); el.innerHTML = '';
  state.images.forEach((im, k) => {
    const d = document.createElement('div');
    d.className = 'row' + (k === state.i ? ' active' : '') + (im.reviewed ? ' done' : '');
    const amb = im.ambiguous ? `<span class="badge amb">${im.ambiguous}</span>` : '';
    d.innerHTML = `<span>${im.name}</span><span>${amb}<span class="badge">${im.n}</span></span>`;
    d.onclick = () => load(k);
    el.appendChild(d);
  });
  const act = el.querySelector('.row.active');
  if (act) act.scrollIntoView({block:'nearest'});
}

async function load(k) {
  if (state.dirty) await save();
  state.i = k; state.sel = -1;
  const im = state.images[k];
  const r = await fetch('api/boxes/' + k); const j = await r.json();
  state.cur = j; state.boxes = j.boxes || [];
  img = new Image();
  img.onload = () => { scale = Math.min(1, 1100 / img.width); draw(); };
  img.src = 'img/' + k + '?t=' + Date.now();
  buildList();
  document.getElementById('status').textContent = im.reviewed ? 'reviewed' : 'not reviewed';
}

function draw() {
  if (!img.width) return;
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  cv.width = w; cv.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  state.boxes.forEach((b, i) => {
    const x = (b.x - b.w / 2) * w, y = (b.y - b.h / 2) * h, bw = b.w * w, bh = b.h * h;
    const amb = (b.note || '').includes('?');
    ctx.lineWidth = i === state.sel ? 3 : 2;
    ctx.strokeStyle = i === state.sel ? '#ffffff' : (amb ? AMB : CLS[b.c].color);
    ctx.strokeRect(x, y, bw, bh);
    ctx.font = '600 12px ui-sans-serif'; 
    const tag = CLS[b.c].name;
    const tw = ctx.measureText(tag).width + 8;
    ctx.fillStyle = amb ? AMB : CLS[b.c].color;
    ctx.fillRect(x, Math.max(0, y - 15), tw, 15);
    ctx.fillStyle = '#0b0d10'; ctx.fillText(tag, x + 4, Math.max(11, y - 4));
    if (i === state.sel) {
      ctx.fillStyle = '#fff';
      [[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]].forEach(([px,py]) => ctx.fillRect(px-3,py-3,6,6));
    }
    if (state.sel === i) { ctx.strokeStyle = '#666'; ctx.lineWidth = 1; ctx.strokeRect(x,y,bw,bh); }
  });
}

function toImg(ev) {
  const r = cv.getBoundingClientRect();
  const sx = cv.width / r.width, sy = cv.height / r.height;
  return { x: (ev.clientX - r.left) * sx, y: (ev.clientY - r.top) * sy };
}
function hit(p) {
  const w = cv.width, h = cv.height;
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const b = state.boxes[i];
    const x = (b.x - b.w/2)*w, y = (b.y - b.h/2)*h, bw = b.w*w, bh = b.h*h;
    const m = 6;
    const corners = [[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]];
    for (let c = 0; c < 4; c++)
      if (Math.abs(p.x-corners[c][0]) < m && Math.abs(p.y-corners[c][1]) < m)
        return { i, corner: c };
    if (p.x >= x && p.x <= x+bw && p.y >= y && p.y <= y+bh) return { i, corner: -1 };
  }
  return null;
}
let drag = null;
cv.addEventListener('mousedown', ev => {
  const p = toImg(ev), w = cv.width, h = cv.height;
  const h0 = hit(p);
  if (h0 && h0.corner >= 0) {
    const b = state.boxes[h0.i];
    drag = { mode:'resize', i:h0.i, corner:h0.corner, b:{...b} };
    state.sel = h0.i;
  } else if (h0) {
    state.sel = h0.i;
    drag = { mode:'move', i:h0.i, off:{ dx:p.x/w - state.boxes[h0.i].x, dy:p.y/h - state.boxes[h0.i].y } };
  } else {
    drag = { mode:'new', x0:p.x/w, y0:p.y/h };
  }
  draw();
});
cv.addEventListener('mousemove', ev => {
  const p = toImg(ev), w = cv.width, h = cv.height;
  if (!drag) {
    const h0 = hit(p);
    cv.style.cursor = h0 ? (h0.corner >= 0 ? 'nwse-resize' : 'move') : 'crosshair';
    return;
  }
  if (drag.mode === 'move') {
    const b = state.boxes[drag.i];
    b.x = Math.min(1 - b.w/2, Math.max(b.w/2, p.x/w - drag.off.dx));
    b.y = Math.min(1 - b.h/2, Math.max(b.h/2, p.y/h - drag.off.dy));
  } else if (drag.mode === 'resize') {
    const b = state.boxes[drag.i];
    const cx = b.x, cy = b.y;
    const corner = drag.corner;
    const ax = (corner === 0 || corner === 2) ? 1 : -1;
    const ay = (corner < 2) ? 1 : -1;
    let nw = Math.abs(p.x/w - cx) * 2, nh = Math.abs(p.y/h - cy) * 2;
    b.w = Math.max(0.004, Math.min(1, ax === 1 ? nw : nw));
    b.h = Math.max(0.004, Math.min(1, nh));
  } else {
    const x0 = Math.min(drag.x0, p.x/w), y0 = Math.min(drag.y0, p.y/h);
    const x1 = Math.max(drag.x0, p.x/w), y1 = Math.max(drag.y0, p.y/h);
    state.boxes.push({ c: state.cls, x:(x0+x1)/2, y:(y0+y1)/2, w:x1-x0, h:y1-y0, note:'' });
    state.sel = state.boxes.length - 1;
    drag = null;
  }
  state.dirty = true; draw();
});
window.addEventListener('mouseup', () => { drag = null; });
cv.addEventListener('mouseleave', () => { drag = null; });

document.addEventListener('keydown', async e => {
  if (e.key === '1') { setCls(0); }
  else if (e.key === '2') { setCls(1); }
  else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (state.sel >= 0) { state.boxes.splice(state.sel, 1); state.sel = -1; state.dirty = true; draw(); }
    e.preventDefault();
  }
  else if (e.key === 'a' || e.key === 'ArrowLeft') { load(Math.max(0, state.i - 1)); e.preventDefault(); }
  else if (e.key === 'd' || e.key === 'ArrowRight') { load(Math.min(state.images.length - 1, state.i + 1)); /* next skips marking */ e.preventDefault(); }
  else if (e.key === 's') { await save(); }
  else if (e.key === 'e') { await save(true); }
});

function setCls(c) {
  state.cls = c;
  document.getElementById('bSpot').classList.toggle('on', c === 0);
  document.getElementById('bHole').classList.toggle('on', c === 1);
  if (state.sel >= 0) { state.boxes[state.sel].c = c; state.dirty = true; draw(); }
}
document.getElementById('bSpot').onclick = () => setCls(0);
document.getElementById('bHole').onclick = () => setCls(1);

async function save(reviewed) {
  const el = document.getElementById('status');
  const body = { i: state.i, boxes: state.boxes, reviewed: !!reviewed };
  let j;
  try {
    const r = await fetch('api/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    j = await r.json();
  } catch (err) {
    // Network/proxy failure: keep the work dirty so the beforeunload beacon retries.
    el.textContent = 'SAVE FAILED (' + err + ') - your boxes are still in this tab, press S again';
    el.style.color = 'var(--warn)';
    return;
  }
  if (!j || !j.ok) {
    el.textContent = 'SAVE REJECTED: ' + ((j && j.error) || 'unknown error') + ' - nothing was written';
    el.style.color = 'var(--warn)';
    return;
  }
  state.dirty = false;
  el.style.color = '';
  el.textContent = (reviewed ? 'reviewed · ' : '') + 'saved ' + j.path;
  if (reviewed) {
    state.images[state.i].reviewed = true;
    buildList();
    if (state.i < state.images.length - 1) load(state.i + 1);
  } else {
    state.images[state.i].n = state.boxes.length;
    state.images[state.i].ambiguous = state.boxes.filter(b => (b.note||'').includes('?')).length;
    buildList();
  }
}
window.addEventListener('beforeunload', () => { if (state.dirty) navigator.sendBeacon('api/save', new Blob([JSON.stringify({i:state.i, boxes:state.boxes, reviewed:false})], {type:'application/json'})); });
boot();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quieter server
        if STATE.get("verbose"):
            sys.stderr.write("  " + (fmt % a) + "\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _get_response(self) -> tuple[int, bytes, str]:
        """Resolve a GET into (status, body, content-type).

        Split out from do_GET so HEAD can answer with the exact same headers and no
        body. A 501 on HEAD is what this looks like from the outside: a proxy or
        uptime check probes the URL, is told the method is unsupported, and reports a
        perfectly healthy server as dead.
        """
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return 200, PAGE.encode(), "text/html; charset=utf-8"
        if u.path == "/api/state":
            return 200, json.dumps({"images": STATE["images"], "classes": CLASS_NAMES}).encode(), "application/json"
        if u.path.startswith("/img/"):
            try:
                k = int(u.path.rsplit("/", 1)[1])
                p = Path(STATE["images"][k]["path"])
            except (ValueError, IndexError):
                return 404, b"no such image", "text/plain"
            if not p.is_file():
                return 404, b"missing", "text/plain"
            return 200, p.read_bytes(), STATE["mime"].get(p.suffix.lower(), "image/jpeg")
        if u.path.startswith("/api/boxes/"):
            try:
                k = int(u.path.rsplit("/", 1)[1])
                p = Path(STATE["images"][k]["path"])
            except (ValueError, IndexError):
                return 404, json.dumps({"boxes": []}).encode(), "application/json"
            draft = load_yolo(Path(STATE["drafts_dir"]) / f"{p.stem}.txt") if STATE["drafts_dir"] else []
            final = load_yolo(Path(STATE["labels_dir"]) / f"{p.stem}.txt")
            notes = STATE["notes"].get(p.stem, {})
            for idx, b in enumerate(draft):
                b["note"] = notes.get(idx, "")
            body = json.dumps({"boxes": final if final else draft,
                               "source": "saved" if final else ("draft" if draft else "empty")})
            return 200, body.encode(), "application/json"
        return 404, b"not found", "text/plain"

    def do_GET(self) -> None:  # noqa: N802
        code, body, ctype = self._get_response()
        self._send(code, body, ctype)

    def do_HEAD(self) -> None:  # noqa: N802
        code, body, ctype = self._get_response()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path != "/api/save":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            rec = resolve_image(STATE["images"], data)
            p = Path(rec["path"])
            out = Path(STATE["labels_dir"]) / f"{p.stem}.txt"
            boxes = data.get("boxes", [])
            # Never destroy work silently: if we are about to change an existing file,
            # keep the previous version as .bak. One slot is enough to undo a bad save
            # or a stray keyboard shortcut.
            new_text = render_yolo(boxes)
            if out.is_file() and out.read_text(encoding="utf-8") != new_text:
                out.with_suffix(".txt.bak").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            save_yolo(out, boxes)
            rec["n"] = len(data.get("boxes", []))
            rec["ambiguous"] = sum(1 for b in data.get("boxes", []) if "?" in str(b.get("note", "")))
            if data.get("reviewed"):
                rec["reviewed"] = True
            return self._json({"ok": True, "path": str(out), "n": rec["n"]})
        except Exception as exc:
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local annotation UI for the unlabelled training images.")
    ap.add_argument("--images", required=True, help="folder of training images to annotate")
    ap.add_argument("--labels", default=None, help="where to write labels (default: <images>/../labels/train)")
    ap.add_argument("--drafts", default=None, help="draft proposals from propose_labels.py to load as a start")
    ap.add_argument("--notes", default=None, help="notes json from propose_labels (--json) for the magenta flags")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    src = Path(args.images).expanduser().resolve()
    if not src.is_dir():
        print(f"[ERROR] not a directory: {src}", file=sys.stderr)
        return 2
    images = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[ERROR] no images under {src}", file=sys.stderr)
        return 2

    labels_dir = Path(args.labels).expanduser().resolve() if args.labels else (src.parent / "labels" / "train")
    drafts_dir = Path(args.drafts).expanduser().resolve() if args.drafts else None
    labels_dir.mkdir(parents=True, exist_ok=True)

    notes: dict[str, dict[int, str]] = {}
    notes_file = Path(args.notes) if args.notes else (drafts_dir / "notes.json" if drafts_dir else None)
    if notes_file and notes_file.is_file():
        try:
            for rec in json.loads(notes_file.read_text(encoding="utf-8")):
                notes[rec["stem"]] = {i: b.get("note", "") for i, b in enumerate(rec.get("boxes", []))}
            print(f"  notes       : {notes_file} ({len(notes)} images annotated with hints)")
        except Exception as exc:
            print(f"  [warn] could not read {notes_file}: {exc}")

    STATE["images"] = [
        {"path": str(p), "name": p.name, "n": len(load_yolo(labels_dir / f"{p.stem}.txt")),
         "reviewed": (labels_dir / f"{p.stem}.txt").is_file()}
        for p in images
    ]
    STATE["labels_dir"] = str(labels_dir)
    STATE["drafts_dir"] = str(drafts_dir) if drafts_dir else None
    STATE["notes"] = notes
    STATE["mime"] = {e: ("image/png" if e == ".png" else "image/webp" if e == ".webp" else "image/jpeg")
                     for e in IMG_EXTS}
    STATE["verbose"] = args.verbose

    done = sum(1 for i in STATE["images"] if i["reviewed"])
    print("=" * 74)
    print("PED annotator")
    print("=" * 74)
    print(f"  images      : {len(images)}  ({src})")
    print(f"  labels out  : {labels_dir}")
    print(f"  drafts      : {drafts_dir or '(none — you are drawing every box by hand)'}")
    print(f"  already have labels: {done}/{len(images)}")
    print(f"\n  ACTION REQUIRED: label every image before training.")
    print(f"  Nothing here is a training label until you save it.\n")
    print(f"  Open  http://localhost:{args.port}/   (or the sandbox preview URL)")
    print("  Classes:  Spot = 0 (bolt, weld spot)   Hole = 1 (nut, opening)")
    print("  Stop with Ctrl-C. Labels are written continuously as you save.\n")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
