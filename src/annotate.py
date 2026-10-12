"""
Alat za označavanje sopstvenog test skupa — pravougaonik oko pčele i oko svake grinje.

Otvara se lokalno u browseru (samo na ovom računaru, 127.0.0.1). Za svaku sliku:
  - PČELA  — jedan pravougaonik oko cele pčele (za obe klase)
  - GRINJA — pravougaonik oko svake vidljive grinje (samo klasa "varoa")

Klasa slike se ne označava — određuje je folder (zdrava/ ili varoa/).

Oznake se čuvaju posle svake izmene u dva fajla:
  oznake.json — sve oznake (pčela + grinje), izvor za evaluaciju
  gt.csv      — grinje u ISTOM formatu kao VarroaDataset:
                <fajl> <oznaka 0/1> [x1 y1 x2 y2]*

Koordinate su u pikselima originalne slike, (x1, y1) gornji levi ugao.

    python -m src.annotate                  # pa otvori http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .paths import ROOT

OWN_DIR = ROOT / "data" / "sopstveni_skup"
CLASSES = ["zdrava", "varoa"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(root: Path = OWN_DIR) -> list[str]:
    return [f"{cls}/{p.name}" for cls in CLASSES
            for p in sorted((root / cls).glob("*")) if p.suffix.lower() in IMAGE_EXT]


def load_annotations(root: Path = OWN_DIR) -> dict:
    path = root / "oznake.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def save_annotations(data: dict, root: Path = OWN_DIR) -> None:
    (root / "oznake.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    lines = []
    for name in list_images(root):
        label = CLASSES.index(name.split("/")[0])
        mites = data.get(name, {}).get("grinje", [])
        lines.append(" ".join([name, str(label)] + [str(v) for box in mites for v in box]))
    (root / "gt.csv").write_text("\n".join(lines) + "\n")


class Handler(BaseHTTPRequestHandler):
    root: Path = OWN_DIR

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            state = {"images": list_images(self.root), "annotations": load_annotations(self.root)}
            self._send(json.dumps(state).encode(), "application/json")
        elif self.path.startswith("/img/"):
            name = unquote(self.path[len("/img/"):])
            if name not in list_images(self.root):  # samo slike iz skupa, ništa drugo sa diska
                return self._send(b"nema", "text/plain", 404)
            self._send((self.root / name).read_bytes(), "image/jpeg")
        else:
            self._send(b"nema", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(b"nema", "text/plain", 404)
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        valid = set(list_images(self.root))
        save_annotations({k: v for k, v in data.items() if k in valid}, self.root)
        self._send(b"ok", "text/plain")

    def log_message(self, *args):  # bez loga za svaki zahtev
        pass


PAGE = r"""<!doctype html>
<html lang="sr"><head><meta charset="utf-8"><title>Označavanje — sopstveni skup</title>
<style>
  body { margin: 0; font: 14px system-ui, sans-serif; background: #f4f1ea; color: #222; display: flex; height: 100vh; }
  #side { width: 210px; overflow-y: auto; border-right: 1px solid #ccc; background: #fff; }
  #side div { padding: 4px 10px; cursor: pointer; display: flex; justify-content: space-between; }
  #side div.cur { background: #ffe9a8; font-weight: 600; }
  #side .ok { color: #2a7a2a; } #side .no { color: #b33; }
  #main { flex: 1; display: flex; flex-direction: column; padding: 10px; gap: 8px; min-width: 0; }
  #bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { font: inherit; padding: 5px 10px; border: 1px solid #999; border-radius: 4px; background: #fff; cursor: pointer; }
  button.on { background: #222; color: #fff; }
  #wrap { flex: 1; min-height: 0; display: flex; align-items: flex-start; justify-content: center; }
  canvas { cursor: crosshair; box-shadow: 0 1px 4px #0004; }
  #help { color: #555; font-size: 13px; }
  #msg { font-weight: 600; }
</style></head><body>
<div id="side"></div>
<div id="main">
  <div id="bar">
    <b id="title"></b>
    <button id="mBee">P — pčela</button>
    <button id="mMite">G — grinja</button>
    <button id="undo">Z — obriši poslednju grinju</button>
    <button id="clear">obriši sve na slici</button>
    <button id="prev">← prethodna</button>
    <button id="next">sledeća →</button>
    <span id="msg"></span>
  </div>
  <div id="help">Razvuci mišem pravougaonik. <b style="color:#1f6fd1">Plavo</b> = pčela (jedna po slici, nova zamenjuje staru),
    <b style="color:#d12f1f">crveno</b> = grinja (koliko god ih ima). Desni klik na pravougaonik ga briše. Sve se snima automatski.</div>
  <div id="wrap"><canvas id="c"></canvas></div>
</div>
<script>
let images = [], ann = {}, idx = 0, mode = 'bee', img = new Image(), scale = 1, drag = null;
const c = document.getElementById('c'), ctx = c.getContext('2d');

function cur() { const n = images[idx]; ann[n] = ann[n] || {pcela: null, grinje: []}; return ann[n]; }
function isVaroa(n) { return n.startsWith('varoa/'); }
function done(n) { const a = ann[n]; return a && a.pcela && (!isVaroa(n) || a.grinje.length > 0); }

function side() {
  const s = document.getElementById('side'); s.innerHTML = '';
  images.forEach((n, i) => {
    const d = document.createElement('div'); d.className = i === idx ? 'cur' : '';
    const a = ann[n] || {grinje: []};
    d.innerHTML = `<span>${n.split('/')[1]}</span><span class="${done(n) ? 'ok' : 'no'}">${done(n) ? '✓' : '·'}${isVaroa(n) ? ' ' + a.grinje.length : ''}</span>`;
    d.onclick = () => go(i); s.appendChild(d);
  });
  const left = images.filter(n => !done(n)).length;
  document.getElementById('msg').textContent = left ? `preostalo: ${left}` : 'sve slike označene ✓';
}

function go(i) {
  idx = (i + images.length) % images.length;
  img.onload = () => {
    const wrap = document.getElementById('wrap');
    scale = Math.min(wrap.clientWidth / img.naturalWidth, wrap.clientHeight / img.naturalHeight);
    c.width = Math.round(img.naturalWidth * scale); c.height = Math.round(img.naturalHeight * scale);
    draw();
  };
  img.src = '/img/' + encodeURIComponent(images[idx]).replace('%2F', '/');
  document.getElementById('title').textContent = images[idx];
  setMode(isVaroa(images[idx]) && cur().pcela ? 'mite' : 'bee');
  side();
}

function rect(b, color, w) { ctx.strokeStyle = color; ctx.lineWidth = w;
  ctx.strokeRect(b[0] * scale, b[1] * scale, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale); }

function draw() {
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const a = cur();
  if (a.pcela) rect(a.pcela, '#1f6fd1', 3);
  a.grinje.forEach(b => rect(b, '#d12f1f', 2));
  if (drag) rect(drag.box, mode === 'bee' ? '#1f6fd1' : '#d12f1f', 1);
}

function pos(e) { const r = c.getBoundingClientRect();
  return [Math.max(0, Math.min(img.naturalWidth, (e.clientX - r.left) / scale)),
          Math.max(0, Math.min(img.naturalHeight, (e.clientY - r.top) / scale))]; }

c.onmousedown = e => { if (e.button !== 0) return; const p = pos(e); drag = {start: p, box: [...p, ...p]}; };
c.onmousemove = e => { if (!drag) return; const p = pos(e), s = drag.start;
  drag.box = [Math.min(s[0], p[0]), Math.min(s[1], p[1]), Math.max(s[0], p[0]), Math.max(s[1], p[1])]; draw(); };
c.onmouseup = () => {
  if (!drag) return;
  const b = drag.box.map(Math.round); drag = null;
  if (b[2] - b[0] < 3 || b[3] - b[1] < 3) return draw();  // slučajan klik
  const a = cur();
  if (mode === 'bee') { a.pcela = b; if (isVaroa(images[idx])) setMode('mite'); } else a.grinje.push(b);
  changed();
};
c.oncontextmenu = e => {
  e.preventDefault(); const [x, y] = pos(e), a = cur(), inside = b => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
  const i = a.grinje.findLastIndex(inside);
  if (i >= 0) a.grinje.splice(i, 1); else if (a.pcela && inside(a.pcela)) a.pcela = null; else return;
  changed();
};

function changed() { draw(); side();
  fetch('/api/save', {method: 'POST', body: JSON.stringify(ann)})
    .catch(() => document.getElementById('msg').textContent = 'GREŠKA: nije snimljeno — da li alat još radi u terminalu?'); }

function setMode(m) { mode = m;
  document.getElementById('mBee').className = m === 'bee' ? 'on' : '';
  document.getElementById('mMite').className = m === 'mite' ? 'on' : ''; }

document.getElementById('mBee').onclick = () => setMode('bee');
document.getElementById('mMite').onclick = () => setMode('mite');
document.getElementById('undo').onclick = () => { cur().grinje.pop(); changed(); };
document.getElementById('clear').onclick = () => { ann[images[idx]] = {pcela: null, grinje: []}; changed(); };
document.getElementById('prev').onclick = () => go(idx - 1);
document.getElementById('next').onclick = () => go(idx + 1);
document.onkeydown = e => {
  const k = e.key.toLowerCase();
  if (k === 'p') setMode('bee'); else if (k === 'g') setMode('mite');
  else if (k === 'z') { cur().grinje.pop(); changed(); }
  else if (e.key === 'ArrowLeft') go(idx - 1); else if (e.key === 'ArrowRight') go(idx + 1);
};
window.onresize = () => go(idx);

fetch('/api/state').then(r => r.json()).then(s => { images = s.images; ann = s.annotations; go(0); });
</script></body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Označavanje pčela i grinja na sopstvenom skupu")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    n = len(list_images())
    if not n:
        print(f"Nema slika u {OWN_DIR}/{{zdrava,varoa}}/")
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{n} slika. Otvori u browseru:  http://127.0.0.1:{args.port}   (Ctrl+C za kraj)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
