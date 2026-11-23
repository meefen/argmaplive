#!/usr/bin/env python3
"""Live argument map: a window that redraws itself as mine-transcript.py adds items.

  ./live-map.py ~/Documents/captions/meeting.mined.jsonl

Same boxes and colors as mined-to-site.py: a small map for each question (or claim) that
has links, and single items as a list. The latest activity is always at the bottom: a map
moves down when it gets a new item, and new items glow for a minute. While you are scrolled
to the bottom the view follows along; scroll up to read without being moved.

Can start before the JSONL exists. Needs Graphviz. Uses only the Python standard library.
Keys in the window:  f  full screen
"""

import argparse
import html
import importlib.util
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mined_to_site", HERE / "mined-to-site.py")
site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site)

FRESH_SECONDS = 60


def read_items(path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:  # a line still being written
                pass
    if not any(i.get("at") for i in items):  # live run: show the clock time each item was mined
        for i in items:
            if i.get("time"):
                i["at"] = f"{int(i['time'][11:13])}:{i['time'][14:16]}"
    return items


_svg_cache = {}


def svg_for(dot):
    """Graphviz is the slow part, so only maps that changed are drawn again."""
    if dot not in _svg_cache:
        _svg_cache[dot] = site.inline_svg(site.render(dot, "svg"))
    return _svg_cache[dot]


def content(items, video):
    """Maps and single items, ordered so that the unit with the newest item comes last."""
    units, by_id = site.build_units(items)
    order = {item["id"]: n for n, item in enumerate(items)}  # file order = order mined
    out, run = [], []

    def flush():
        if run:
            out.append('<ul class="items">' + "".join(run) + "</ul>")
            run.clear()

    for unit in sorted(units, key=lambda u: max(order[m["id"]] for m in u["members"])):
        if not unit["is_map"]:
            run.extend(site.item_li(m, by_id, video) for m in unit["members"])
            continue
        flush()
        head, key = unit["head"], site.esc(unit["head"]["id"])
        svg, width = svg_for(site.unit_dot(unit, video, f"m-{unit['head']['id']}"))
        label = ("Q: " if head["type"] == "question" else "") + head["text"]
        lis = "".join(site.item_li(m, by_id, video) for m in unit["members"])
        out.append(f'<section class="unit" data-key="{key}"><h3>{site.esc(label)}{site.time_link(video, head.get("at"))}</h3>'
                   f'<figure class="argmap{" wide" if width > 880 else ""}">{svg}</figure>'
                   f'<details data-key="{key}"><summary>Quotes ({len(unit["members"])})</summary>'
                   f'<ul class="items">{lis}</ul></details></section>')
    flush()
    return "\n".join(out)


class State:
    def __init__(self):
        self.cond = threading.Condition()
        self.version, self.data = 0, json.dumps({"status": "Starting…"})

    def publish(self, **data):
        with self.cond:
            self.version += 1
            self.data = json.dumps(data)
            self.cond.notify_all()


def watch(path, state, video, interval):
    last, seen = None, None  # last = (inode, mtime, size) of the file as drawn, or "missing"
    while True:
        try:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            signature = None
        if signature is None:
            if last != "missing":  # not there yet, or deleted: clear the view and wait
                state.publish(status=f"Waiting for {path.name} (mine-transcript.py writes it after its first round)…",
                              html="", fresh=[])
                last, seen = "missing", set()  # whatever appears next counts as new
        elif signature != last:
            if isinstance(last, tuple) and (signature[0] != last[0] or signature[2] < last[2]):
                seen = set()  # replaced by a different file: start over
            last = signature
            try:
                items = read_items(path)
            except FileNotFoundError:  # deleted just now; the next check clears the view
                time.sleep(interval)
                continue
            try:
                ids = [i["id"] for i in items]
                fresh = [] if seen is None else [i for i in ids if i not in seen]
                seen = set(ids)
                counts = {t: sum(i["type"] == t for i in items) for t in site.TYPES}
                day = (items[0].get("time") or "")[:10] if items else ""
                old = f"Items from {time.strftime('%b %-d', time.strptime(day, '%Y-%m-%d'))}, not today · " \
                    if day and day != time.strftime("%Y-%m-%d") else ""
                status = (f"{old}{counts['question']} questions, {counts['claim']} claims, {counts['evidence']} evidence"
                          f" · updated {time.strftime('%H:%M:%S')}" + (f" · {len(fresh)} new" if fresh else ""))
                state.publish(status=status, html=content(items, video), fresh=fresh)
            except FileNotFoundError:
                state.publish(status="Graphviz not found. Install it with: brew install graphviz")
            except Exception as e:  # keep watching; the next change may fix it
                state.publish(status=f"Couldn't draw the map: {e}")
        time.sleep(interval)


def make_handler(state, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with state.cond:
                            state.cond.wait_for(lambda: state.version != seen, timeout=15)
                            changed, seen, data = state.version != seen, state.version, state.data
                        self.wfile.write((f"data: {data}\n\n" if changed else ": ping\n\n").encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def open_window(url):
    for app in ("Google Chrome", "Brave Browser", "Microsoft Edge"):
        if Path(f"/Applications/{app}.app").exists():
            subprocess.Popen(["open", "-na", app, "--args", f"--app={url}"])
            return
    subprocess.Popen(["open", url])


LIVE_CSS = """
#status { min-height: 1.5em; }
section.unit { margin-bottom: 1.6rem; }
g.node.fresh path, g.node.fresh polygon { stroke: #e8a317; stroke-width: 5; }
ul.items li.fresh { background: rgba(255, 196, 40, 0.22); }
#jump { position: fixed; right: 1.4rem; bottom: 1.4rem; padding: 0.5rem 1rem; border: 1px solid #e8a317;
  border-radius: 999px; background: #fff3cf; color: #222; font: 600 0.95rem var(--sans); cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15); }
"""

SCRIPT = """
const maps = document.getElementById('maps'), statusLine = document.getElementById('status');
const jump = document.getElementById('jump'), freshAt = {}, FRESH_MS = %d;
const root = document.documentElement;
let following = true;

const nearBottom = () => innerHeight + scrollY >= root.scrollHeight - 120;
addEventListener('scroll', () => { following = nearBottom(); if (following) jump.hidden = true; });
jump.onclick = () => scrollTo({ top: root.scrollHeight, behavior: 'smooth' });
addEventListener('keydown', e => {
  if (e.key === 'f' && !e.metaKey && !e.ctrlKey && !e.altKey)
    document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen();
});

function markFresh() {
  const now = Date.now();
  maps.querySelectorAll('.fresh').forEach(el => el.classList.remove('fresh'));
  for (const [id, t] of Object.entries(freshAt)) {
    if (now - t > FRESH_MS) { delete freshAt[id]; continue; }
    maps.querySelectorAll(`g.node[id$="-${CSS.escape(id)}"], li[id="${CSS.escape(id)}"]`)
      .forEach(el => el.classList.add('fresh'));
  }
}
setInterval(markFresh, 5000);

function topVisible() {  // the first unit on screen, to keep it in place while reading
  for (const el of maps.querySelectorAll('section.unit, ul.items > li')) {
    const top = el.getBoundingClientRect().top;
    if (top >= 0) return { key: el.dataset.key || el.id, top };
  }
  return null;
}

new EventSource('/events').onmessage = e => {
  const s = JSON.parse(e.data);
  statusLine.textContent = s.status;
  if (s.html === undefined) return;
  const open = new Set([...maps.querySelectorAll('details[open]')].map(d => d.dataset.key));
  const anchor = following ? null : topVisible();
  maps.innerHTML = s.html;
  maps.querySelectorAll('details').forEach(d => { if (open.has(d.dataset.key)) d.open = true; });
  const now = Date.now();
  s.fresh.forEach(id => { freshAt[id] = now; });
  markFresh();
  if (following) {
    scrollTo(0, root.scrollHeight);
  } else {
    const el = anchor && maps.querySelector(`[data-key="${CSS.escape(anchor.key)}"], li[id="${CSS.escape(anchor.key)}"]`);
    if (el) scrollBy(0, el.getBoundingClientRect().top - anchor.top);
    if (s.fresh.length) jump.hidden = false;
  }
};
""" % (FRESH_SECONDS * 1000)


def main():
    ap = argparse.ArgumentParser(description="Show mined questions, claims and evidence as a live argument map.")
    ap.add_argument("jsonl", help="the .mined.jsonl file mine-transcript.py writes")
    ap.add_argument("--title", help="window title (default: from the file name)")
    ap.add_argument("--port", type=int, default=8766, help="local web port (default 8766)")
    ap.add_argument("--interval", type=float, default=2, help="seconds between checks for new items (default 2)")
    ap.add_argument("--no-window", action="store_true", help="don't open a browser window")
    a = ap.parse_args()

    path = Path(a.jsonl).expanduser()
    video = (re.search(r"youtube-([A-Za-z0-9_-]{11})", path.name) or [None, None])[1]
    title = a.title or f"Live map: {path.name.split('.')[0]}"
    css = (site.QUICK_DOCS / "quickdoc.css").read_text(encoding="utf-8") if (site.QUICK_DOCS / "quickdoc.css").exists() else ""
    theme = (site.QUICK_DOCS / "theme.html").read_text(encoding="utf-8") if (site.QUICK_DOCS / "theme.html").exists() else ""
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{css}
{site.EXTRA_CSS}
{LIVE_CSS}
</style>
{theme}
</head>
<body>
<header id="title-block-header">
<h1 class="title">{html.escape(title)}</h1>
<p class="date" id="status">Connecting…</p>
</header>
{site.LEGEND}
<div id="maps"></div>
<button id="jump" hidden>New items below ↓</button>
<script>{SCRIPT}</script>
</body>
</html>
"""

    state = State()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state, page))
    except OSError:  # port busy: pick any free one
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, page))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=watch, args=(path, state, video, a.interval), daemon=True).start()
    print(f"Live map window: {url}\nWatching {path}\nPress Ctrl+C to stop.")
    if not a.no_window:
        open_window(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
