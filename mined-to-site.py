#!/usr/bin/env python3
"""Publish mine-transcript.py results as a web page plus a set of figures.

  ./mined-to-site.py talk.mined.jsonl --title "Talk title" \
      --part "0:00=Opening" --part "16:12=Second speaker" --out ~/src/quick-docs/my-talk

Writes
  <out>/index.html      one self-contained page in the quick-docs style: an argument map for
                        each key question (hover a box for the exact quote, click it to watch
                        that moment), with the items and quotes listed under each map
  <out>-figures/        one PNG and SVG per map, captioned with the source, for slides and posts

Live runs (a meeting mined from live captions, no video) work too. Times then come from the
clock time each item was mined, and the page goes next to the JSONL as a single .html file
that can be sent by email:

  ./mined-to-site.py ~/Documents/captions/seminar.mined.jsonl --title "Lab meeting" \
      --out ~/Documents/captions/lab-meeting.html

Needs Graphviz (brew install graphviz). Fix extraction mistakes in the JSONL, then rerun.
"""

import argparse
import html
import json
import re
import subprocess
import textwrap
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import settings

QUICK_DOCS = settings.path("SITE_DIR")  # quickdoc.css, theme.html, and one folder per site
TYPES = {  # fill, border, label
    "question": ("#fde6cf", "#d9720f", "Question"),
    "claim": ("#dde8f4", "#4e79a7", "Claim"),
    "evidence": ("#e1f0dd", "#4f9446", "Evidence"),
}
RELATIONS = {  # color, style
    "supports": ("#4f9446", "solid"),
    "challenges": ("#c0392b", "solid"),
    "answers": ("#8a8a8a", "dashed"),
}


def seconds(stamp):
    if not stamp:
        return None
    total = 0
    for part in stamp.split(":"):
        total = total * 60 + int(part)
    return total


def clock(sec):
    h, rest = divmod(sec, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


def slug(text, limit=60):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].rstrip("-")


def esc(text):
    return html.escape(text or "", quote=True)


# --- grouping items into maps ---------------------------------------------------

def build_units(items):
    """One unit per question (with its answers and their support trees), then one per
    remaining root claim. Each item belongs to the first unit that reaches it."""
    by_id = {i["id"]: i for i in items}
    kids, answers, has_parent = defaultdict(list), defaultdict(list), set()
    for i in items:
        target, rel = i.get("link_to"), i.get("relation")
        if target in by_id and target != i["id"]:
            if rel in ("supports", "challenges"):
                kids[target].append(i)
                has_parent.add(i["id"])
            elif rel == "answers":
                answers[target].append(i)
                has_parent.add(i["id"])

    owner, units = {}, []

    def grow(item, unit):
        if item["id"] in owner:
            return
        owner[item["id"]] = unit
        unit["members"].append(item)
        for child in kids[item["id"]]:
            grow(child, unit)

    def start(head):
        unit = {"head": head, "members": []}
        units.append(unit)
        grow(head, unit)
        for answer in answers[head["id"]]:
            grow(answer, unit)

    for q in items:
        if q["type"] == "question" and q["id"] not in owner:
            start(q)
    for i in items:
        if i["id"] not in owner and i["id"] not in has_parent:
            start(i)
    for i in items:  # anything left only links in a loop
        if i["id"] not in owner:
            start(i)

    for unit in units:
        times = [seconds(m.get("at")) for m in unit["members"] if m.get("at")]
        unit["time"] = seconds(unit["head"].get("at")) if unit["head"].get("at") else (min(times) if times else None)
        unit["members"].sort(key=lambda m: seconds(m.get("at")) if m.get("at") else 10**9)
        external = {m.get("link_to") for m in unit["members"]} - {m["id"] for m in unit["members"]}
        unit["ghosts"] = [by_id[t] for t in external if t in by_id]
        unit["is_map"] = len(unit["members"]) + len(unit["ghosts"]) >= 2
    return units, by_id


# --- figures --------------------------------------------------------------------

def dot_quote(text):
    return '"' + (text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def unit_dot(unit, video, prefix, caption=None):
    lines = [
        "digraph map {",
        '  graph [rankdir=BT, nodesep=0.28, ranksep=0.42, pad=0.25, bgcolor="white", tooltip=" ",'
        ' fontname="Helvetica", fontsize=10, fontcolor="#666666", labelloc=b, labeljust=l'
        + (f", label={dot_quote(caption)}" if caption else "") + "]",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.09", penwidth=1.4]',
        '  edge [arrowsize=0.75, penwidth=1.4, fontname="Helvetica", fontsize=8.5, fontcolor="#777777"]',
    ]

    def node(item, ghost=False):
        fill, border, label = TYPES[item["type"]]
        head = label.upper() + (f" · {item['at']}" if item.get("at") else "") + (" · in another map" if ghost else "")
        body = '<br align="left"/>'.join(esc(line) for line in textwrap.wrap(item["text"], 44))
        lab = f'<<font point-size="8.5" color="{border}"><b>{esc(head)}</b></font><br align="left"/>{body}<br align="left"/>>'
        style = "rounded,filled,dashed" if ghost or item.get("confidence") == "low" else "rounded,filled"
        attrs = [f"id={dot_quote(prefix + '-' + item['id'])}", f"label={lab}", f'style="{style}"',
                 f'fillcolor="{"#f3f3f1" if ghost else fill}"', f'color="{border}"']
        if ghost:
            attrs.append('fontcolor="#555555"')
        tip = f"“{item.get('quote', '')}”" + (f"  (ASR: {item['asr_note']})" if item.get("asr_note") else "")
        attrs.append(f"tooltip={dot_quote(tip)}")
        if video and item.get("at"):
            attrs += [f'URL="https://youtu.be/{video}?t={seconds(item["at"])}"', 'target="_blank"']
        lines.append(f"  {dot_quote(item['id'])} [{', '.join(attrs)}]")

    for item in unit["members"]:
        node(item)
    for item in unit["ghosts"]:
        node(item, ghost=True)
    shown = {i["id"] for i in unit["members"] + unit["ghosts"]}
    for item in unit["members"]:
        rel = item.get("relation")
        if item.get("link_to") in shown and rel in RELATIONS:
            color, style = RELATIONS[rel]
            lines.append(f"  {dot_quote(item['id'])} -> {dot_quote(item['link_to'])} "
                         f'[id={dot_quote(prefix + "-" + item["id"] + "-edge")}, color="{color}", style={style}, label=" {rel}"]')
    lines.append("}")
    return "\n".join(lines)


def render(dot, fmt, out=None, dpi=None):
    args = ["dot", f"-T{fmt}"] + ([f"-Gdpi={dpi}"] if dpi else []) + ([f"-o{out}"] if out else [])
    result = subprocess.run(args, input=dot.encode(), capture_output=True, check=True)
    return result.stdout.decode()


def inline_svg(svg):
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r"<title>.*?</title>\s*", "", svg, flags=re.S)  # keep only the quote tooltips
    width = re.search(r'width="([\d.]+)pt"', svg)
    return svg, (float(width.group(1)) * 4 / 3 if width else 0)


# --- page -----------------------------------------------------------------------

EXTRA_CSS = """
figure.argmap { margin: 1.1rem 0 0.6rem; padding: 0.7rem; background: #fff; border: 1px solid var(--rule);
  border-radius: 8px; overflow-x: auto; text-align: center; }
figure.argmap svg { max-width: 100%; height: auto; }
figure.argmap a:hover path, figure.argmap a:hover polygon { stroke-width: 2.8; }
@media (min-width: 960px) {
  figure.argmap.wide { width: min(76rem, calc(100vw - 4rem)); position: relative; left: 50%; transform: translateX(-50%); }
}
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; margin: 1.2rem 0 2.2rem; font: 0.95rem/1.7 var(--sans); color: var(--muted); }
.chip { display: inline-block; padding: 0 0.45rem; border: 1.5px solid; border-radius: 4px; color: #222;
  font: 600 0.72rem/1.65 var(--sans); letter-spacing: 0.05em; text-transform: uppercase; vertical-align: 0.1em; }
.chip.question { background: #fde6cf; border-color: #d9720f; }
.chip.claim { background: #dde8f4; border-color: #4e79a7; }
.chip.evidence { background: #e1f0dd; border-color: #4f9446; }
.arrow { display: inline-block; width: 1.7rem; border-top: 2px solid; vertical-align: middle; margin-right: 0.35rem; }
.arrow.supports { border-color: #4f9446; } .arrow.challenges { border-color: #c0392b; } .arrow.answers { border-color: #8a8a8a; border-top-style: dashed; }
h2 .when, h3 .when { margin-left: 0.5rem; font: 500 0.9rem var(--sans); }
details { margin: 0 0 1.6rem; font: 1rem/1.5 var(--sans); }
summary { cursor: pointer; color: var(--muted); }
ul.items { list-style: none; margin: 0.5rem 0 1.4rem; padding: 0; font: 1rem/1.55 var(--sans); }
ul.items li { padding: 0.6rem 0; border-bottom: 1px solid var(--rule); }
ul.items .quote { margin-top: 0.25rem; color: var(--muted); font-size: 0.92rem; }
ul.items .meta { color: var(--muted); font-size: 0.85rem; }
footer { margin-top: 4rem; font: 0.9rem/1.5 var(--sans); color: var(--muted); }
"""


LEGEND = ('<div class="legend"><span><span class="chip question">Question</span> <span class="chip claim">Claim</span> '
          '<span class="chip evidence">Evidence</span></span>'
          '<span><span class="arrow supports"></span>supports</span><span><span class="arrow challenges"></span>challenges</span>'
          '<span><span class="arrow answers"></span>answers</span><span>dashed box: low confidence or shown in another map</span></div>')


def time_link(video, at):
    if not at:
        return ""
    if not video:
        return f'<span class="when meta">{esc(at)}</span>'
    return f'<a class="when" href="https://youtu.be/{video}?t={seconds(at)}" target="_blank" rel="noopener">{esc(at)}</a>'


def item_li(item, by_id, video):
    chip = f'<span class="chip {item["type"]}">{TYPES[item["type"]][2]}</span>'
    rel = ""
    target = by_id.get(item.get("link_to") or "")
    if target and item.get("relation") in RELATIONS:
        short = target["text"] if len(target["text"]) <= 70 else target["text"][:67].rstrip() + "..."
        rel = f'<div class="meta">{item["relation"]}: “{esc(short)}”</div>'
    asr = f'<div class="meta">Caption repair: {esc(item["asr_note"])}</div>' if item.get("asr_note") else ""
    quote = f'<div class="quote">“{esc(item.get("quote"))}”</div>' if item.get("quote") else ""
    return f'<li id="{esc(item["id"])}">{chip} {esc(item["text"])} {time_link(video, item.get("at"))}{rel}{quote}{asr}</li>'


def main():
    ap = argparse.ArgumentParser(description="Build a web page and figures from mined claims, evidence and questions.")
    ap.add_argument("jsonl")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--part", action="append", default=[], metavar="TIME=TITLE",
                    help='a part of the event, e.g. "16:12=Sarah Cottingham: three commitments" (repeatable)')
    ap.add_argument("--video", help="YouTube video id (default: from a youtube-<id> file name)")
    ap.add_argument("--about", help="replace the 'How this was made' note (HTML allowed)")
    ap.add_argument("--purpose", help="replace the purpose paragraph under that note (HTML allowed)")
    ap.add_argument("--out", help=f"site folder, or a .html file (default: {QUICK_DOCS}/<title slug>; "
                                  "for live runs, a .html file next to the JSONL)")
    ap.add_argument("--figures", help="figures folder (default: <out>-figures)")
    a = ap.parse_args()

    src = Path(a.jsonl).expanduser()
    items = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    video = a.video or (re.search(r"youtube-([A-Za-z0-9_-]{11})", src.name) or [None, None])[1]
    stamps = sorted(i["time"] for i in items if i.get("time"))
    live = not video and not any(i.get("at") for i in items)
    if live:  # no recording timestamps: show the clock time each item was mined, as H:MM.
        for i in items:  # seconds() reads "10:51" as 651, which still sorts within one day
            if i.get("time"):
                i["at"] = f"{int(i['time'][11:13])}:{i['time'][14:16]}"
    out = Path(a.out).expanduser() if a.out else (src.with_suffix(".html") if live else QUICK_DOCS / slug(a.title))
    page_file = out if out.suffix == ".html" else out / "index.html"
    stem = out.with_suffix("") if out.suffix == ".html" else out
    figs = Path(a.figures).expanduser() if a.figures else stem.parent / (stem.name + "-figures")
    page_file.parent.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    parts = sorted(((seconds(p.split("=", 1)[0].strip()), p.split("=", 1)[1].strip()) for p in a.part), key=lambda p: p[0])
    parts = parts or [(0, "")]
    units, by_id = build_units(items)
    sections = defaultdict(list)
    for unit in units:
        t = unit["time"] if unit["time"] is not None else 0
        index = max(i for i, (start, _) in enumerate(parts) if start <= t or i == 0)
        sections[index].append(unit)

    if video:
        source = f"https://youtu.be/{video}"
        source_html = f'<a href="{esc(source)}">{esc(source)}</a>'
    elif live and stamps:
        day = datetime.fromisoformat(stamps[0])
        source = f"live captions, {day:%B %-d, %Y}, {stamps[0][11:16]}–{stamps[-1][11:16]}"
        source_html = esc(source[0].upper() + source[1:])
    else:
        source = src.name
        source_html = esc(source)
    models = sorted({i.get("model", "") for i in items if i.get("model")})
    counts = {t: sum(i["type"] == t for i in items) for t in TYPES}
    about = a.about or (
        f"The questions, claims and evidence below were extracted automatically by a language model "
        f"({', '.join(models) or 'LLM'}) from live captions of the meeting, then arranged around the questions "
        f"raised. Both the captions and the extraction contain errors, and speakers are not attributed. Times are "
        f"when each item was extracted, a minute or two after it was said. Hover over a box to read the exact "
        f"words from the captions." if live else
        f"The questions, claims and evidence below were extracted automatically from the event’s YouTube captions "
        f"by a language model ({', '.join(models) or 'LLM'}), then arranged around the questions raised. Both the "
        f"captions and the extraction contain errors, and speakers are not fully attributed. Hover over a box to read "
        f"the exact words; click it to watch that moment in the video.")
    purpose = a.purpose or (
        "It is one reading of the discussion, meant as a shared starting point for the next one. Views on what was "
        "said and how to interpret it can differ." if live else
        "The purpose of this map is to support public dialogue about the session. It is one "
        "reading of what was said, and views on how to interpret the session can differ.")

    body, toc, n_fig = [], [], 0
    for index, (start, title) in enumerate(parts):
        group = sorted(sections.get(index, []), key=lambda u: u["time"] if u["time"] is not None else 10**9)
        if not group:
            continue
        part_id = f"part-{index + 1}"
        if title:
            body.append(f'<h2 id="{part_id}">{esc(title)}{time_link(video, clock(start)) if video or live else ""}</h2>')
        toc_units = []
        for unit in (u for u in group if u["is_map"]):
            n_fig += 1
            head = unit["head"]
            uid = f"map-{n_fig}"
            name = f"{n_fig:02d}-{slug(head['text'])}"
            svg, width = inline_svg(render(unit_dot(unit, video, uid), "svg"))
            caption = f"{a.title}  ·  {source}  ·  extracted automatically from captions; may contain errors"
            captioned = unit_dot(unit, video, uid, caption=caption)
            render(captioned, "png", figs / f"{name}.png", dpi=200)
            render(captioned, "svg", figs / f"{name}.svg")
            label = ("Q: " if head["type"] == "question" else "") + head["text"]
            body.append(f'<h3 id="{uid}">{esc(label)}{time_link(video, head.get("at"))}</h3>')
            body.append(f'<figure class="argmap{" wide" if width > 880 else ""}">{svg}</figure>')
            lis = "".join(item_li(m, by_id, video) for m in unit["members"])
            body.append(f"<details><summary>Quotes and timestamps ({len(unit['members'])})</summary>"
                        f'<ul class="items">{lis}</ul></details>')
            toc_units.append(f'<li><a href="#{uid}">{esc(label if len(label) <= 95 else label[:92] + "...")}</a></li>')
        loose = [m for u in group if not u["is_map"] for m in u["members"]]
        if loose:
            body.append("<h4>Other points</h4>")
            body.append('<ul class="items">' + "".join(item_li(m, by_id, video) for m in loose) + "</ul>")
        if title:
            toc.append(f'<li><a href="#{part_id}">{esc(title)}</a>' + (f"<ul>{''.join(toc_units)}</ul>" if toc_units else "") + "</li>")
        else:
            toc.extend(toc_units)

    css = (QUICK_DOCS / "quickdoc.css").read_text(encoding="utf-8") if (QUICK_DOCS / "quickdoc.css").exists() else ""
    theme = (QUICK_DOCS / "theme.html").read_text(encoding="utf-8") if (QUICK_DOCS / "theme.html").exists() else ""
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(a.title)}</title>
<style>
{css}
{EXTRA_CSS}
</style>
{theme}
</head>
<body>
<header id="title-block-header">
<h1 class="title">{esc(a.title)}</h1>
{f'<p class="subtitle">{esc(a.subtitle)}</p>' if a.subtitle else ''}
<p class="date">Source: {source_html} · {counts['question']} questions, {counts['claim']} claims, {counts['evidence']} pieces of evidence</p>
</header>
<div class="note"><p><strong>How this was made</strong> {about}</p><p>{purpose}</p></div>
{LEGEND}
<nav id="TOC"><ul>{''.join(toc)}</ul></nav>
{chr(10).join(body)}
<footer>{date.today().strftime("%B %-d, %Y")}</footer>
</body>
</html>
"""
    page_file.write_text(page, encoding="utf-8")
    print(f"Page:    {page_file}  ({len(page) // 1024} KB, {n_fig} maps)")
    print(f"Figures: {figs}/  ({n_fig} PNG + SVG)")


if __name__ == "__main__":
    main()
