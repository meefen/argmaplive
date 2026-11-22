#!/usr/bin/env python3
"""Turn mine-transcript.py output (JSONL) into an Argdown argument map.

  ./mined-to-argdown.py talk.mined.jsonl            # writes talk.mined.argdown
  ./mined-to-argdown.py talk.mined.jsonl --watch    # rewrite whenever the JSONL grows (live mining)

Open the .argdown file in VS Code with the Argdown extension ("Argdown: Show Map"),
or export it with the Argdown CLI: npx @argdown/cli map talk.mined.argdown out --format svg

Layout: each question becomes a group holding the claims that answer it; everything
else is grouped by 10-minute section. "supports" and "challenges" links become Argdown
support (+) and attack (-) relations. Node colors: claim blue, evidence green, question orange.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

SECTION_MINUTES = 10
FRONTMATTER = """===
title: {title}
color:
    tagColors:
        claim: "#4e79a7"
        evidence: "#59a14f"
        question: "#f28e2b"
===
"""


def seconds(stamp):
    if not stamp:
        return None
    total = 0
    for part in stamp.split(":"):
        total = total * 60 + int(part)
    return total


def clean(text):
    """Keep item text from being read as Argdown syntax (tags, references, YAML)."""
    text = re.sub(r"#(?=\d)", "no. ", text).replace("#", "")
    return text.translate(str.maketrans({"[": "(", "]": ")", "{": "(", "}": ")", "<": "‹", ">": "›", "\n": " "}))


def youtube_id(path):
    m = re.search(r"youtube-([A-Za-z0-9_-]{11})", Path(path).name)
    return m.group(1) if m else None


def convert(items, title, video):
    children = defaultdict(list)  # id -> [(relation, item)] for supports/challenges
    answers = defaultdict(list)   # question id -> [item]
    ids = {i["id"] for i in items}
    for item in items:
        target, rel = item.get("link_to"), item.get("relation")
        if target in ids and rel in ("supports", "challenges"):
            children[target].append((rel, item))
        elif target in ids and rel == "answers":
            answers[target].append(item)

    placed, out = set(), [FRONTMATTER.format(title=json.dumps(title))]

    def node(item):
        at = item.get("at")
        text = clean(item["text"]) + (f" ({at})" if at else "")
        tags = f" #{item['type']}" + (" #low-confidence" if item.get("confidence") == "low" else "")
        meta = {"at": at, "confidence": item.get("confidence"), "quote": clean(item.get("quote") or "")}
        if item.get("asr_note"):
            meta["asr"] = clean(item["asr_note"])
        if video and at:
            meta["url"] = f"https://youtu.be/{video}?t={seconds(at)}"
        yaml = ", ".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items() if v)
        return f"[{item['id']}]: {text}{tags} {{{yaml}}}"

    def tree(item, depth=0, rel=None):
        indent, sign = "    " * depth, {"supports": "+ ", "challenges": "- "}.get(rel, "")
        if item["id"] in placed:  # already on the map: just draw the relation
            out.append(f"{indent}{sign}[{item['id']}]")
            return
        placed.add(item["id"])
        out.append(f"{indent}{sign}{node(item)}")
        for child_rel, child in children[item["id"]]:
            tree(child, depth + 1, child_rel)

    # Questions first: one group per question, holding its answers and their support trees.
    for q in (i for i in items if i["type"] == "question"):
        if q["id"] in placed:
            continue
        out.append(f"\n# {clean(q['text'])}\n")
        tree(q)
        for answer in answers[q["id"]]:
            out.append("")
            tree(answer)

    # Everything else, by section of the recording (or by mining batch if untimed).
    sections = defaultdict(list)
    for item in items:
        if item["id"] in placed or (item.get("link_to") in ids and item.get("relation") != "answers"):
            continue  # placed already, or drawn under its parent
        sec = seconds(item.get("at"))
        key = (sec // (SECTION_MINUTES * 60)) if sec is not None else f"batch {item.get('batch', 0)}"
        sections[key].append(item)
    for key, group in sections.items():
        clock = lambda m: f"{m // 60}:{m % 60:02d}:00" if m >= 60 else f"{m}:00"
        label = f"{clock(key * SECTION_MINUTES)}-{clock((key + 1) * SECTION_MINUTES)}" if isinstance(key, int) else key
        out.append(f"\n# Other claims, {label}\n")
        for item in group:
            if item["id"] not in placed:
                tree(item)
                out.append("")
    leftover = [i for i in items if i["id"] not in placed]  # e.g. items that only link to each other
    if leftover:
        out.append("\n# Other links\n")
        for item in leftover:
            if item["id"] not in placed:
                tree(item)
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def write(src, dst, title):
    items = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    dst.write_text(convert(items, title, youtube_id(src)), encoding="utf-8")
    return len(items)


def main():
    ap = argparse.ArgumentParser(description="Convert mined JSONL to an Argdown argument map.")
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", help="output .argdown file (default: next to the JSONL)")
    ap.add_argument("--title", help="map title (default: file name)")
    ap.add_argument("--watch", action="store_true", help="keep rewriting the map as the JSONL grows")
    a = ap.parse_args()
    src = Path(a.jsonl).expanduser()
    dst = Path(a.out).expanduser() if a.out else src.with_suffix(".argdown")
    title = a.title or src.name.split(".")[0]
    last = None
    while True:
        stamp = src.stat().st_mtime if src.exists() else None
        if stamp != last and stamp is not None:
            print(f"{write(src, dst, title)} items -> {dst}", flush=True)
            last = stamp
        if not a.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
