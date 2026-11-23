#!/usr/bin/env python3
"""Run a whole live session from one command: captions, mining and the live map.

  ./session.py --title "Lab meeting" --topic "Seminar on AI for teaching; names: Piaget, Skinner"
  ./session.py --title "Lab meeting" --topic "..." --gemini   # mine with Gemini (sends the text to Google)

Makes a folder for the session, e.g. ~/Documents/captions/lab-meeting-2026-11-22-1030/,
and keeps everything there, named after the folder:
  <name>.txt            the transcript (live-captions.py --save)
  <name>.mined.jsonl    questions, claims and evidence (mine-transcript.py; the title and
                        --topic go to it as its topic, to help it fix misheard words)
  <name>.html           the page to share, built when the session ends, with <name>-figures/
  <name>.argdown        the same items for Argdown
  session.json          title, topic, start and end times, which model mined it
  captions.log, live-map.log   what those two scripts printed

The captions and the live map open in their own windows; this terminal shows the mined
items. Press Ctrl+C once to end: the captions stop first, the miner mines the last bit of
talk, the windows close, and the page is built. Ctrl+C again skips that last mining round.

Settings come from .env: SESSIONS_DIR (where session folders go), MIC, the models, and so on.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import files
import settings

HERE = Path(__file__).resolve().parent
PY = sys.executable


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50].rstrip("-") or "session"


def make_folder(base, title):
    stem = f"{slug(title)}-{datetime.now():%Y-%m-%d-%H%M}"
    folder, n = base / stem, 1
    while folder.exists():
        n += 1
        folder = base / f"{stem}-{n}"
    folder.mkdir(parents=True)
    return folder


def child_setup():
    # Own process group, so Ctrl+C reaches only this script, which then stops the others
    # in order. Ctrl+C is turned back on in case this shell had it ignored.
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def start(cmd, log=None):
    out = open(log, "w", encoding="utf-8") if log else None
    return subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=subprocess.STDOUT if out else None,
                            preexec_fn=child_setup)


def stop(proc, wait=10):
    """Ask a script to finish as if Ctrl+C was pressed; force it after `wait` seconds."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def last_lines(path, n=8):
    try:
        return "".join(Path(path).read_text(errors="replace").splitlines(True)[-n:])
    except FileNotFoundError:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="Run captions, mining and the live map for one session, with all files in one folder.")
    ap.add_argument("--title", required=True,
                    help='session title, e.g. "Lab meeting"; names the folder, the live map and the page')
    ap.add_argument("--topic", default="",
                    help="what the session is about, with names and terms; helps the miner fix misheard words")
    ap.add_argument("--gemini", action="store_true", help="mine with Gemini instead of the local model (sends text to Google)")
    ap.add_argument("--mic", help="microphone by part of its name (default: MIC in .env)")
    ap.add_argument("--no-map", action="store_true", help="don't run the live map")
    ap.add_argument("--no-window", action="store_true", help="don't open the captions and map windows")
    ap.add_argument("--dir", help="parent folder for the session (default: SESSIONS_DIR in .env)")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep messages in order with the miner's
    signal.signal(signal.SIGINT, signal.default_int_handler)

    base = Path(a.dir).expanduser() if a.dir else settings.path("SESSIONS_DIR")
    folder = make_folder(base, a.title)
    name = folder.name
    transcript, mined = folder / f"{name}.txt", folder / f"{name}.mined.jsonl"
    topic = f"{a.title}. {a.topic}" if a.topic else a.title
    record = {"title": a.title, "topic": a.topic, "started": datetime.now().isoformat(timespec="seconds"),
              "miner": f"Gemini ({settings.get('GEMINI_MODEL')})" if a.gemini else f"local ({settings.get('OLLAMA_MODEL')})"}
    (folder / "session.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Session folder: {files.tilde(folder)}\n")

    window = ["--no-window"] if a.no_window else []
    captions = start([PY, HERE / "live-captions.py", "--save", transcript, *window]
                     + (["--mic", a.mic] if a.mic else []), folder / "captions.log")
    time.sleep(1.5)  # let it check the mic and create the transcript before the others start
    if captions.poll() is not None:
        sys.exit(f"The captions didn't start:\n{last_lines(folder / 'captions.log')}")
    live_map = None if a.no_map else start([PY, HERE / "live-map.py", mined, "--title", a.title, *window],
                                           folder / "live-map.log")
    miner = start([PY, HERE / "mine-transcript.py", transcript, "--topic", topic] + (["--gemini"] if a.gemini else []))

    def end_signal(signum, frame):  # terminal closed or `kill`: end the session the same way
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, end_signal)

    warned = set()
    try:
        while captions.poll() is None:
            for label, proc in (("miner", miner), ("live map", live_map)):
                if proc is not None and proc.poll() is not None and label not in warned:
                    warned.add(label)
                    print(f"\n! The {label} stopped (exit {proc.returncode}). The captions keep running"
                          + (f"; see {files.tilde(folder / 'live-map.log')}." if label == "live map" else "."))
            time.sleep(0.5)
        print(f"\n! The captions stopped on their own:\n{last_lines(folder / 'captions.log')}")
    except KeyboardInterrupt:
        pass

    try:
        print("\nEnding the session: stopping the captions...")
        stop(captions)
        if miner.poll() is None:  # it mines what's left and says "Ctrl+C again to skip"
            miner.send_signal(signal.SIGINT)
            try:
                miner.wait(timeout=300)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                stop(miner)  # a second Ctrl+C tells the miner to skip the last round
    except KeyboardInterrupt:
        stop(captions, wait=3)
        stop(miner, wait=3)
    finally:
        stop(live_map, wait=3)
        for proc in (captions, miner):
            if proc.poll() is None:
                proc.kill()

    record["ended"] = datetime.now().isoformat(timespec="seconds")
    items = []
    if mined.exists():
        items = [json.loads(line) for line in mined.read_text(encoding="utf-8").splitlines() if line.strip()]
    record["items"] = {t: sum(i.get("type") == t for i in items) for t in ("question", "claim", "evidence")}
    (folder / "session.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    words = len(transcript.read_text(encoding="utf-8", errors="replace").split()) if transcript.exists() else 0
    print(f"\nSession folder: {files.tilde(folder)}")
    print(f"  transcript:  {transcript.name} ({words} words)")
    if not items:
        print("  no items were mined, so no page was built.")
        return
    page = folder / f"{name}.html"
    subtitle = f"Argument map mined live from the discussion on {datetime.now():%B %-d, %Y}"
    build = [PY, HERE / "mined-to-site.py", mined, "--title", a.title, "--subtitle", subtitle, "--out", page]
    subprocess.run([str(c) for c in build], stdout=subprocess.DEVNULL)
    subprocess.run([PY, str(HERE / "mined-to-argdown.py"), str(mined), "-o", str(folder / f"{name}.argdown"),
                    "--title", a.title], stdout=subprocess.DEVNULL)
    n = record["items"]
    print(f"  items:       {n['question']} questions, {n['claim']} claims, {n['evidence']} evidence")
    print(f"  page:        {page.name} (one file, ready to email), figures in {name}-figures/")
    print("\nTo split the page into sections, rebuild it with --part, e.g.:")
    print(f'  {files.tilde(HERE / "mined-to-site.py")} {files.tilde(mined)} --title "{a.title}" \\\n'
          f'    --part "10:30=First topic" --part "11:00=Second topic" --out {files.tilde(page)}')


if __name__ == "__main__":
    main()
