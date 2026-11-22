#!/usr/bin/env python3
"""Live captions window for whisper.cpp.

Runs whisper.cpp's whisper-stream on the microphone and shows the transcript in its
own browser window with large text. Uses only the Python standard library.

Usage:
  ./live-captions.py                         # settings from .env, continuous mode
  ./live-captions.py --save talk.txt         # also append finished lines to a file
  ./live-captions.py --list-mics             # show the microphones and their numbers
  ./live-captions.py --mic Logi              # pick the microphone by part of its name
  ./live-captions.py -m models/ggml-base.en.bin --step 0 --length 8000 -vth 0.6
                                             # other flags go to whisper-stream

Settings (in .env): WHISPER_CPP_DIR (where whisper.cpp is built), WHISPER_MODEL
(relative to WHISPER_CPP_DIR, as is -m), MIC (part of a microphone name). A MIC that
isn't connected falls back to the default microphone with a warning; -c N overrides it.

Keys in the window:  + / -  text size,  t  light/dark,  c  clear,  f  full screen
"""

import argparse
import codecs
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import settings

WHISPER_DIR = settings.path("WHISPER_CPP_DIR")
STREAM_BIN = WHISPER_DIR / "build" / "bin" / "whisper-stream"
SDL_LIBS = ("/opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib", "/usr/local/opt/sdl2/lib/libSDL2-2.0.0.dylib")
MAX_LINES = 300  # finished lines kept for the window's scrollback

TIMESTAMP = re.compile(r"^\[\d{2}:\d{2}[\d:.]*\s*-->\s*[\d:.]+\]\s*")  # phrase-mode prefix
ANNOTATION = re.compile(r"\[[^\]]*\]")  # [BLANK_AUDIO], [Music], [Start speaking] ...
SOUND_ONLY = re.compile(r"^[(*].*[)*]$")  # (silence), *music* ...


def clean(text):
    text = text.strip()
    if text.startswith("###"):  # phrase-mode START/END markers
        return ""
    text = ANNOTATION.sub("", TIMESTAMP.sub("", text)).strip()
    return "" if SOUND_ONLY.match(text) else text


class Captions:
    """Shared transcript state; web clients wait on `cond` for changes."""

    def __init__(self):
        self.cond = threading.Condition()
        self.lines, self.total, self.partial = [], 0, ""
        self.status, self.version = "loading", 0

    def _changed(self):
        self.version += 1
        self.cond.notify_all()

    def commit(self, line):
        with self.cond:
            self.lines.append(line)
            del self.lines[:-MAX_LINES]
            self.total += 1
            self.partial = ""
            self._changed()

    def set_partial(self, text):
        with self.cond:
            if text != self.partial:
                self.partial = text
                self._changed()

    def set_status(self, status):
        with self.cond:
            self.status = status
            self._changed()

    def snapshot(self):  # caller holds self.cond
        return json.dumps({"lines": self.lines, "total": self.total,
                           "partial": self.partial, "status": self.status})


class OutputParser:
    """Turns whisper-stream's terminal output into finished lines + the line in progress.

    Continuous mode redraws the current line with ESC[2K + \\r and ends it with \\n;
    phrase mode prints whole timestamped lines between ### markers.
    """

    def __init__(self, captions, on_line):
        self.captions, self.on_line = captions, on_line
        self.buf, self.esc = [], None

    def feed(self, text):
        for ch in text:
            if self.esc is not None:  # inside an ANSI escape sequence
                self.esc += ch
                if ch.isalpha():
                    if ch == "K":  # erase line
                        self.buf.clear()
                    self.esc = None
            elif ch == "\x1b":
                self.esc = ""
            elif ch == "\r":
                self.buf.clear()
            elif ch == "\n":
                self._end_line()
            else:
                self.buf.append(ch)
        self.captions.set_partial(clean("".join(self.buf)))

    def _end_line(self):
        raw = "".join(self.buf).strip()
        self.buf.clear()
        if raw == "[Start speaking]":
            self.captions.set_status("listening")
        text = clean(raw)
        if text:
            self.captions.commit(text)
            self.on_line(text)
        else:
            self.captions.set_partial("")


def make_handler(captions):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
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
                        with captions.cond:
                            captions.cond.wait_for(lambda: captions.version != seen, timeout=15)
                            changed = captions.version != seen
                            seen = captions.version
                            data = captions.snapshot()
                        msg = f"data: {data}\n\n" if changed else ": ping\n\n"
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def with_defaults(extra):
    def has(*flags):
        return any(a in flags for a in extra)

    args = list(extra)
    if not has("-m", "--model"):
        args += ["-m", str(settings.path("WHISPER_MODEL", base=WHISPER_DIR))]
    if not has("-t", "--threads"):
        args += ["-t", "8"]
    if not has("--step"):
        # whisper-stream drops audio when one update takes over 2x the step. medium.en needs
        # ~0.6s per update alone and ~1.3s while an LLM (mine-transcript.py) shares the GPU.
        args += ["--step", "1000"] + ([] if has("--length") else ["--length", "5000"])
    return args


def capture_devices():
    """Microphone names in the order whisper-stream numbers them (-c N). Opens no device."""
    import ctypes
    import ctypes.util

    lib = next((p for p in SDL_LIBS if Path(p).exists()), None) or ctypes.util.find_library("SDL2")
    if not lib:
        return None
    sdl = ctypes.CDLL(lib)
    sdl.SDL_GetAudioDeviceName.restype = ctypes.c_char_p
    if sdl.SDL_Init(0x10) != 0:  # SDL_INIT_AUDIO
        return None
    try:
        return [(sdl.SDL_GetAudioDeviceName(i, 1) or b"?").decode("utf-8", "replace")
                for i in range(sdl.SDL_GetNumAudioDevices(1))]
    finally:
        sdl.SDL_Quit()


def device_list(devices):
    return "\n".join(f"  -c {i}   {name}" for i, name in enumerate(devices))


def pick_mic(name, required):
    """Number of the first microphone whose name contains `name`, or None for the default."""
    devices = capture_devices()
    if devices is None:
        problem = "can't list microphones (SDL2 not found)"
    else:
        for i, device in enumerate(devices):
            if name.lower() in device.lower():
                print(f"Microphone: {device} (-c {i})")
                return i
        problem = f"no microphone matches '{name}'. Connected:\n{device_list(devices)}"
    if required:
        sys.exit(problem)
    print(f"! MIC={name} in .env: {problem}\n! Using the default microphone.")
    return None


def open_window(url):
    # Chromium browsers can open a page as a standalone app window (no tabs or address bar).
    for app in ("Google Chrome", "Brave Browser", "Microsoft Edge"):
        if Path(f"/Applications/{app}.app").exists():
            subprocess.Popen(["open", "-na", app, "--args", f"--app={url}"])
            return
    subprocess.Popen(["open", url])


def main():
    ap = argparse.ArgumentParser(
        description="Show whisper-stream live transcription in a big-text window.",
        epilog="Any other flags are passed through to whisper-stream (e.g. -c 1, --step 0, -m MODEL). "
               f"Settings: {settings.ENV_FILE}",
        allow_abbrev=False)
    ap.add_argument("--port", type=int, default=8765, help="local web port (default 8765)")
    ap.add_argument("--save", metavar="FILE", help="append finished lines to FILE")
    ap.add_argument("--no-window", action="store_true", help="don't open a browser window")
    ap.add_argument("--mic", metavar="NAME", help="microphone by part of its name, e.g. Logi (default: MIC in .env)")
    ap.add_argument("--list-mics", action="store_true", help="list the microphones and exit")
    args, stream_args = ap.parse_known_args()

    if args.list_mics:
        devices = capture_devices()
        if devices is None:
            sys.exit("Can't list microphones (SDL2 not found).")
        print(device_list(devices))
        return
    if not STREAM_BIN.exists():
        sys.exit(f"{STREAM_BIN} not found. Set WHISPER_CPP_DIR in {settings.ENV_FILE} to your whisper.cpp\n"
                 "folder, built with -DWHISPER_SDL2=ON.")
    if not any(a in ("-c", "--capture") for a in stream_args):
        name = args.mic if args.mic is not None else settings.get("MIC")
        index = pick_mic(name, required=args.mic is not None) if name else None
        if index is not None:
            stream_args += ["-c", str(index)]

    captions = Captions()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(captions))
    except OSError:  # port busy: pick any free one
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(captions))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    save = open(args.save, "a", encoding="utf-8") if args.save else None

    def on_line(text):
        print(text, flush=True)
        if save:
            save.write(text + "\n")
            save.flush()

    log_path = Path(tempfile.gettempdir()) / "live-captions-whisper.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen([str(STREAM_BIN), *with_defaults(stream_args)],
                                cwd=WHISPER_DIR, stdout=subprocess.PIPE, stderr=log)

    def pump():
        parser = OutputParser(captions, on_line)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        fd = proc.stdout.fileno()
        while chunk := os.read(fd, 4096):
            parser.feed(decoder.decode(chunk))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    print(f"Captions window: {url}")
    print(f"whisper-stream log: {log_path}")
    print("Press Ctrl+C to stop.\n")
    if not args.no_window:
        open_window(url)

    def stop(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    interrupted = False
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        # Make sure the mic is released even if the signal only reached this script.
        interrupted = True
        proc.terminate()
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()

    reader.join(timeout=2)
    captions.set_status("stopped")
    time.sleep(0.3)  # let the window receive the final state
    if save:
        save.close()
    if code != 0 and not interrupted:
        print(f"\nwhisper-stream exited with code {code}. Last log lines:")
        print("".join(log_path.read_text(errors="replace").splitlines(True)[-15:]))
        sys.exit(code)


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Live captions</title>
<style>
  :root { --bg: #0e0f11; --fg: #f4f4f2; --dim: #7d8087; --size: 52px; }
  body.light { --bg: #fbfaf7; --fg: #16181b; --dim: #8a8d93; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); }
  body { font: 500 var(--size)/1.32 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; }
  #scroll { height: 100%; overflow-y: auto; box-sizing: border-box;
            padding: 4vh 5vw 14vh; display: flex; flex-direction: column; }
  #text { margin-top: auto; }  /* keeps text anchored at the bottom, still scrollable */
  .line { color: var(--dim); margin: 0 0 .3em; }
  .line:last-child { color: var(--fg); opacity: .72; }
  #partial { color: var(--fg); min-height: 1.32em; }
  #bar { position: fixed; left: 0; right: 0; bottom: 0; padding: 12px 5vw;
         display: flex; gap: 22px; align-items: center;
         font: 14px -apple-system, sans-serif; color: var(--dim);
         background: linear-gradient(transparent, var(--bg) 45%); transition: opacity .6s; }
  body.idle #bar { opacity: 0; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
         margin-right: 7px; background: #d9a400; }
  #status.listening .dot { background: #2fbf71; }
  #status.stopped .dot, #status.offline .dot { background: #d9534f; }
  #status { color: var(--fg); margin-right: auto; }
  kbd { font: 12px ui-monospace, monospace; border: 1px solid var(--dim);
        border-radius: 4px; padding: 0 5px; margin-right: 3px; }
</style>
</head>
<body>
<div id="scroll"><div id="text"><div id="lines"></div><div id="partial"></div></div></div>
<div id="bar">
  <span id="status"><span class="dot"></span><span id="status-text">Connecting…</span></span>
  <span><kbd>+</kbd><kbd>-</kbd>size</span>
  <span><kbd>t</kbd>theme</span>
  <span><kbd>c</kbd>clear</span>
  <span><kbd>f</kbd>full screen</span>
</div>
<script>
const $ = id => document.getElementById(id);
const scroller = $('scroll');
const labels = { loading: 'Loading model…', listening: 'Listening', stopped: 'Stopped', offline: 'Disconnected' };
let size = +localStorage.captionSize || 52, clearedAt = 0, last = null;

function setSize(s) {
  size = Math.max(20, Math.min(160, s));
  localStorage.captionSize = size;
  document.documentElement.style.setProperty('--size', size + 'px');
}
function setStatus(st) {
  $('status').className = st;
  $('status-text').textContent = labels[st] || st;
}
function render(s, forceBottom) {
  last = s;
  const atBottom = forceBottom || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  const first = s.total - s.lines.length;
  const shown = s.lines.slice(Math.max(0, clearedAt - first));
  $('lines').replaceChildren(...shown.map(t => {
    const d = document.createElement('div');
    d.className = 'line';
    d.textContent = t;
    return d;
  }));
  $('partial').textContent = s.partial;
  setStatus(s.status);
  if (atBottom) scroller.scrollTop = scroller.scrollHeight;
}

setSize(size);
if (localStorage.captionTheme === 'light') document.body.classList.add('light');

const events = new EventSource('/events');
events.onmessage = e => render(JSON.parse(e.data));
events.onerror = () => setStatus('offline');

addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === '+' || e.key === '=') setSize(size + 4);
  else if (e.key === '-' || e.key === '_') setSize(size - 4);
  else if (e.key === 't') {
    document.body.classList.toggle('light');
    localStorage.captionTheme = document.body.classList.contains('light') ? 'light' : 'dark';
  }
  else if (e.key === 'c') { if (last) clearedAt = last.total; }
  else if (e.key === 'f') {
    document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  }
  else return;
  if (last) render(last, true);
});

// Hide the help bar when the mouse and keyboard are idle.
let idleTimer;
function wake() {
  document.body.classList.remove('idle');
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => document.body.classList.add('idle'), 3000);
}
addEventListener('mousemove', wake);
addEventListener('keydown', wake);
wake();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
