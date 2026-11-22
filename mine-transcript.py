#!/usr/bin/env python3
"""Mine a transcript for claims, evidence and questions as it grows.

Follows a transcript file (for example the one `live-captions.py --save` writes)
and every minute or so sends the new text to an LLM. Items stream to the terminal
and to a JSONL file, one item per line. Uses only the Python standard library.

By default the LLM runs locally through Ollama and nothing leaves this machine.
With --gemini it uses Google's Gemini API instead; put GEMINI_API_KEY=... in the
.env file next to this script. Models and the Ollama address are set there too.

Usage:
  ./mine-transcript.py talk.txt                       # follow talk.txt as it grows
  ./mine-transcript.py talk.txt --gemini              # same, with GEMINI_MODEL
  ./mine-transcript.py talk.txt --topic "Seminar on feedback; names: Scardamalia, Bereiter"
  ./mine-transcript.py recording.txt --once           # mine a finished transcript and exit

Output goes to talk.mined.jsonl next to the transcript unless -o is given.
Restarting on the same transcript resumes where the last run stopped.
Lines may start with a timestamp, "[1:02] ..." (YouTube) or "[00:01:02.000 --> ...] ..."
(whisper-cli); items then record where in the recording their quote starts ("at").
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import settings

OLLAMA_MODEL = settings.get("OLLAMA_MODEL")
GEMINI_MODEL = settings.get("GEMINI_MODEL")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TYPES = ("claim", "evidence", "question")
PREFIX = {"claim": "c", "evidence": "e", "question": "q"}
RECENT_ITEMS = 20  # shown to the model so it doesn't repeat itself and can link back

SYSTEM_PROMPT = """\
You help a researcher follow a live discussion (a class, seminar, or meeting) by pulling out its argument components as it unfolds. You read text from automatic speech recognition (ASR).

The transcript is noisy:
- Words are often misheard, especially names, technical terms, and short words (for example "non-dining vote" for "non-binding vote").
- Punctuation and sentence boundaries are unreliable, and phrases can repeat across line breaks.
- There are no speaker labels, and several people may be talking.

Extract three kinds of items:
- claim: a position someone puts forward that others could agree or disagree with, such as an assertion, interpretation, proposal, prediction, or judgment.
- evidence: something offered as grounds for or against a claim, such as data, an example, an experience, an observation, a source, or a reason.
- question: a substantive question or wondering raised in the discussion. Skip procedural and social questions ("can everyone hear me?", "should we take a break?").

How to write each item:
- text: one clear, self-contained sentence that makes sense to someone who did not hear the discussion. Paraphrase faithfully. Do not add facts, reasons, or details that were not said.
- Repair misheard words, names, and terms in text whenever the intended word is reasonably clear from context or the TOPIC, including well-known scholars, theories, and products (for example "vie got ski's zone of proximal development" is "Vygotsky's zone of proximal development").
- quote: the exact words from NEW TEXT the item is based on, copied verbatim with any errors left in. Keep it under 40 words.
- confidence: "high" if clearly stated and clearly transcribed, "medium" if somewhat implicit or the wording needed repair, "low" if the transcript is garbled or your reading is a guess.
- If you cannot tell what a misheard name or term was, do not guess: keep it as heard, add "(unclear)", and set confidence to "low". A wrong name is worse than an unclear one.
- asr_note: if you repaired a likely misrecognition, name the change in a few words, for example "'non-dining' read as 'non-binding'". Otherwise "". Keep it under 15 words and do not reason in it.
- link_to and relation: read them as "this item <relation> link_to". Evidence supports or challenges a claim. A claim can answer a question, or support or challenge another claim. A question can challenge a claim. Give the target's id (from RECENT ITEMS, or the tmp id of another item in your output) and the relation "supports", "challenges", or "answers". If there is no clear link, use "" and "none".
- tmp: a temporary id for the item: n1, n2, n3, ...

Rules:
- Extract only from NEW TEXT. EARLIER TEXT is context; its items were already extracted.
- A thought that starts in EARLIER TEXT and is completed in NEW TEXT counts as new.
- If NEW TEXT ends mid-thought, leave the unfinished part for the next round.
- A claim's text states the position only, never its reasons. When someone says "X because Y" or "X; Y shows it", the claim is X, and Y is a separate evidence item that links to the claim with "supports".
- Do not repeat anything already in RECENT ITEMS, even reworded.
- Prefer a few meaningful items over many trivial ones. Skip small talk, logistics, and passages too garbled to interpret. Returning no items is fine.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tmp": {"type": "string"},
                    "type": {"type": "string", "enum": list(TYPES)},
                    "text": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "asr_note": {"type": "string"},
                    "link_to": {"type": "string"},
                    "relation": {"type": "string", "enum": ["supports", "challenges", "answers", "none"]},
                },
                "required": ["tmp", "type", "text", "quote", "confidence", "asr_note", "link_to", "relation"],
            },
        }
    },
    "required": ["items"],
}


class LLMUnavailable(Exception):
    """The model couldn't be reached right now; the chunk stays pending and is retried."""


TRANSIENT = {429, 500, 502, 503, 504}  # busy / rate limited / server hiccup: worth retrying


def post_json(url, body, headers, service, timeout=600, retry_waits=()):
    data = json.dumps(body).encode()
    for wait in (*retry_waits, None):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"] if service == "Gemini" else json.loads(detail)["error"]
            except (ValueError, KeyError, TypeError):
                pass
            error = LLMUnavailable(f"{service} error {e.code}: {detail}")
            if e.code not in TRANSIENT:
                raise error from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            hint = " Is the Ollama app running?" if service == "Ollama" else ""
            error = LLMUnavailable(f"can't reach {service} ({e}).{hint}")
        if wait is None:
            raise error
        print(f"  ! {error}\n  ! retrying in {wait}s...", flush=True)
        time.sleep(wait)


def ask_gemini(api_key, model, system, user):
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": SCHEMA}},
            "thinkingConfig": {"thinkingLevel": "low"},  # extraction doesn't need deep thinking
        },
    }
    reply = post_json(GEMINI_URL.format(model=model), body, {"x-goog-api-key": api_key}, "Gemini",
                      timeout=150, retry_waits=(10, 30, 60))
    candidates = reply.get("candidates") or []
    if not candidates:
        raise ValueError(f"no answer (blocked: {reply.get('promptFeedback', {}).get('blockReason', 'unknown')})")
    parts = candidates[0].get("content", {}).get("parts", [])
    return json.loads("".join(p["text"] for p in parts if p.get("text") and not p.get("thought")))


def ask_ollama(host, model, system, user):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": SCHEMA,
        "think": False,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    reply = post_json(f"{host}/api/chat", body, {}, "Ollama")
    return json.loads(reply["message"]["content"])


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def shorten(text, limit):
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def valid_link(src_type, relation, target_type):
    """Only answers -> question; supports/challenges -> claim or evidence; questions can only challenge."""
    if relation == "answers":
        return target_type == "question" and src_type != "question"
    if relation == "supports":
        return target_type != "question" and src_type != "question"
    if relation == "challenges":
        return target_type != "question"
    return False


# "[1:02] text" (YouTube) or "[00:01:02.000 --> 00:01:05.000] text" (whisper-cli)
STAMP = re.compile(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,]\d+)?(?:\s*-->\s*[\d:.,]+)?\]\s*")


def split_stamp(line):
    m = STAMP.match(line)
    if not m:
        return None, line
    h, mins, secs = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    return (f"{h}:{mins:02d}:{secs:02d}" if h else f"{mins}:{secs:02d}"), line[m.end():]


def find_stamp(quote, chunk):
    """Timestamp of the transcript line where the quote starts, if the lines had timestamps."""
    seq = [(w, stamp) for line, _, _, stamp in chunk for w in words(line)]
    q = words(quote)[:6]
    if not q or not any(stamp for _, stamp in seq):
        return None
    best, best_i = 0, None
    for i in range(len(seq)):
        hits = sum(1 for j, w in enumerate(q) if i + j < len(seq) and seq[i + j][0] == w)
        if hits > best:
            best, best_i = hits, i
    return seq[best_i][1] if best_i is not None and best >= max(2, 0.6 * len(q)) else None


def is_grounded(quote, source):
    """True if most of the quote's words appear in the transcript we sent."""
    q = words(quote)
    if not q:
        return False
    src = set(words(source))
    return sum(w in src for w in q) / len(q) >= 0.6


class Style:
    def __init__(self, enabled):
        def code(c):
            return c if enabled else ""
        self.reset, self.dim, self.bold = code("\033[0m"), code("\033[2m"), code("\033[1m")
        self.type = {"claim": code("\033[36m"), "evidence": code("\033[32m"), "question": code("\033[33m")}


class Miner:
    def __init__(self, args):
        self.args = args
        self.path = Path(args.transcript).expanduser().resolve()
        self.out_path = Path(args.out).expanduser() if args.out else self.path.with_suffix(".mined.jsonl")
        self.style = Style(sys.stdout.isatty())
        self.items, self.counts = [], {t: 0 for t in TYPES}
        self.pending = []  # (line, end byte offset, arrival time) not yet mined
        self.read_pos, self.batch, self.context = 0, 0, ""
        self.timestamped = False  # transcript lines carry recording timestamps
        if args.gemini:
            self.ask = lambda system, user: ask_gemini(args.api_key, args.model, system, user)
        else:
            self.ask = lambda system, user: ask_ollama(args.host, args.model, system, user)
        self._resume()

    def _resume(self):
        if not self.out_path.exists():
            return
        for raw in self.out_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(raw)
                n = int(item["id"][1:])
                self.counts[item["type"]] = max(self.counts[item["type"]], n)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
            self.items.append(item)
            self.read_pos = max(self.read_pos, item.get("pos", 0))
            self.batch = max(self.batch, item.get("batch", 0))
        if self.items:
            print(f"Resuming: {len(self.items)} items already in {self.out_path.name}")

    # --- reading the transcript -------------------------------------------------

    def read_new_lines(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.read_pos)
                data = f.read()
        except FileNotFoundError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return  # no complete line yet
        now = datetime.now()
        for raw in data[: end + 1].splitlines(keepends=True):
            self.read_pos += len(raw)
            stamp, line = split_stamp(raw.decode("utf-8", "replace").strip())
            self.timestamped |= stamp is not None
            if line:
                self.pending.append((line, self.read_pos, now, stamp))

    def pending_words(self):
        return sum(len(line.split()) for line, *_ in self.pending)

    def drain(self):
        """Mine everything pending in chunks of about --max-words. Returns False if Ollama is unreachable."""
        while self.pending:
            n, count = 0, 0
            while n < len(self.pending) and (n == 0 or count < self.args.max_words):
                count += len(self.pending[n][0].split())
                n += 1
            if not self.mine(self.pending[:n]):
                return False
            del self.pending[:n]
        return True

    # --- mining -----------------------------------------------------------------

    def build_prompt(self, new_text):
        recent = self.items[-RECENT_ITEMS:]
        recent_text = "\n".join(f"[{i['id']}] {i['type']}: {i['text']}" for i in recent) or "(none)"
        return (f"TOPIC (from the user; may help with names and terms): {self.args.topic or 'not given'}\n\n"
                f"RECENT ITEMS:\n{recent_text}\n\n"
                f"EARLIER TEXT:\n{self.context or '(start of discussion)'}\n\n"
                f"NEW TEXT:\n{new_text}\n")

    def mine(self, chunk):
        new_text = " ".join(line for line, *_ in chunk)
        end_pos, heard = chunk[-1][1], chunk[-1][2]
        started = time.time()
        try:
            result = self.ask(SYSTEM_PROMPT, self.build_prompt(new_text))
            raw_items = result.get("items", [])
        except LLMUnavailable as e:
            print(f"{self.style.dim}! {e} Will retry.{self.style.reset}", flush=True)
            return False
        except (ValueError, KeyError, AttributeError) as e:  # ValueError covers bad JSON
            print(f"{self.style.dim}! Model reply wasn't usable ({e}); skipping this chunk.{self.style.reset}", flush=True)
            raw_items = []

        self.batch += 1
        source = f"{self.context} {new_text}"
        seen_texts = {re.sub(r"\W+", " ", i["text"].lower()).strip() for i in self.items}
        tmp_to_id, added = {}, []

        for raw in raw_items:
            kind, text = raw.get("type"), (raw.get("text") or "").strip()
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if kind not in TYPES or not text or key in seen_texts:
                continue
            seen_texts.add(key)
            self.counts[kind] += 1
            item_id = f"{PREFIX[kind]}{self.counts[kind]}"
            if raw.get("tmp"):
                tmp_to_id[raw["tmp"]] = item_id
            grounded = is_grounded(raw.get("quote", ""), source)
            added.append((raw, {
                "id": item_id,
                "type": kind,
                "text": text,
                "quote": (raw.get("quote") or "").strip(),
                "confidence": raw.get("confidence", "low") if grounded else "low",
                "grounded": grounded,
                "asr_note": shorten((raw.get("asr_note") or "").strip(), 160),
                "link_to": None,
                "relation": None,
                "time": heard.isoformat(timespec="seconds"),
                "at": find_stamp(raw.get("quote", ""), chunk),  # position in the recording, if timestamped
                "batch": self.batch,
                "pos": end_pos,
                "source": str(self.path),
                "model": self.args.model,
            }))

        # Links can point at earlier items or at items from this same reply.
        types = {i["id"]: i["type"] for i in self.items}
        types.update({item["id"]: item["type"] for _, item in added})
        for raw, item in added:
            target = tmp_to_id.get(raw.get("link_to", ""), raw.get("link_to", ""))
            relation = raw.get("relation")
            if target in types and target != item["id"] and valid_link(item["type"], relation, types[target]):
                item["link_to"], item["relation"] = target, relation

        with open(self.out_path, "a", encoding="utf-8") as out:
            for _, item in added:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                self.items.append(item)
                self.show(item)

        s = self.style
        print(f"{s.dim}── batch {self.batch} · {len(new_text.split())} words · "
              f"{len(added)} items · {time.time() - started:.0f}s{s.reset}", flush=True)
        self.context = " ".join(f"{self.context} {new_text}".split()[-self.args.context_words:])
        return True

    def show(self, item):
        s = self.style
        dim = s.dim if item["confidence"] == "low" else ""
        tail = [item["confidence"]]
        if item["link_to"]:
            tail.append(f"{item['relation']} {item['link_to']}")
        if not item["grounded"]:
            tail.append("quote not found in transcript")
        # Recording position if known; "?" if the quote couldn't be placed; else wall-clock time (live).
        clock = item.get("at") or ("?" if self.timestamped else item["time"][11:16])
        print(f"{dim}{clock:>7}  {s.type[item['type']]}{s.bold}{item['type'].upper():<8}{s.reset}{dim} "
              f"{item['id']:<4} {item['text']}  {s.dim}({', '.join(tail)}){s.reset}")
        if item["asr_note"]:
            print(f"{s.dim}{'':22}ASR: {item['asr_note']}{s.reset}")
        sys.stdout.flush()

    # --- main loop --------------------------------------------------------------

    def run(self):
        a = self.args
        if not self.path.exists():
            if a.once:
                sys.exit(f"{self.path} not found")
            print(f"Waiting for {self.path} to appear...")
        mode = "reading once" if a.once else f"following, mining every {a.interval}s"
        print(f"Transcript: {self.path}  ({mode})")
        where = "Gemini API, text is sent to Google" if a.gemini else "local, via Ollama"
        print(f"Model: {a.model} ({where})   Output: {self.out_path}")
        if not a.once:
            print("Press Ctrl+C to stop.")
        print()

        idle_ticks = 0
        try:
            while True:
                tick = time.time()
                self.read_new_lines()
                n = self.pending_words()
                # Mine once enough new talk has piled up, or when talk has paused for a couple of ticks.
                if n and (a.once or n >= a.min_words or idle_ticks >= 2):
                    ok = self.drain()
                    if a.once and not ok:
                        sys.exit(f"Stopped after {len(self.items)} items because the model is unavailable. "
                                 "Run the same command again later; it resumes where it stopped.")
                    idle_ticks = 0
                elif n:
                    idle_ticks += 1
                if a.once:
                    break
                time.sleep(max(1.0, a.interval - (time.time() - tick)))
        except KeyboardInterrupt:
            self.read_new_lines()
            if self.pending:
                print("\nMining the last bit of transcript... (Ctrl+C again to skip)")
                try:
                    self.drain()
                except KeyboardInterrupt:
                    pass
        print(f"\n{len(self.items)} items in {self.out_path}")


def main():
    ap = argparse.ArgumentParser(description="Stream claims, evidence and questions out of a growing transcript.")
    ap.add_argument("transcript", help="transcript text file, e.g. from live-captions.py --save")
    ap.add_argument("-o", "--out", help="JSONL output file (default: <transcript>.mined.jsonl)")
    ap.add_argument("--topic", default="", help="what the discussion is about, names, jargon; helps fix ASR errors")
    ap.add_argument("--once", action="store_true", help="mine what's in the file now and exit")
    ap.add_argument("--interval", type=int, default=60, help="seconds between mining rounds (default 60)")
    ap.add_argument("--min-words", type=int, default=40, help="new words needed before a round runs (default 40)")
    ap.add_argument("--max-words", type=int, default=400, help="max new words sent per request (default 400)")
    ap.add_argument("--context-words", type=int, default=250, help="earlier words sent as context (default 250)")
    ap.add_argument("--gemini", action="store_true",
                    help="use Google's Gemini API instead of the local model (key in .env; sends text to Google)")
    ap.add_argument("--model", help=f"model name (default {OLLAMA_MODEL}, or {GEMINI_MODEL} with --gemini)")
    ap.add_argument("--host", default=settings.get("OLLAMA_HOST"), help="Ollama address (default: OLLAMA_HOST)")
    args = ap.parse_args()
    args.model = args.model or (GEMINI_MODEL if args.gemini else OLLAMA_MODEL)
    args.api_key = settings.get("GEMINI_API_KEY") if args.gemini else None
    if args.gemini and not args.api_key:
        sys.exit(f"--gemini needs a Gemini API key. Add a line GEMINI_API_KEY=... to {settings.ENV_FILE}\n"
                 "(create a key at https://aistudio.google.com/apikey).")
    Miner(args).run()


if __name__ == "__main__":
    main()
