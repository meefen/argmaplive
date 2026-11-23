# argmaplive

Live argument maps from talk. Captions a discussion with whisper.cpp, mines the growing transcript for questions, claims and evidence with an LLM, and turns the result into an argument map: an Argdown file, or a single HTML page with one map per question that can be emailed or published.

Python standard library only. Needs a built [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (with `-DWHISPER_SDL2=ON`) for live captions, Graphviz (`brew install graphviz`) for the page, and either Ollama or a Gemini API key for mining.

| Script | What it does |
|---|---|
| `session.py` | Runs the next three together for one session, with all files in one folder; builds the page when it ends |
| `live-captions.py` | Runs whisper-stream on the microphone and shows big captions in their own window; `--save FILE` appends the finished lines to a transcript |
| `mine-transcript.py` | Follows a transcript as it grows and, about once a minute, extracts questions, claims and evidence into a JSONL file. Local model through Ollama by default; `--gemini` uses Google's API |
| `live-map.py` | A live argument map in its own window: redraws within seconds of each mining round, latest activity at the bottom, new items highlighted |
| `mined-to-argdown.py` | JSONL to an Argdown map (`--watch` rewrites it as the JSONL grows) |
| `mined-to-site.py` | JSONL to a self-contained HTML page plus a PNG and SVG per map |
| `settings.py`, `files.py` | Shared helpers: settings from `.env`, file names |

## Setup

```
cp .env.example .env && chmod 600 .env
```

Then edit `.env`: where whisper.cpp is, which whisper model and microphone, the Ollama address and model, the Gemini model and key, and the folder with the page styles (`SITE_DIR`). A shell environment variable with the same name overrides a line in `.env`, and command-line flags override both.

## A live meeting, with one command

```
./session.py --title "Lab meeting" --topic "Seminar on AI for teaching; names: Piaget, Skinner"
./session.py --title "Lab meeting" --topic "..." --gemini   # mine with Gemini instead of the local model
```

This starts the captions, the miner and the live map together, with all files in a new folder named after the title and start time, e.g. `~/Documents/captions/lab-meeting-2026-11-22-1030/` (`SESSIONS_DIR` in `.env`). `--title` is required; `--topic` (names, terms, what the session is about) goes to the miner, with the title, to help it fix misheard words. The captions and the map open in their own windows, and the terminal shows the mined items. Press Ctrl+C once to end: the captions stop, the miner mines the last bit of talk, and the page to share is built in the folder (`<name>.html`, plus `<name>.argdown` and `session.json`). Options: `--mic NAME`, `--no-map`, `--no-window`, `--dir FOLDER`.

## A live meeting, step by step

Keep transcripts and results outside this folder, for example in `~/Documents/captions/`.

```
./live-captions.py --save ~/Documents/captions/meeting.txt
./mine-transcript.py ~/Documents/captions/meeting.txt --topic "Lab meeting; terms: WonderBits, CSCL"
./live-map.py ~/Documents/captions/meeting.mined.jsonl                       # optional
```

Run each in its own terminal and stop each with Ctrl+C. `--save` creates the transcript and its folder. If the file already has text, the script stops before the mic turns on and suggests a dated name instead, so an earlier session is never overwritten or mixed in; add `--append` to continue a transcript on purpose (for example after a crash). Likewise, if a `.mined.jsonl` from an earlier session sits next to a new transcript, `live-captions.py` and `mine-transcript.py` stop and print the `mv` command that sets it aside, so old items never mix into a new session. After a meeting, rename the transcript and its `.mined.jsonl` together. `--topic` helps the model repair names and jargon that the captions mishear. Add `--gemini` to the miner for faster, better extraction, at the cost of sending the transcript to Google; leave it off for meetings that should stay on this machine.

`live-map.py` draws the same boxes as the shared page: a small map for each question or claim that has links, and single items as a list. A map moves to the bottom when it gets a new item, and new items glow for a minute. Scrolled to the bottom, the view follows along; scroll up to read and it stays put, with a button to jump back down. It can start before the JSONL exists, clears and starts over if the file is deleted or replaced, and says so when the items aren't from today.

`mined-to-argdown.py --watch` is the Argdown alternative, but the VS Code Argdown preview does not refresh when another program changes the file, and it lays every group side by side in one very wide map.

`./live-captions.py --list-mics` shows the microphones. `MIC` in `.env` (or `--mic NAME`) picks one by part of its name, which is safer than a number because numbers change when devices connect or disconnect.

Afterwards, build a page to share:

```
./mined-to-site.py ~/Documents/captions/meeting.mined.jsonl --title "Lab meeting" \
    --part "10:50=First topic" --part "11:30=Second topic"
```

For a live run this writes `meeting.mined.html` next to the JSONL: one file, no outside resources, ready to attach to an email. Times are the clock times when items were extracted.

## A recording or a YouTube talk

Any transcript works. Lines that start with a YouTube timestamp (`[1:02] ...`) or a whisper-cli timestamp (`[00:01:02.000 --> ...] ...`) give each item its place in the recording.

```
~/src/whisper.cpp/build/bin/whisper-cli -m ~/src/whisper.cpp/models/ggml-large-v3-turbo.bin -f talk.wav > talk.txt
./mine-transcript.py talk.txt --once --gemini --topic "..."
./mined-to-site.py talk.mined.jsonl --title "..." --out ~/src/quick-docs/my-talk
```

For a file named `youtube-<id>-transcript.txt`, every box on the page links to its moment in the video. The page goes to `<SITE_DIR>/<title slug>/index.html` unless `--out` says otherwise, ready for `npx wrangler pages deploy`.

## Fixing mistakes

The JSONL is the source of truth: one item per line, with `id`, `type` (question, claim, evidence), `text`, `quote` (the caption words), `confidence`, `asr_note` (caption repairs), `link_to` and `relation` (answers, supports, challenges), and `at` (time in the recording). Edit it, then rerun `mined-to-argdown.py` or `mined-to-site.py`. A miner restarted on the same transcript resumes where it stopped.

## Caveats

Captions and extraction both make errors, speakers are not attributed, and live mining in one-minute batches misses many links between items said minutes apart. Treat a map as one reading of a discussion.
