"""Shared settings for the argmaplive scripts.

Each setting comes from the first of these that has it:
  1. a command-line flag (where the script offers one)
  2. a shell environment variable with the same name
  3. the .env file next to these scripts (see .env.example)
  4. the default below
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"

DEFAULTS = {
    "WHISPER_CPP_DIR": "~/src/whisper.cpp",
    "WHISPER_MODEL": "models/ggml-medium.en.bin",  # relative to WHISPER_CPP_DIR
    "MIC": "",                                      # empty: the system default microphone
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    "OLLAMA_MODEL": "qwen3.8:27b-mlx",
    "GEMINI_MODEL": "gemini-4-hydrogen",
    "GEMINI_API_KEY": "",
    "SITE_DIR": "~/src/quick-docs",
    "SESSIONS_DIR": "~/Documents/captions",        # session.py makes one folder per session here
}


def _read_env_file(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ")
        name, sep, value = line.partition("=")
        if not sep or line.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()  # allow a comment after the value
        values[name.strip()] = value
    return values


_FILE = _read_env_file(ENV_FILE)


def get(name):
    """The setting as a string ("" if unset)."""
    for source in (os.environ, _FILE, DEFAULTS):
        if source.get(name):
            return source[name]
    return ""


def path(name, base=None):
    """The setting as a Path, with ~ expanded; relative paths are taken from `base` if given."""
    p = Path(get(name)).expanduser()
    return Path(base) / p if base is not None and not p.is_absolute() else p
