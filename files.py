"""Small file-name helpers shared by the argmaplive scripts."""

from datetime import date
from pathlib import Path


def tilde(path):
    """The path with the home folder written as ~, for messages people copy and paste."""
    path, home = str(path), str(Path.home())
    return "~" + path[len(home):] if path.startswith(home + "/") else path


def file_day(path):
    return date.fromtimestamp(Path(path).stat().st_mtime)


def dated_name(path, day=None):
    """A free name with a date added: talk.txt -> talk-2026-11-22.txt, then -2, -3, ...
    The date goes before all extensions, so talk.mined.jsonl -> talk-2026-11-22.mined.jsonl."""
    path = Path(path)
    base, dot, rest = path.name.partition(".")
    stamp, ext = f"{base}-{(day or date.today()):%Y-%m-%d}", dot + rest
    candidate, n = path.with_name(stamp + ext), 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{stamp}-{n}{ext}")
    return candidate


def leftover_message(mined, transcript):
    """Explains a .mined.jsonl left from an earlier session and how to set it aside."""
    return (f"{tilde(mined)} is left from an earlier session ({file_day(mined):%b %-d}), and {Path(transcript).name} "
            f"is new.\nThe miner would mix the old items into this session, so nothing was started. "
            f"Rename the old file first:\n  mv {tilde(mined)} {tilde(dated_name(mined, file_day(mined)))}")
