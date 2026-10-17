"""Download the model files for a profile into ComfyUI/models/. Resumable.

Usage:
    python download_models.py --profile klein-4b
    python download_models.py --list

For gated repos (klein-9b) provide a Hugging Face read token as HF_TOKEN, either
in a git-ignored .env file next to this script (see .env.example) or as an
environment variable. A real environment variable wins over .env.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

from profiles import DEFAULT_PROFILE, PROFILES

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "ComfyUI" / "models"
CHUNK = 1024 * 1024


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=value lines, '#' comments, optional quotes."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv()


def _headers(extra=None):
    h = dict(extra or {})
    token = os.environ.get("HF_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = requests.head(url, headers=_headers(), allow_redirects=True, timeout=30)
    if head.status_code in (401, 403):
        sys.exit(
            f"Access denied for {url}\n"
            "This repo is gated. Accept its terms on huggingface.co, then put your token "
            "in the .env file as HF_TOKEN=hf_... and re-run."
        )
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))

    if dest.exists() and (total == 0 or dest.stat().st_size == total):
        print(f"  already present: {dest.name}")
        return

    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if total and done > total:
        part.unlink()
        done = 0

    headers = _headers({"Range": f"bytes={done}-"} if done else {})
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # nothing left to fetch
            part.replace(dest)
            return
        r.raise_for_status()
        if done and r.status_code != 206:  # server ignored Range; start over
            done = 0
        mode = "ab" if done else "wb"
        last = 0.0
        with open(part, mode) as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 1:
                    last = now
                    pct = f" {done / total * 100:5.1f}%" if total else ""
                    print(f"\r  {dest.name}: {done / 1e9:6.2f} / {total / 1e9:.2f} GB{pct}", end="", flush=True)
    print()

    if total and part.stat().st_size != total:
        sys.exit(f"Incomplete download of {dest.name}; re-run to resume.")
    part.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=PROFILES)
    ap.add_argument("--list", action="store_true", help="list profiles and exit")
    args = ap.parse_args()

    if args.list:
        for name, p in PROFILES.items():
            print(f"{name:14} {p['description']}")
        return

    profile = PROFILES[args.profile]
    if profile["gated"] and not os.environ.get("HF_TOKEN"):
        sys.exit(f"Profile '{args.profile}' is gated: put HF_TOKEN=hf_... in the .env file "
                 "(copy .env.example to .env) and re-run.")

    print(f"Downloading profile '{args.profile}': {profile['description']}")
    for f in profile["files"]:
        print(f"- {f['folder']}/{f['name']}")
        download(f["url"], MODELS_DIR / f["folder"] / f["name"])
    print("All models ready.")


if __name__ == "__main__":
    main()
