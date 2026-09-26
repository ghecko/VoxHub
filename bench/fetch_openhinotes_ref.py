#!/usr/bin/env python3
"""
Pull a hand-corrected transcription out of OpenHiNotes as a bench reference.

    python bench/fetch_openhinotes_ref.py --base https://hinotes.local \
        --email you@example.com --password '...' \
        --id 6f1c...-uuid --stem meeting_scope --out bench/data

Writes ``<out>/<stem>.ref.json`` (segments with the speaker DISPLAY names you
set in the UI, so merged/renamed speakers are honoured) and, when the audio
is still stored on the server, ``<out>/<stem>.<ext>`` next to it. The
result is exactly what ``run_bench.py`` expects.

Workflow: transcribe in OpenHiNotes, fix it in the UI (merge / split /
reassign speakers, edit text), then run this script. Correcting a draft is
about 10x faster than transcribing from scratch.

Authentication: ``--token`` (the value of ``auth_token`` in the browser's
localStorage) or ``--email`` + ``--password``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error


def _sniff_ext(head: bytes) -> str | None:
    """Guess the audio container from its first bytes (what run_bench accepts)."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return ".mp3"
    return None


def _disposition_ext(header: str) -> str | None:
    """Extension of the filename in a Content-Disposition header, if any."""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', header)
    if not m:
        return None
    ext = os.path.splitext(m.group(1).strip())[1].lower()
    return ext or None


def _req(base: str, path: str, token: str | None = None, data: dict | None = None):
    url = f"{base.rstrip('/')}/api{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    req.add_header("Accept", "application/json")
    if body:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="OpenHiNotes URL, e.g. https://hinotes.local")
    ap.add_argument("--id", required=True, help="Transcription id (UUID from the page URL)")
    ap.add_argument("--stem", required=True, help="File stem for the reference, e.g. meeting_scope")
    ap.add_argument("--out", default="bench/data")
    ap.add_argument("--token")
    ap.add_argument("--email")
    ap.add_argument("--password")
    ap.add_argument("--no-audio", action="store_true", help="Only write the .ref.json")
    ap.add_argument("--insecure", action="store_true", help="Skip TLS verification (self-signed)")
    args = ap.parse_args()

    if args.insecure:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

    token = args.token
    if not token:
        if not (args.email and args.password):
            print("Provide --token or --email/--password", file=sys.stderr)
            return 2
        with _req(args.base, "/auth/login", data={"email": args.email, "password": args.password}) as r:
            token = json.load(r)["access_token"]

    with _req(args.base, f"/transcriptions/{args.id}", token) as r:
        t = json.load(r)

    names = t.get("speakers") or {}
    ref = []
    for seg in t.get("segments") or []:
        if seg.get("start") is None or seg.get("end") is None:
            continue
        label = seg.get("speaker") or "SPEAKER_00"
        ref.append({
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": names.get(label, label),
            "text": (seg.get("text") or "").strip(),
        })
    os.makedirs(args.out, exist_ok=True)
    ref_path = os.path.join(args.out, f"{args.stem}.ref.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=1)
    speakers = sorted({s["speaker"] for s in ref})
    print(f"wrote {ref_path}: {len(ref)} segments, speakers={speakers}")
    if any(s.startswith("SPEAKER_") for s in speakers):
        print("  note: some speakers still carry generic labels; fine for scoring, "
              "but rename them in the UI if you want readable mappings")

    if args.no_audio:
        return 0
    if not t.get("audio_available"):
        print("audio not stored on the server for this transcription (keep_audio was off); "
              "re-upload from the HiDock with 'Save audio on server' or copy the file by hand")
        return 0
    try:
        with _req(args.base, f"/transcriptions/audio/{args.id}", token) as r:
            # HiDock uploads are stored as .hda and served as octet-stream, so
            # the Content-Type is useless: sniff the container instead, and
            # fall back to the original filename's extension.
            head = r.read(1 << 20)
            ext = _sniff_ext(head) or _disposition_ext(r.headers.get("Content-Disposition", "")) or ".hda"
            audio_path = os.path.join(args.out, f"{args.stem}{ext}")
            with open(audio_path, "wb") as f:
                f.write(head)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"wrote {audio_path}")
    except urllib.error.HTTPError as e:
        print(f"audio download failed: HTTP {e.code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
