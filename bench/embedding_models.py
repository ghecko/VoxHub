#!/usr/bin/env python3
"""
Compare speaker-embedding models on your own recordings.

The question a voice profile has to answer is "is this the same person as in
that other recording?", so the numbers that matter are cross-recording: the
distance between the same name in two files versus the distance between two
different names. This script takes the bench references (``<stem>.ref.json``
with human speaker names, consistent across files), builds one embedding per
(file, speaker) from the reference segments with each candidate model, and
prints for every model the same-voice and different-voice distributions and
the margin between them. A model is usable when the largest same-voice
distance sits clearly below the smallest different-voice one.

    docker compose exec voxhub-api python bench/embedding_models.py --data bench/data \\
        --model pyannote/embedding \\
        --model pyannote/wespeaker-voxceleb-resnet34-LM \\
        --model speechbrain/spkrec-ecapa-voxceleb \\
        --model diarization          # whatever the diarization pipeline clusters with

Names that denote the same voice under two labels (e.g. two Teams
announcements) can be merged with ``--alias MS2=MS``. Speakers with less
than ``--min-seconds`` of reference speech are skipped (short samples give
unreliable vectors whatever the model).
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".hda", ".webm")


def _load_refs(data_dir: str) -> List[Tuple[str, str, List[Dict]]]:
    out = []
    for audio in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not audio.lower().endswith(AUDIO_EXT):
            continue
        stem = os.path.splitext(audio)[0]
        ref = f"{stem}.ref.json"
        if os.path.exists(ref):
            with open(ref, encoding="utf-8") as f:
                out.append((os.path.basename(stem), audio, json.load(f)))
    return out


def _resolve_model(model_id: str, hf_token: str | None) -> str:
    if model_id != "diarization":
        return model_id
    from pyannote.audio import Pipeline

    name = os.getenv("VOXHUB_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
    pipeline = Pipeline.from_pretrained(name, token=hf_token or os.getenv("HF_TOKEN"))
    emb = getattr(pipeline, "embedding", None)
    if not isinstance(emb, str):
        raise SystemExit(f"cannot read the embedding model id from {name} (got {emb!r})")
    print(f"[diarization] {name} clusters with {emb}")
    return emb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with audio files and <stem>.ref.json references")
    ap.add_argument("--model", action="append", help="Model id (repeatable); 'diarization' = the pipeline's own")
    ap.add_argument("--alias", action="append", default=[], help="Treat NAME=CANON as the same voice (repeatable)")
    ap.add_argument("--min-seconds", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=60.0, help="Audio per speaker (longest segments first)")
    ap.add_argument("--out", default=None, help="Write every distance as JSON here")
    args = ap.parse_args()

    from core.audio import load_audio  # noqa: E402  (VoxHub's loader: mono float32 16 kHz)
    from core.embeddings import EmbeddingBackend, _concat_speaker_audio  # noqa: E402

    models = args.model or [
        "pyannote/embedding",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "speechbrain/spkrec-ecapa-voxceleb",
    ]
    alias = dict(a.split("=", 1) for a in args.alias)
    hf_token = os.getenv("HF_TOKEN")

    refs = _load_refs(args.data)
    if not refs:
        print(f"No <stem>.ref.json next to an audio file in {args.data}", file=sys.stderr)
        return 1
    audios = {stem: load_audio(path) for stem, path, _ in refs}

    # (stem, name) -> waveform of that speaker's reference speech
    samples: Dict[Tuple[str, str], np.ndarray] = {}
    for stem, _path, segs in refs:
        by_name: Dict[str, List[Dict]] = {}
        for s in segs:
            name = alias.get(s["speaker"], s["speaker"])
            by_name.setdefault(name, []).append(s)
        for name, group in by_name.items():
            wav, total = _concat_speaker_audio(audios[stem], group, 16000, args.max_seconds)
            if total >= args.min_seconds:
                samples[(stem, name)] = wav
            else:
                print(f"[skip] {stem}/{name}: {total:.1f}s of reference speech")
    keys = sorted(samples)
    print(f"{len(keys)} (file, speaker) samples from {len(refs)} recordings\n")

    report = {}
    for model_id in models:
        try:
            resolved = _resolve_model(model_id, hf_token)
            t0 = time.time()
            backend = EmbeddingBackend(resolved, hf_token)
            load_s = time.time() - t0
        except Exception as e:  # noqa: BLE001
            print(f"== {model_id}: cannot load ({type(e).__name__}: {e})\n")
            continue
        t0 = time.time()
        vecs = {k: backend.embed(samples[k]) for k in keys}
        embed_s = time.time() - t0

        same, diff, pairs = [], [], []
        for a, b in itertools.combinations(keys, 2):
            d = float(1.0 - np.dot(vecs[a], vecs[b]))
            kind = "same" if a[1] == b[1] else "diff"
            (same if kind == "same" else diff).append(d)
            pairs.append((d, kind, a, b))
        pairs.sort()

        def stats(xs):
            return (f"n={len(xs):3d}  min {min(xs):.3f}  median {statistics.median(xs):.3f}  max {max(xs):.3f}"
                    if xs else "n=  0")

        margin = (min(diff) - max(same)) if same and diff else float("nan")
        print(f"== {resolved}  (dim {backend.dim}, load {load_s:.0f}s, {len(keys)} embeddings in {embed_s:.1f}s)")
        print(f"   same voice, other recording : {stats(same)}")
        print(f"   different voices            : {stats(diff)}")
        print(f"   margin (min diff - max same): {margin:+.3f}  "
              + ("→ separable, threshold between "
                 f"{max(same):.2f} and {min(diff):.2f}" if margin > 0 else "→ NOT separable"))
        # The pairs around the boundary are the ones to look at.
        worst_same = [p for p in pairs if p[1] == "same"][-3:]
        worst_diff = [p for p in pairs if p[1] == "diff"][:3]
        for d, kind, a, b in worst_same:
            print(f"   farthest same : {d:.3f}  {a[0]}/{a[1]}  ↔  {b[0]}/{b[1]}")
        for d, kind, a, b in worst_diff:
            print(f"   closest diff  : {d:.3f}  {a[0]}/{a[1]}  ↔  {b[0]}/{b[1]}")
        print()
        report[resolved] = {
            "dim": backend.dim, "same": same, "diff": diff, "margin": margin,
            "pairs": [{"d": round(d, 4), "kind": k, "a": list(a), "b": list(b)} for d, k, a, b in pairs],
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"details written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
