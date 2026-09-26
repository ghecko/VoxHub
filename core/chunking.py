"""
Acoustic chunking for the word-alignment pipeline.

The legacy pipeline cut the audio on *speaker turns* and transcribed each
turn on its own. That starves the ASR model of context (a 0.4 s "oui" has no
chance against the LM prior) and couples transcription quality to
diarization quality.

This module cuts the audio on *silences* instead, producing chunks that are
long enough for the ASR model to have context (target ~60 s) but short
enough to keep a hallucination loop, a vLLM timeout or a progress stall
local to a small window (max ~3 min). Speaker labels are attached later,
word by word, by ``core.align``.

Input is a list of speech regions as returned by Silero
(``[{"start": s, "end": e}, ...]`` in seconds, sorted, non-overlapping).
"""

from __future__ import annotations

import math
from typing import Dict, List


def build_chunks(
    speech_regions: List[Dict],
    total_duration: float,
    target_duration: float = 60.0,
    max_duration: float = 180.0,
    min_duration: float = 8.0,
    pad: float = 0.25,
    edge_pad: float = 5.0,
) -> List[Dict]:
    """Group speech regions into transcription chunks separated by silences.

    Strategy: walk the speech regions in order, accumulating them into the
    current chunk. Every gap between two regions is a candidate cut point.
    Once the chunk would exceed ``target_duration`` we cut at the *longest*
    silence seen so far (ties resolved towards the latest gap), provided the
    chunk before that gap is at least ``min_duration`` long. If a single
    speech region is longer than ``max_duration`` (no silence at all, e.g.
    a lecture with music bed) it is split into equal parts.

    Every chunk is then padded by ``pad`` seconds into the surrounding
    silence (never into a neighbouring chunk) so CTC alignment has a little
    leading/trailing context and the first phoneme is never clipped. The
    first chunk is additionally pulled back to 0 (and the last one pushed to
    ``total_duration``) when it begins (ends) within ``edge_pad`` seconds of
    the file edge: Silero regularly misses a short first utterance ("Très
    bien, ..." before the first pause), and a few seconds of extra audio at
    the edges cost nothing while a dropped first word is unrecoverable.

    Returns ``[{"start": float, "end": float, "index": int}, ...]`` sorted by
    start. Returns an empty list when there is no speech.
    """
    regions = [
        {"start": float(r["start"]), "end": float(r["end"])}
        for r in speech_regions
        if r["end"] > r["start"]
    ]
    regions.sort(key=lambda r: r["start"])
    if not regions:
        return []

    # Pre-split pathological regions longer than max_duration so the greedy
    # walk below never sees a region it cannot place.
    regions = _split_long_regions(regions, max_duration)

    chunks: List[Dict] = []
    i = 0
    n = len(regions)
    while i < n:
        start = regions[i]["start"]
        best_cut = None      # index of the last region of the chunk
        best_score = -1.0
        j = i
        while True:
            span = regions[j]["end"] - start
            if j + 1 < n and span >= min_duration:
                gap = regions[j + 1]["start"] - regions[j]["end"]
                # Long silences are natural cut points, but a long silence
                # 9 s into a 60 s target would leave a stubby chunk: weight
                # the gap by how close we are to the target length.
                score = gap * (0.5 + 0.5 * min(span / target_duration, 1.0))
                if score >= best_score:
                    best_cut, best_score = j, score
            if j == n - 1 or span >= target_duration:
                break
            if regions[j + 1]["end"] - start > max_duration:
                break  # adding the next region would blow the hard ceiling
            j += 1

        if j == n - 1 and regions[j]["end"] - start <= max_duration:
            # Tail of the file fits in one chunk: keep it whole rather than
            # leaving a tiny orphan chunk at the very end.
            cut = j
        elif best_cut is not None:
            cut = best_cut
        else:
            cut = j  # never reached min_duration before the ceiling: cut here
        chunks.append({"start": start, "end": regions[cut]["end"]})
        i = cut + 1

    # Merge a trailing micro-chunk into its predecessor when that keeps the
    # predecessor under max_duration.
    merged: List[Dict] = []
    for c in chunks:
        if (
            merged
            and (c["end"] - c["start"]) < min_duration
            and (c["end"] - merged[-1]["start"]) <= max_duration
        ):
            merged[-1]["end"] = c["end"]
        else:
            merged.append(dict(c))
    chunks = merged

    # Pad into silence without overlapping neighbours.
    for k, c in enumerate(chunks):
        lo = chunks[k - 1]["end"] if k > 0 else 0.0
        hi = chunks[k + 1]["start"] if k + 1 < len(chunks) else total_duration
        c["start"] = round(max(lo, c["start"] - pad, 0.0), 3)
        c["end"] = round(min(hi, c["end"] + pad, total_duration), 3)
        c["index"] = k
    if chunks and chunks[0]["start"] <= edge_pad:
        chunks[0]["start"] = 0.0
    if chunks and total_duration - chunks[-1]["end"] <= edge_pad:
        chunks[-1]["end"] = round(total_duration, 3)

    return chunks


def _split_long_regions(regions: List[Dict], max_duration: float) -> List[Dict]:
    out: List[Dict] = []
    for r in regions:
        dur = r["end"] - r["start"]
        if dur <= max_duration:
            out.append(r)
            continue
        parts = int(math.ceil(dur / max_duration))
        step = dur / parts
        for p in range(parts):
            out.append({
                "start": r["start"] + p * step,
                "end": r["start"] + (p + 1) * step if p < parts - 1 else r["end"],
            })
    return out


def split_chunk(chunk: Dict, speech_regions: List[Dict], min_part: float = 4.0) -> List[Dict]:
    """Split one chunk in two at the longest internal silence.

    Used when the ASR backend returned nothing (or a repetition-loop
    artefact) for a chunk that is clearly too long to be silence: instead of
    dropping minutes of speech we retry on two halves. Falls back to a plain
    midpoint split when no silence is known inside the chunk. Returns the
    original chunk unchanged (as a 1-element list) when it is too short to
    split.
    """
    start, end = chunk["start"], chunk["end"]
    if end - start < 2 * min_part:
        return [dict(chunk)]

    inner = [
        r for r in speech_regions
        if r["start"] >= start - 1e-3 and r["end"] <= end + 1e-3
    ]
    inner.sort(key=lambda r: r["start"])
    best_pos = None
    best_gap = 0.0
    for a, b in zip(inner, inner[1:]):
        gap = b["start"] - a["end"]
        mid = (a["end"] + b["start"]) / 2
        if gap > best_gap and (mid - start) >= min_part and (end - mid) >= min_part:
            best_gap, best_pos = gap, mid
    if best_pos is None:
        best_pos = (start + end) / 2

    first = {**chunk, "start": start, "end": round(best_pos, 3)}
    second = {**chunk, "start": round(best_pos, 3), "end": end}
    return [first, second]
