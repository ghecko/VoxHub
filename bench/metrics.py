"""
Transcription + diarization metrics, dependency-free.

    wer(ref_text, hyp_text)                  word error rate
    cpwer(ref_segments, hyp_segments)        concatenated-minimum-permutation WER
                                             (speaker-attributed transcription quality)
    der(ref_segments, hyp_segments)          diarization error rate (frame based,
                                             optimal speaker mapping, with collar)

Segments are ``{"start": s, "end": e, "speaker": str, "text": str}``; the
``text`` key is not needed for DER, ``start``/``end`` are not needed for
cpWER. When ``jiwer``, ``meeteval`` or ``pyannote.metrics`` are installed
``run_bench.py`` prefers them; these implementations exist so the harness
runs anywhere (including the API container) and give the same numbers on
the usual cases.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_APOS = str.maketrans({"’": "'", "‘": "'"})


def normalize_text(text: str, strip_accents: bool = False) -> List[str]:
    """Lowercase, unify apostrophes, drop punctuation, split on whitespace."""
    t = text.translate(_APOS).lower()
    if strip_accents:
        t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    t = _PUNCT.sub(" ", t)
    return t.split()


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> Tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) of the optimal alignment a→b."""
    n, m = len(a), len(b)
    # dp[i][j] = (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cand = prev[j - 1]
            else:
                s = prev[j - 1]
                cand = (s[0] + 1, s[1] + 1, s[2], s[3])
            d = prev[j]
            dcand = (d[0] + 1, d[1], d[2] + 1, d[3])
            ins = cur[j - 1]
            icand = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
            best = cand
            if dcand[0] < best[0]:
                best = dcand
            if icand[0] < best[0]:
                best = icand
            cur.append(best)
        prev = cur
    _, S, D, I = prev[m]
    return S, D, I


def wer(ref_text: str, hyp_text: str, strip_accents: bool = False) -> Dict[str, float]:
    ref = normalize_text(ref_text, strip_accents)
    hyp = normalize_text(hyp_text, strip_accents)
    if not ref:
        return {"wer": 0.0 if not hyp else 1.0, "S": 0, "D": 0, "I": len(hyp), "N": 0}
    S, D, I = _edit_distance(ref, hyp)
    return {"wer": (S + D + I) / len(ref), "S": S, "D": D, "I": I, "N": len(ref)}


def _by_speaker(segments: Iterable[Dict]) -> Dict[str, str]:
    out: Dict[str, List[str]] = {}
    for s in sorted(segments, key=lambda x: (x.get("start") or 0.0)):
        out.setdefault(str(s.get("speaker", "SPEAKER_00")), []).append(s.get("text", ""))
    return {k: " ".join(v) for k, v in out.items()}


def cpwer(ref_segments: List[Dict], hyp_segments: List[Dict], strip_accents: bool = False,
          max_speakers: int = 8) -> Dict[str, float]:
    """Concatenated minimum-permutation WER.

    Per speaker, concatenate the reference and hypothesis text; find the
    speaker mapping that minimises the total edit cost; report the summed
    errors over the total reference word count. Unmatched hypothesis
    speakers count fully as insertions, unmatched reference speakers as
    deletions. Brute-forces permutations, fine up to ~8 speakers.
    """
    ref = {k: normalize_text(v, strip_accents) for k, v in _by_speaker(ref_segments).items()}
    hyp = {k: normalize_text(v, strip_accents) for k, v in _by_speaker(hyp_segments).items()}
    N = sum(len(v) for v in ref.values())
    if not ref:
        total_hyp = sum(len(v) for v in hyp.values())
        return {"cpwer": 0.0 if total_hyp == 0 else 1.0, "errors": total_hyp, "N": 0, "mapping": {}}

    ref_keys, hyp_keys = list(ref), list(hyp)
    if len(ref_keys) > max_speakers or len(hyp_keys) > max_speakers:
        raise ValueError("too many speakers for brute-force cpWER")

    # cost matrix
    cost = {(r, h): sum(_edit_distance(ref[r], hyp[h])) for r in ref_keys for h in hyp_keys}
    best_total, best_map = None, {}
    longer, shorter, ref_is_longer = (ref_keys, hyp_keys, True) if len(ref_keys) >= len(hyp_keys) else (hyp_keys, ref_keys, False)
    for perm in itertools.permutations(longer, len(shorter)):
        total = 0
        mapping = {}
        matched_long = set(perm)
        for s_key, l_key in zip(shorter, perm):
            r, h = (l_key, s_key) if ref_is_longer else (s_key, l_key)
            total += cost[(r, h)]
            mapping[h] = r
        for l_key in longer:
            if l_key not in matched_long:
                total += len(ref[l_key]) if ref_is_longer else len(hyp[l_key])
        if best_total is None or total < best_total:
            best_total, best_map = total, mapping
    return {"cpwer": best_total / N, "errors": best_total, "N": N, "mapping": best_map}


def der(ref_segments: List[Dict], hyp_segments: List[Dict], collar: float = 0.25,
        frame: float = 0.01, skip_overlap: bool = False) -> Dict[str, float]:
    """Frame-based diarization error rate with optimal 1:1 speaker mapping.

    DER = (missed + false alarm + confusion) / total reference speech, where
    frames within ``collar`` seconds of a reference boundary are ignored (the
    NIST convention, 0.25 s collar on each side). Overlapping reference speech
    is scored unless ``skip_overlap``.
    """
    end = max([s["end"] for s in ref_segments + hyp_segments] + [0.0])
    n = int(end / frame) + 1

    def frames(segs):
        table: Dict[str, bytearray] = {}
        for s in segs:
            spk = str(s.get("speaker", "SPEAKER_00"))
            arr = table.setdefault(spk, bytearray(n))
            a, b = int(round(s["start"] / frame)), int(round(s["end"] / frame))
            for i in range(max(a, 0), min(b, n)):
                arr[i] = 1
        return table

    ref_t, hyp_t = frames(ref_segments), frames(hyp_segments)

    # collar mask: frames near reference boundaries are excluded
    mask = bytearray(b"\x01") * n
    c = int(round(collar / frame))
    for s in ref_segments:
        for edge in (s["start"], s["end"]):
            e = int(round(edge / frame))
            for i in range(max(e - c, 0), min(e + c, n)):
                mask[i] = 0
    ref_count = [sum(t[i] for t in ref_t.values()) for i in range(n)]
    if skip_overlap:
        for i in range(n):
            if ref_count[i] > 1:
                mask[i] = 0

    ref_keys, hyp_keys = list(ref_t), list(hyp_t)
    if len(ref_keys) > 8 or len(hyp_keys) > 8:
        raise ValueError("too many speakers for brute-force DER mapping")

    # overlap matrix between every ref/hyp speaker pair (masked frames)
    overlap = {
        (r, h): sum(1 for i in range(n) if mask[i] and ref_t[r][i] and hyp_t[h][i])
        for r in ref_keys for h in hyp_keys
    }
    longer, shorter, ref_is_longer = (ref_keys, hyp_keys, True) if len(ref_keys) >= len(hyp_keys) else (hyp_keys, ref_keys, False)
    best_hit, best_map = -1, {}
    for perm in itertools.permutations(longer, len(shorter)):
        hit = 0
        mapping = {}
        for s_key, l_key in zip(shorter, perm):
            r, h = (l_key, s_key) if ref_is_longer else (s_key, l_key)
            hit += overlap[(r, h)]
            mapping[h] = r
        if hit > best_hit:
            best_hit, best_map = hit, mapping

    total = miss = fa = conf = 0
    for i in range(n):
        if not mask[i]:
            continue
        r_active = {r for r in ref_keys if ref_t[r][i]}
        h_active = {best_map.get(h, f"__unmapped_{h}") for h in hyp_keys if hyp_t[h][i]}
        nr, nh = len(r_active), len(h_active)
        total += nr
        correct = len(r_active & h_active)
        if nr > nh:
            miss += nr - nh
        elif nh > nr:
            fa += nh - nr
        conf += min(nr, nh) - correct
    if total == 0:
        return {"der": 0.0 if fa == 0 else 1.0, "miss": 0.0, "false_alarm": fa * frame, "confusion": 0.0, "total": 0.0, "mapping": best_map}
    return {
        "der": (miss + fa + conf) / total,
        "miss": miss / total,
        "false_alarm": fa / total,
        "confusion": conf / total,
        "total": total * frame,
        "mapping": best_map,
    }


def read_rttm(path: str) -> List[Dict]:
    """Parse an RTTM file into segments (SPEAKER lines only)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start, dur = float(parts[3]), float(parts[4])
            out.append({"start": start, "end": start + dur, "speaker": parts[7]})
    return out


def write_rttm(segments: List[Dict], path: str, uri: str = "file") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(
                f"SPEAKER {uri} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} <NA> <NA> "
                f"{s.get('speaker', 'SPEAKER_00')} <NA> <NA>\n"
            )
