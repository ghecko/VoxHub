"""
Word-level forced alignment + speaker reconciliation (WhisperX-style).

Voxtral (open weights) and most other backends VoxHub serves return plain
text without word timestamps. To attach a speaker to every word we need
those timestamps, so this module:

    1. ``ForcedAligner``          — aligns the transcript of a chunk to the
                                    chunk audio with a CTC acoustic model
                                    (torchaudio MMS_FA by default, or any
                                    HF wav2vec2 CTC checkpoint) and returns
                                    ``{word, start, end, score}`` per word.
    2. ``assign_word_speakers``   — projects every word onto the pyannote
                                    speaker turns (max temporal overlap,
                                    with nearest-turn / neighbour fallback).
    3. ``words_to_segments``      — regroups labelled words into speaker
                                    turns for display (speaker change,
                                    long pause, hard duration ceiling).

Everything except ``ForcedAligner`` is pure Python and unit-testable
without torch.
"""

from __future__ import annotations

import bisect
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_PUNCT_STRIP = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?…]+[\"'»)]*$")

# Digraph / ligature fixes applied before NFKD stripping when the target
# vocabulary is bare ASCII letters (MMS_FA).
_LATIN_FIXES = str.maketrans({
    "œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss",
    "’": "'", "‘": "'", "ʼ": "'",
})


def _strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def split_words(text: str) -> List[str]:
    """Split a transcript into alignable words.

    French typographic spacing ("c'est ça ?", "oui !", "bon :") makes
    ``str.split`` emit punctuation-only tokens. Such a token has no acoustic
    content: aligned on its own it inherits interpolated timing and can be
    handed to the *next* speaker, so the "?" of one sentence opens the next
    person's segment. Glue it to the previous word instead (or to the next
    one when it opens the text) so it follows the word it belongs to.
    """
    out: List[str] = []
    pending: List[str] = []
    for tok in text.split():
        if not any(ch.isalnum() for ch in tok):
            if out:
                out[-1] = f"{out[-1]} {tok}"
            else:
                pending.append(tok)
            continue
        if pending:
            tok = " ".join(pending + [tok])
            pending = []
        out.append(tok)
    if pending:  # punctuation only
        out.append(" ".join(pending))
    return out


def normalize_word(word: str, vocab: Dict[str, int], lowercase: bool, ascii_only: bool) -> str:
    """Reduce a transcript word to the characters the CTC vocabulary knows.

    Leading/trailing punctuation is dropped, apostrophes inside a word are
    kept when the vocabulary has them, characters that still are not in
    the vocabulary are dropped after a diacritics-stripping fallback. The
    result may be empty (e.g. "2026", "…"): the caller then interpolates the
    word's timing from its neighbours.
    """
    w = _PUNCT_STRIP.sub("", word).translate(_LATIN_FIXES)
    w = w.lower() if lowercase else w.upper()
    if ascii_only:
        w = _strip_diacritics(w)
    out = []
    for ch in w:
        if ch in vocab:
            out.append(ch)
            continue
        alt = _strip_diacritics(ch)
        if alt and alt in vocab:
            out.append(alt)
    return "".join(out)


# ---------------------------------------------------------------------------
# Acoustic backends
# ---------------------------------------------------------------------------

class _TorchaudioBundleBackend:
    """torchaudio.pipelines bundle (MMS_FA, WAV2VEC2_ASR_BASE_960H, ...)."""

    def __init__(self, name: str, device):
        import torch
        import torchaudio

        bundle = getattr(torchaudio.pipelines, name)
        try:
            # MMS_FA accepts with_star; other bundles don't take kwargs.
            self.model = bundle.get_model(with_star=False)
            labels = list(bundle.get_labels(star=None))
        except TypeError:
            self.model = bundle.get_model()
            labels = list(bundle.get_labels())
        self.model = self.model.to(device).eval()
        self.device = device
        self.sample_rate = bundle.sample_rate
        self.blank_idx = 0
        self.vocab = {}
        self.sep_idx: Optional[int] = None
        for i, lab in enumerate(labels):
            if lab in ("-", "<blank>", "<pad>"):
                self.blank_idx = i
            elif lab == "|":
                self.sep_idx = i
            elif len(lab) == 1:
                self.vocab[lab] = i
        letters = [c for c in self.vocab if c.isalpha()]
        self.lowercase = bool(letters) and letters[0].islower()
        self.ascii_only = all(ord(c) < 128 for c in self.vocab)
        self._torch = torch

    def emissions(self, audio: np.ndarray, window_s: float = 60.0):
        torch = self._torch
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        win = int(window_s * self.sample_rate)
        outs = []
        with torch.inference_mode():
            for off in range(0, wav.shape[1], win):
                piece = wav[:, off:off + win]
                if piece.shape[1] < 400:  # below the model's receptive field
                    break
                em, _ = self.model(piece.to(self.device))
                outs.append(torch.log_softmax(em[0].float(), dim=-1).cpu())
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, len(self.vocab) + 2))


class _HFCTCBackend:
    """Any HuggingFace Wav2Vec2ForCTC checkpoint (per-language aligners)."""

    def __init__(self, model_id: str, device):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.sample_rate = 16000
        tok = self.processor.tokenizer
        raw_vocab = tok.get_vocab()
        self.blank_idx = tok.pad_token_id
        self.sep_idx = raw_vocab.get(getattr(tok, "word_delimiter_token", "|"))
        self.vocab = {k: v for k, v in raw_vocab.items() if len(k) == 1 and k != "|"}
        letters = [c for c in self.vocab if c.isalpha()]
        self.lowercase = bool(letters) and sum(c.islower() for c in letters) >= len(letters) / 2
        self.ascii_only = all(ord(c) < 128 for c in self.vocab)
        self._torch = torch

    def emissions(self, audio: np.ndarray, window_s: float = 60.0):
        torch = self._torch
        win = int(window_s * self.sample_rate)
        outs = []
        with torch.inference_mode():
            for off in range(0, len(audio), win):
                piece = audio[off:off + win]
                if len(piece) < 400:
                    break
                inputs = self.processor(
                    piece, sampling_rate=self.sample_rate, return_tensors="pt"
                ).input_values.to(self.device)
                logits = self.model(inputs).logits[0]
                outs.append(torch.log_softmax(logits.float(), dim=-1).cpu())
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, len(self.vocab) + 2))


# ---------------------------------------------------------------------------
# Forced aligner
# ---------------------------------------------------------------------------

class ForcedAligner:
    """Align transcript words to audio with a CTC model.

    Args:
        model:  ``"MMS_FA"`` (default, multilingual, ~1.2 GB), any other
                ``torchaudio.pipelines`` bundle name, or a HuggingFace
                ``Wav2Vec2ForCTC`` model id (contains a ``/``), e.g.
                ``jonatasgrosman/wav2vec2-large-xlsr-53-french`` which keeps
                French accents in its vocabulary.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
    """

    def __init__(self, model: str = "MMS_FA", device: str = "auto"):
        import torch

        dev_name = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_name)
        self.model_name = model
        if "/" in model:
            self.backend = _HFCTCBackend(model, self.device)
        else:
            self.backend = _TorchaudioBundleBackend(model, self.device)
        self._torch = torch
        logger.info(
            "[Aligner] Loaded %s on %s (vocab=%d chars, sep=%s, lowercase=%s)",
            model, dev_name, len(self.backend.vocab),
            self.backend.sep_idx is not None, self.backend.lowercase,
        )

    # ── public ──────────────────────────────────────────────────────────
    def align(self, audio: np.ndarray, text: str, offset: float = 0.0) -> List[Dict]:
        """Return ``[{word, start, end, score}]`` for *text* spoken in *audio*.

        Times are absolute when ``offset`` (chunk start in the full file) is
        given. Words the vocabulary cannot represent (numbers, symbols) get
        interpolated timings and ``score=0``. If the alignment itself is
        impossible (text longer than the audio can carry, empty emissions)
        the words are spread uniformly over the chunk and a warning logged.
        """
        words = split_words(text)
        if not words:
            return []
        duration = len(audio) / self.backend.sample_rate

        try:
            return self._align(audio, words, offset, duration)
        except Exception as e:  # never fail a transcription over alignment
            logger.warning("[Aligner] alignment failed (%s); using uniform timing", e)
            return uniform_word_times(words, offset, offset + duration)

    # ── internals ───────────────────────────────────────────────────────
    def _align(self, audio, words: List[str], offset: float, duration: float) -> List[Dict]:
        torch = self._torch
        be = self.backend

        tokens: List[int] = []
        spans: List[Optional[tuple]] = []   # (first_token_pos, last_token_pos) per word
        for wi, w in enumerate(words):
            norm = normalize_word(w, be.vocab, be.lowercase, be.ascii_only)
            if not norm:
                spans.append(None)
                continue
            if be.sep_idx is not None and tokens:
                tokens.append(be.sep_idx)
            first = len(tokens)
            tokens.extend(be.vocab[c] for c in norm)
            spans.append((first, len(tokens) - 1))

        if not tokens:
            return uniform_word_times(words, offset, offset + duration)

        emission = be.emissions(audio)              # [T, C] log-probs
        T = emission.shape[0]
        if T < len(tokens) + 2:
            raise ValueError(f"{len(tokens)} tokens for {T} frames")

        frame_dur = duration / T
        path = _viterbi_align(emission, tokens, be.blank_idx)   # per token: (start_frame, end_frame, score)

        out: List[Dict] = []
        for w, span in zip(words, spans):
            if span is None:
                out.append({"word": w, "start": None, "end": None, "score": 0.0})
                continue
            f0, f1, sc = path[span[0]][0], path[span[1]][1], float(
                np.mean([path[k][2] for k in range(span[0], span[1] + 1)])
            )
            out.append({
                "word": w,
                "start": round(offset + f0 * frame_dur, 3),
                "end": round(offset + (f1 + 1) * frame_dur, 3),
                "score": round(sc, 3),
            })
        _interpolate_missing(out, offset, offset + duration)
        return out


def _viterbi_align(emission, tokens: Sequence[int], blank: int) -> List[tuple]:
    """CTC forced alignment. Returns per-token ``(start_frame, end_frame, mean_prob)``.

    Uses ``torchaudio.functional.forced_align`` (C++/CUDA) when available and
    falls back to the classic Python trellis otherwise, so the aligner keeps
    working across torchaudio versions.
    """
    import torch

    try:
        import torchaudio.functional as F  # noqa: N812
        log_probs = emission.unsqueeze(0)
        targets = torch.tensor([tokens], dtype=torch.int32)
        alignments, scores = F.forced_align(log_probs, targets, blank=blank)
        ali = alignments[0].tolist()
        probs = scores[0].exp().tolist()
        return _merge_alignment(ali, probs, tokens, blank)
    except Exception as e:  # ImportError, AttributeError, RuntimeError on odd shapes
        logger.debug("[Aligner] torchaudio forced_align unavailable (%s); python trellis", e)

    return _python_trellis(emission, list(tokens), blank)


def _merge_alignment(ali: List[int], probs: List[float], tokens: Sequence[int], blank: int) -> List[tuple]:
    """Collapse a frame-level CTC path into one (start, end, score) per target token."""
    out: List[tuple] = []
    ti = 0
    t = 0
    T = len(ali)
    while t < T and ti < len(tokens):
        if ali[t] == blank:
            t += 1
            continue
        # a run of the same emitted token = one target token
        start = t
        ps = []
        while t < T and ali[t] == ali[start]:
            ps.append(probs[t])
            t += 1
        out.append((start, t - 1, float(np.mean(ps))))
        ti += 1
    while len(out) < len(tokens):  # degenerate: pad with last frame
        last = out[-1] if out else (0, 0, 0.0)
        out.append((last[1], last[1], 0.0))
    return out


def _python_trellis(emission, tokens: List[int], blank: int) -> List[tuple]:
    import torch

    T = emission.size(0)
    N = len(tokens)
    trellis = torch.zeros((T + 1, N + 1))
    trellis[1:, 0] = torch.cumsum(emission[:, blank], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-N:, 0] = float("inf")
    tok = torch.tensor(tokens)
    for t in range(T):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank],
            trellis[t, :-1] + emission[t, tok],
        )
    # backtrack
    j = N
    t = int(torch.argmax(trellis[:, j]).item())
    path = []  # (token_index, frame, prob)
    while j > 0:
        assert t > 0
        p_stay = trellis[t - 1, j] + emission[t - 1, blank]
        p_change = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if p_change > p_stay else blank].exp().item()
        t -= 1
        if p_change > p_stay:
            path.append((j - 1, t, prob))
            j -= 1
    path.reverse()
    out: List[tuple] = []
    for idx in range(N):
        frames = [(f, p) for (k, f, p) in path if k == idx]
        if frames:
            out.append((frames[0][0], frames[-1][0], float(np.mean([p for _, p in frames]))))
        else:
            prev_end = out[-1][1] if out else 0
            out.append((prev_end, prev_end, 0.0))
    return out


def uniform_word_times(words: List[str], start: float, end: float) -> List[Dict]:
    """Spread words evenly over [start, end] (alignment fallback)."""
    n = max(len(words), 1)
    step = (end - start) / n
    return [
        {
            "word": w,
            "start": round(start + i * step, 3),
            "end": round(start + (i + 1) * step, 3),
            "score": 0.0,
        }
        for i, w in enumerate(words)
    ]


def _interpolate_missing(words: List[Dict], lo: float, hi: float) -> None:
    """Fill ``start``/``end`` of unalignable words from their neighbours (in place)."""
    n = len(words)
    i = 0
    while i < n:
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < n and words[j]["start"] is None:
            j += 1
        left = words[i - 1]["end"] if i > 0 else lo
        right = words[j]["start"] if j < n else hi
        if right < left:
            right = left
        step = (right - left) / (j - i)
        for k in range(i, j):
            words[k]["start"] = round(left + (k - i) * step, 3)
            words[k]["end"] = round(left + (k - i + 1) * step, 3)
        i = j


# ---------------------------------------------------------------------------
# Speaker reconciliation
# ---------------------------------------------------------------------------

def assign_word_speakers(
    words: List[Dict],
    turns: List[Dict],
    max_gap: float = 1.0,
    default_speaker: str = "SPEAKER_00",
    min_overlap: float = 0.5,
    ambiguity: float = 0.5,
    continuity_window: float = 2.0,
) -> List[Dict]:
    """Attach a ``speaker`` to every word from diarization turns.

    For each word the speaker with the largest temporal overlap wins
    (overlaps are summed over that speaker's turns, so a turn pyannote split
    in two still counts once). A word is *confident* when its best speaker
    covers at least ``min_overlap`` of its duration and no other speaker
    covers ``ambiguity`` or more of it. The two other cases are where the
    max-overlap rule is a coin flip and cost most speaker errors on real
    meetings:

    * the word straddles a turn boundary (best overlap below ``min_overlap``):
      pyannote boundaries jitter by a few hundred ms, and the CTC aligner
      packs the words of an interjection right up to them;
    * the word sits inside *overlapping* turns (two speakers each cover
      ``ambiguity`` of it): Voxtral transcribes one stream, the one that was
      already speaking.

    Such words take the speaker of the nearest confident word within
    ``continuity_window`` seconds (previous first, then next), provided that
    speaker is one of the word's candidates, i.e. overlaps it or is the
    nearest turn within ``max_gap``. Otherwise they keep the max-overlap
    speaker. A word overlapping no turn takes the nearest turn if it is
    within ``max_gap`` seconds; failing that it inherits the previous word's
    speaker (or the next word's at the very beginning). With no turns at all
    every word gets ``default_speaker``.
    """
    if not words:
        return []
    if not turns:
        return [{**w, "speaker": default_speaker} for w in words]

    turns = sorted(
        ({"start": float(t["start"]), "end": float(t["end"]), "speaker": t.get("speaker", default_speaker)}
         for t in turns if t["end"] > t["start"]),
        key=lambda t: t["start"],
    )
    starts = [t["start"] for t in turns]
    max_len = max(t["end"] - t["start"] for t in turns)

    # Pass 1: per-word candidates and a confident label when the overlap is
    # unambiguous. ``pick`` is the max-overlap (or nearest-turn) fallback.
    info: List[Dict] = []
    for w in words:
        ws, we = w["start"], w["end"]
        dur = max(we - ws, 1e-3)
        # Candidate turns: any that starts before the word ends and could
        # still overlap it (start >= ws - max_len).
        lo = bisect.bisect_left(starts, ws - max_len - max_gap)
        hi = bisect.bisect_right(starts, we + max_gap)
        overlap: Dict[str, float] = {}
        near, near_d = None, float("inf")
        for t in turns[lo:hi]:
            ov = min(we, t["end"]) - max(ws, t["start"])
            if ov > 0:
                overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + ov
            d = max(t["start"] - we, ws - t["end"], 0.0)
            if d < near_d:
                near, near_d = t, d
        candidates = set(overlap)
        if near is not None and near_d <= max_gap:
            candidates.add(near["speaker"])
        if overlap:
            ranked = sorted(overlap.items(), key=lambda kv: -kv[1])
            pick = ranked[0][0]
            r1 = ranked[0][1] / dur
            r2 = ranked[1][1] / dur if len(ranked) > 1 else 0.0
            confident = r1 >= min_overlap and r2 < ambiguity
        elif near is not None and near_d <= max_gap:
            pick, confident = near["speaker"], False
        else:
            pick, confident = None, False
        info.append({"pick": pick, "confident": confident, "candidates": candidates})

    # Pass 2: resolve the ambiguous words by continuity. Left to right, so a
    # run of ambiguous words (an interjection in an overlap) stays with the
    # speaker it was attached to, instead of flipping word by word.
    labels: List[Optional[str]] = [i["pick"] if i["confident"] else None for i in info]
    resolved: List[bool] = [i["confident"] for i in info]
    for k, (w, i) in enumerate(zip(words, info)):
        if i["confident"] or not i["candidates"]:
            continue
        chosen = None
        # Nearest resolved word before, then nearest confident word after,
        # each within the window; a neighbour whose speaker is not a
        # candidate ends the search on that side.
        for j in range(k - 1, -1, -1):
            if w["start"] - words[j]["end"] > continuity_window:
                break
            if resolved[j]:
                if labels[j] in i["candidates"]:
                    chosen = labels[j]
                break
        if chosen is None:
            for j in range(k + 1, len(words)):
                if words[j]["start"] - w["end"] > continuity_window:
                    break
                if info[j]["confident"]:
                    if info[j]["pick"] in i["candidates"]:
                        chosen = info[j]["pick"]
                    break
        labels[k] = chosen if chosen is not None else i["pick"]
        resolved[k] = True

    out = [{**w, "speaker": spk} for w, spk in zip(words, labels)]

    # Fill the holes (no turn within max_gap) from neighbours.
    last = None
    for w in out:
        if w["speaker"] is None:
            w["speaker"] = last
        else:
            last = w["speaker"]
    nxt = None
    for w in reversed(out):
        if w["speaker"] is None:
            w["speaker"] = nxt if nxt is not None else default_speaker
        else:
            nxt = w["speaker"]
    return out


def words_to_segments(
    words: List[Dict],
    max_pause: float = 1.0,
    max_duration: float = 30.0,
    soft_duration: float = 15.0,
) -> List[Dict]:
    """Regroup speaker-labelled words into display segments.

    A new segment starts when the speaker changes, when the silence between
    two words exceeds ``max_pause``, when the running segment exceeds
    ``soft_duration`` and the previous word ends a sentence, or
    unconditionally past ``max_duration``. Each segment carries its words
    and a ``confidence`` (mean alignment score of its aligned words).
    """
    segments: List[Dict] = []
    cur: Optional[Dict] = None

    def flush():
        nonlocal cur
        if cur and cur["words"]:
            scores = [w["score"] for w in cur["words"] if w.get("score")]
            cur["text"] = " ".join(w["word"] for w in cur["words"])
            cur["confidence"] = round(float(np.mean(scores)), 3) if scores else 0.0
            cur["id"] = len(segments)
            segments.append(cur)
        cur = None

    for w in words:
        if cur is not None:
            pause = w["start"] - cur["end"]
            dur = w["end"] - cur["start"]
            prev_word = cur["words"][-1]["word"]
            if (
                w["speaker"] != cur["speaker"]
                or pause > max_pause
                or dur > max_duration
                or (dur > soft_duration and _SENTENCE_END.search(prev_word))
            ):
                flush()
        if cur is None:
            cur = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "words": []}
        cur["end"] = max(cur["end"], w["end"])
        cur["words"].append({
            "word": w["word"], "start": w["start"], "end": w["end"],
            "score": w.get("score", 0.0), "speaker": w["speaker"],
        })
    flush()
    return segments
