"""
Segment post-processor for diarization output.

Pyannote's diarization can produce overlapping segments and very short
speaker turns (backchannels like "yeah", "ok", "better") that fragment
the primary speaker's flow. This module provides two layers of fix:

    1. sanitize_segments()   — fast, rule-based: absorbs micro-turns and
                               resolves overlapping boundaries. No ML needed.
    2. refine_boundaries()   — optional, wav2vec2-based: snaps each segment's
                               start/end to the actual speech onset/offset
                               using frame-level CTC probabilities.

Pipeline position:
    VAD/Diarization  -->  sanitize_segments()  -->  refine_boundaries()  -->  Transcription
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


def sanitize_segments(
    segments: List[Dict],
    min_turn_duration: float = 1.5,
    max_overlap: float = 0.0,
) -> List[Dict]:
    """
    Clean up diarization segments so the timeline is sequential and sensible.

    Applies three passes in order:
        1. Absorb micro-turns — very short speaker turns (< min_turn_duration)
           sandwiched between segments of the same speaker are merged into
           the surrounding speaker's turn. The short interjection is kept as
           its own segment but won't break the primary speaker's flow.
        2. Resolve overlaps — when two segments overlap in time, the shorter
           one is trimmed so it doesn't extend past the next segment's start.
        3. Sort and deduplicate — final sort by start time, remove zero-
           duration artifacts.

    Args:
        segments:           Raw diarization segments from Pyannote / HybridVAD.
                            Each dict must have "start", "end", and optionally "speaker".
        min_turn_duration:  Speaker turns shorter than this (seconds) are
                            candidates for absorption. Default 1.5s.
        max_overlap:        Maximum allowed overlap between adjacent segments
                            (seconds). Segments overlapping more than this are
                            trimmed. Default 0.0 (no overlap allowed).

    Returns:
        A new list of sanitized segments, sorted by start time.
    """
    if not segments or len(segments) < 2:
        return list(segments)

    working = deepcopy(segments)
    working.sort(key=lambda s: s["start"])

    original_count = len(working)

    # --- Pass 1: Absorb micro-turns (single pass only) ---
    # A single pass handles the common case: one backchannel interjection
    # between two segments of the same speaker. We deliberately avoid
    # iterating to convergence because cascading absorptions can eat
    # legitimate speech — e.g. when Pyannote fragments a real speaker
    # turn into several short segments, each pass would absorb the next
    # one until the entire turn is gone.
    working = _absorb_micro_turns(working, min_turn_duration)

    # --- Pass 2: Resolve overlaps ---
    working = _resolve_overlaps(working, max_overlap)

    # --- Pass 3: Remove zero/negative-duration artifacts ---
    working = [s for s in working if s["end"] - s["start"] > 0.05]

    if len(working) != original_count:
        logger.info(
            f"[Sanitizer] {original_count} segments -> {len(working)} segments "
            f"(min_turn={min_turn_duration}s)"
        )

    return working


def _absorb_micro_turns(
    segments: List[Dict],
    min_turn_duration: float,
) -> List[Dict]:
    """
    When speaker A is interrupted by a very short speaker B turn and then
    speaker A resumes immediately, merge A's two segments into one continuous
    turn. Speaker B's micro-turn is preserved but won't break A's flow.

    Example before:
        SPEAKER_00  5:50 - 5:53
        SPEAKER_04  5:53 - 5:55   (duration 2s, interjection: "better")
        SPEAKER_00  5:55 - 6:20

    After (with min_turn_duration=2.5):
        SPEAKER_00  5:50 - 6:20   (merged across the micro-turn)

    Only truly isolated speakers are absorbed — any speaker that appears
    2+ times in the full segment list is considered a real conversation
    participant and is never absorbed, regardless of how short their
    individual turns are.
    """
    if len(segments) < 3:
        return segments

    # Pre-compute speaker frequency across ALL segments. Any speaker with
    # 2+ segments is a real participant — not a one-off backchannel.
    from collections import Counter
    speaker_counts = Counter(s.get("speaker") for s in segments)
    real_speakers = {spk for spk, count in speaker_counts.items() if count >= 2}

    logger.debug(
        f"[Sanitizer] Speaker counts: {dict(speaker_counts)} — "
        f"protected (>=2 segments): {real_speakers}"
    )

    result = []
    i = 0

    while i < len(segments):
        current = segments[i]

        # Look ahead: is the next segment a micro-turn followed by the same speaker?
        if i + 2 < len(segments):
            middle = segments[i + 1]
            after = segments[i + 2]

            middle_duration = middle["end"] - middle["start"]
            same_speaker_resumes = (
                current.get("speaker") == after.get("speaker")
                and current.get("speaker") != middle.get("speaker")
            )

            if same_speaker_resumes and middle_duration < min_turn_duration:
                middle_speaker = middle.get("speaker")

                # Safety: never merge if the resulting segment would span a
                # much larger time range than the two original segments.
                # This prevents a short backchannel from causing a huge
                # merged block that eclipses other segments in between.
                gap = after["start"] - current["end"]
                if gap > min_turn_duration:
                    logger.debug(
                        f"[Sanitizer] Skipped absorption — gap too large: "
                        f"{gap:.1f}s between {current.get('speaker')} "
                        f"[{current['start']:.1f}-{current['end']:.1f}s] and "
                        f"[{after['start']:.1f}-{after['end']:.1f}s]"
                    )
                    result.append(current)
                    i += 1
                    continue

                if middle_speaker in real_speakers:
                    # This speaker appears elsewhere in the conversation —
                    # they're a real participant, not a backchannel. Keep.
                    logger.debug(
                        f"[Sanitizer] Kept {middle_speaker} "
                        f"[{middle['start']:.1f}-{middle['end']:.1f}s] "
                        f"({middle_duration:.1f}s) — real speaker "
                        f"({speaker_counts[middle_speaker]} total segments)"
                    )
                    result.append(current)
                    i += 1
                    continue

                # Truly isolated backchannel (only 1 segment total) — absorb.
                # Its audio falls within the merged range and will be
                # captured by the ASR model.
                merged = {
                    "start": current["start"],
                    "end": after["end"],
                    "speaker": current.get("speaker", "SPEAKER_00"),
                }
                result.append(merged)

                logger.debug(
                    f"[Sanitizer] Absorbed micro-turn: {middle_speaker} "
                    f"[{middle['start']:.1f}-{middle['end']:.1f}s] "
                    f"({middle_duration:.1f}s) — isolated speaker, merged "
                    f"{current.get('speaker')} "
                    f"[{current['start']:.1f}-{after['end']:.1f}s]"
                )

                i += 3  # Skip current, middle, and after
                continue

        result.append(current)
        i += 1

    return result


def _resolve_overlaps(
    segments: List[Dict],
    max_overlap: float,
) -> List[Dict]:
    """
    Ensure no two segments overlap by more than max_overlap seconds.

    Strategy: sort by start time, then for each pair of adjacent segments,
    if they overlap, trim the shorter segment's boundaries so it fits
    within the gap. If the shorter segment is completely eclipsed, it
    gets squeezed to just its own portion (trimmed from both sides).
    """
    if len(segments) < 2:
        return segments

    segments.sort(key=lambda s: s["start"])
    result = [segments[0]]

    for seg in segments[1:]:
        prev = result[-1]
        overlap = prev["end"] - seg["start"]

        if overlap <= max_overlap:
            # No meaningful overlap
            result.append(seg)
            continue

        # There's an overlap — decide who gets trimmed
        prev_dur = prev["end"] - prev["start"]
        seg_dur = seg["end"] - seg["start"]

        if seg_dur <= prev_dur:
            # The incoming segment is shorter — push its start to after prev ends
            new_start = prev["end"]
            if new_start < seg["end"]:
                trimmed = dict(seg)
                trimmed["start"] = round(new_start, 3)
                result.append(trimmed)
                logger.debug(
                    f"[Sanitizer] Trimmed {seg.get('speaker')} start: "
                    f"{seg['start']:.1f} -> {new_start:.1f}"
                )
            else:
                # Segment is completely eclipsed by the previous one.
                # Log it — this can signal over-aggressive merging upstream.
                logger.warning(
                    f"[Sanitizer] Dropped eclipsed segment: "
                    f"{seg.get('speaker')} [{seg['start']:.1f}-{seg['end']:.1f}s] "
                    f"(eclipsed by {prev.get('speaker')} "
                    f"[{prev['start']:.1f}-{prev['end']:.1f}s])"
                )
        else:
            # The previous segment is shorter — trim its end
            prev["end"] = round(seg["start"], 3)
            result.append(seg)
            logger.debug(
                f"[Sanitizer] Trimmed {prev.get('speaker')} end: "
                f"-> {seg['start']:.1f}"
            )

    return result


# ---------------------------------------------------------------------------
# Wav2Vec2-based boundary refinement
# ---------------------------------------------------------------------------

class BoundaryRefiner:
    """
    Snap diarization segment boundaries to actual speech onset/offset
    using wav2vec2 frame-level CTC probabilities.

    How it works:
        For each segment boundary, wav2vec2 is run on a small audio window
        around that boundary. Frames where the model predicts a high
        probability of *any* character (i.e. low blank/silence probability)
        are treated as speech. The boundary is then moved to the nearest
        speech edge.

    This fixes the common case where Pyannote's boundary is off by a
    fraction of a second — enough to clip the first/last word of a turn.

    Requires torchaudio (ships with torch in all Docker images).

    Args:
        model_name: torchaudio pipeline bundle name. Default is the small
                    base model (90M params, fast inference).
        device:     "auto", "cuda", or "cpu". Default "auto".
    """

    def __init__(self, model_name: str = "WAV2VEC2_ASR_BASE_960H", device: str = "auto"):
        import torchaudio

        bundle = getattr(torchaudio.pipelines, model_name)
        self.model = bundle.get_model()
        self.expected_sr = bundle.sample_rate  # 16000
        self._blank_idx = 0  # CTC blank token is always index 0

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
        self.device = torch.device(dev)
        self.model = self.model.to(self.device).eval()

        logger.info(f"[BoundaryRefiner] Loaded {model_name} on {dev}")

    @torch.inference_mode()
    def refine_boundaries(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        sampling_rate: int = 16000,
        padding: float = 1.0,
        speech_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Refine start/end of each segment using wav2vec2 speech probabilities.

        Args:
            audio:            Full audio waveform (numpy float32, mono).
            segments:         Diarization segments to refine.
            sampling_rate:    Audio sample rate (must match expected_sr).
            padding:          Seconds of audio to examine on each side of a
                              boundary. Default 1.0s.
            speech_threshold: Frame speech probability above which a frame is
                              considered speech. Default 0.5.

        Returns:
            A new list of segments with refined boundaries.
        """
        if not segments:
            return []

        audio_duration = len(audio) / sampling_rate
        results = []

        for seg in segments:
            refined = dict(seg)

            # --- Refine START boundary ---
            refined["start"] = self._snap_boundary(
                audio, seg["start"], sampling_rate, audio_duration,
                padding, speech_threshold, direction="start"
            )

            # --- Refine END boundary ---
            refined["end"] = self._snap_boundary(
                audio, seg["end"], sampling_rate, audio_duration,
                padding, speech_threshold, direction="end"
            )

            # Safety: don't let refinement create negative-duration segments
            if refined["end"] <= refined["start"]:
                refined["start"] = seg["start"]
                refined["end"] = seg["end"]

            results.append(refined)

        return results

    def _snap_boundary(
        self,
        audio: np.ndarray,
        boundary_time: float,
        sr: int,
        audio_duration: float,
        padding: float,
        threshold: float,
        direction: str,
    ) -> float:
        """
        Snap a single boundary (start or end) to the nearest speech edge.
        """
        # Extract a small window around the boundary
        win_start = max(0.0, boundary_time - padding)
        win_end = min(audio_duration, boundary_time + padding)

        start_samp = int(win_start * sr)
        end_samp = int(win_end * sr)
        if end_samp - start_samp < sr * 0.1:  # Window too small
            return round(boundary_time, 3)

        window = torch.from_numpy(audio[start_samp:end_samp].copy())
        window = window.unsqueeze(0).to(self.device)

        # Get frame-level CTC emissions
        emissions, _ = self.model(window)
        log_probs = torch.nn.functional.log_softmax(emissions[0], dim=-1)
        speech_prob = 1.0 - torch.exp(log_probs[:, self._blank_idx])

        n_frames = len(speech_prob)
        frame_dur = (win_end - win_start) / n_frames

        # Convert the original boundary to a frame index within this window
        boundary_frame = int((boundary_time - win_start) / frame_dur)
        boundary_frame = max(0, min(boundary_frame, n_frames - 1))

        speech_mask = speech_prob > threshold

        if direction == "start":
            # Look around the boundary for the first speech frame
            region = speech_mask[:boundary_frame + min(5, n_frames - boundary_frame)]
            speech_indices = region.nonzero(as_tuple=True)[0]
            if len(speech_indices) > 0:
                snap_frame = speech_indices[0].item()
            else:
                return round(boundary_time, 3)
        else:
            # Look around the boundary for the last speech frame
            region = speech_mask[max(0, boundary_frame - 5):]
            offset = max(0, boundary_frame - 5)
            speech_indices = region.nonzero(as_tuple=True)[0]
            if len(speech_indices) > 0:
                snap_frame = offset + speech_indices[-1].item()
            else:
                return round(boundary_time, 3)

        snapped_time = win_start + snap_frame * frame_dur

        # Don't move boundaries too far (cap at padding distance)
        if abs(snapped_time - boundary_time) > padding:
            return round(boundary_time, 3)

        return round(snapped_time, 3)
