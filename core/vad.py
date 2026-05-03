import os
import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from core.diarize import DiarizationAnalyzer

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero Voice Activity Detection - Lightweight and fast.
    """
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 500):
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self.model = load_silero_vad()
            self.get_speech_timestamps = get_speech_timestamps
            self.threshold = threshold
            self.min_speech_duration_ms = min_speech_duration_ms
            self.min_silence_duration_ms = min_silence_duration_ms
        except ImportError:
            raise ImportError("silero-vad not installed. Run 'pip install silero-vad'")

    def detect(self, audio: np.ndarray, sampling_rate: int = 16000) -> List[Dict]:
        """
        Detect speech segments in the audio.
        Returns [{"start": float_s, "end": float_s}]
        """
        wav = torch.from_numpy(audio.copy())
        timestamps = self.get_speech_timestamps(
            wav, self.model,
            sampling_rate=sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )
        return [{"start": t["start"], "end": t["end"]} for t in timestamps]

    def detect_with_probabilities(self, audio: np.ndarray, sampling_rate: int = 16000) -> List[Dict]:
        """
        Detect speech segments AND return per-segment confidence scores.
        Used by HybridVAD to make override decisions.

        Uses a single streaming pass over the full audio to collect frame-level
        probabilities, then maps each detected segment to the peak probability
        observed within its time range. This is O(n) over the audio length
        regardless of how many segments exist — much faster than the previous
        approach of re-running the model per segment.

        Returns [{"start": float_s, "end": float_s, "probability": float}]
        """
        wav = torch.from_numpy(audio.copy())
        frame_size = 512  # Silero operates on 512-sample (32ms @ 16kHz) frames

        # --- Pass 1: Single streaming pass to collect all frame probabilities ---
        self.model.reset_states()
        frame_probs = []
        for i in range(0, len(wav), frame_size):
            frame = wav[i:i + frame_size]
            if len(frame) < frame_size:
                frame = torch.nn.functional.pad(frame, (0, frame_size - len(frame)))
            prob = self.model(frame.unsqueeze(0), sampling_rate).item()
            frame_probs.append(prob)

        # --- Pass 2: Get speech timestamps (Silero resets internally) ---
        self.model.reset_states()
        timestamps = self.get_speech_timestamps(
            wav, self.model,
            sampling_rate=sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=True,
        )

        # --- Map each segment to its peak frame probability ---
        results = []
        for t in timestamps:
            start_frame = int(t["start"] * sampling_rate) // frame_size
            end_frame = int(t["end"] * sampling_rate) // frame_size + 1
            end_frame = min(end_frame, len(frame_probs))

            if start_frame < end_frame:
                max_prob = max(frame_probs[start_frame:end_frame])
            else:
                max_prob = 0.0

            results.append({
                "start": t["start"],
                "end": t["end"],
                "probability": round(max_prob, 4),
            })

        return results


class HybridVAD:
    """
    Hybrid VAD Pipeline: Silero (gate) + Pyannote (refiner).

    Strategy ("Refined Gating"):
        1. Silero runs first with a sensitive threshold to capture all potential speech.
        2. Only Silero-detected regions are passed to Pyannote for fine-grained
           segmentation and optional speaker diarization.
        3. If Pyannote rejects a Silero segment but Silero's confidence was above
           the override threshold, the segment is kept anyway (safety net).

    This gives you Silero's recall with Pyannote's precision, while the override
    threshold acts as a safety net for high-confidence speech that Pyannote
    might drop (e.g. very short utterances, overlapping speakers).

    Design note — Pyannote on full audio vs. gated regions:
        Pyannote's diarization model runs on the FULL audio, not just the Silero-gated
        regions. This is intentional: Pyannote's speaker embedding and clustering
        require global context to assign consistent speaker labels across the entire
        file. Feeding it isolated chunks would produce fragmented, inconsistent speaker
        IDs (e.g., the same person labeled SPEAKER_00 in one chunk and SPEAKER_02 in
        another).

        If you don't need diarization and only want precise VAD boundaries, a future
        "gated-only" variant could pass only Silero's regions to Pyannote's
        segmentation model (not the full diarization pipeline) for a significant
        speed-up. This is not yet implemented.

    Args:
        silero_threshold:  Silero detection sensitivity. Lower = more sensitive.
                           Default 0.35 (more aggressive than standalone default of 0.5).
        override_threshold: Silero probability above which a segment is kept even
                            if Pyannote disagrees. Default 0.8.
        hf_token:          HuggingFace token for Pyannote authentication.
    """

    def __init__(
        self,
        silero_threshold: float = 0.35,
        override_threshold: float = 0.8,
        hf_token: Optional[str] = None,
    ):
        self.silero_threshold = silero_threshold
        self.override_threshold = override_threshold
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._silero: Optional[SileroVAD] = None
        self._pyannote: Optional[DiarizationAnalyzer] = None

    @property
    def silero(self) -> SileroVAD:
        if self._silero is None:
            self._silero = SileroVAD(
                threshold=self.silero_threshold,
                min_speech_duration_ms=200,   # More permissive for gating
                min_silence_duration_ms=300,
            )
        return self._silero

    @property
    def pyannote(self) -> DiarizationAnalyzer:
        if self._pyannote is None:
            self._pyannote = DiarizationAnalyzer(auth_token=self.hf_token)
        return self._pyannote

    def detect(self, audio: np.ndarray, sampling_rate: int = 16000, **kwargs) -> List[Dict]:
        """
        Run the hybrid pipeline.

        Returns a list of segments. If diarize=True, segments include "speaker" labels
        from Pyannote. Override segments (Silero-only) get speaker="SPEAKER_OVERRIDE".
        """
        diarize = kwargs.get("diarize", False)
        on_progress: Optional[Callable] = kwargs.get("on_progress")

        # --- Step 1: Silero gate (sensitive) ---
        logger.info(
            f"[HybridVAD] Step 1/3: Silero gate (threshold={self.silero_threshold})"
        )
        if on_progress:
            on_progress("vad", 0.05)  # Silero starting

        silero_segments = self.silero.detect_with_probabilities(audio, sampling_rate)
        logger.info(f"[HybridVAD]   -> {len(silero_segments)} candidate segments")

        if not silero_segments:
            return []

        if on_progress:
            on_progress("vad", 0.10)  # Silero complete

        # --- Step 2: Pyannote refiner on gated regions ---
        logger.info("[HybridVAD] Step 2/3: Pyannote refinement")
        if on_progress:
            on_progress("diarizing", 0.12)  # Pyannote starting

        pyannote_kwargs = {}
        if diarize:
            for key in ("num_speakers", "min_speakers", "max_speakers"):
                if kwargs.get(key) is not None:
                    pyannote_kwargs[key] = kwargs[key]

        # Build a pyannote sub-progress callback that maps pyannote's internal
        # steps (segmentation/embeddings/clustering) into the 0.12–0.80 range.
        def _diarize_progress(step_name: str, frac: float):
            if on_progress:
                # Map pyannote's 0-1 fraction into overall 0.12-0.80 range
                overall = 0.12 + frac * 0.68
                on_progress("diarizing", overall)

        pyannote_segments = self.pyannote.diarize(
            audio, sampling_rate=sampling_rate,
            on_progress=_diarize_progress,
            **pyannote_kwargs,
        )
        logger.info(f"[HybridVAD]   -> {len(pyannote_segments)} refined segments")

        # --- Step 3: Reconcile with safety-net override ---
        # For each Silero segment, find sub-regions NOT covered by Pyannote.
        # If Silero's confidence for that segment was above the override threshold,
        # add the uncovered sub-regions as override segments.
        #
        # This fixes the "whole-segment coverage" bug: a 3-minute Silero segment
        # could be 80% covered by Pyannote, but the remaining 20% (36s of speech)
        # would be silently dropped. Now those gaps are detected and preserved.
        logger.info(
            f"[HybridVAD] Step 3/3: Reconciling (override_threshold={self.override_threshold})"
        )
        if on_progress:
            on_progress("diarizing", 0.85)  # Reconciliation
        final_segments = list(pyannote_segments)  # Start with Pyannote's output

        overrides_added = 0
        for s_seg in silero_segments:
            gaps = self._find_uncovered_regions(s_seg, pyannote_segments)
            if not gaps:
                continue

            # Only override if Silero was confident about this segment
            if s_seg["probability"] >= self.override_threshold:
                for gap in gaps:
                    final_segments.append({
                        "start": gap["start"],
                        "end": gap["end"],
                        "speaker": "SPEAKER_OVERRIDE" if diarize else "SPEAKER_00",
                    })
                    overrides_added += 1
            else:
                logger.debug(
                    f"[HybridVAD]   Skipping {len(gaps)} gap(s) in "
                    f"[{s_seg['start']:.1f}-{s_seg['end']:.1f}s] — "
                    f"Silero prob {s_seg['probability']:.2f} < override {self.override_threshold}"
                )

        if overrides_added:
            logger.info(
                f"[HybridVAD]   -> {overrides_added} Silero override segment(s) added"
            )

        # Sort by start time
        final_segments.sort(key=lambda s: s["start"])

        # --- Step 4: Assign override segments to nearest known speaker ---
        # Override segments have no speaker label from Pyannote (it missed them).
        # Assign each one to the nearest Pyannote-labeled speaker by time proximity.
        # This keeps the diarization coherent instead of showing "SPEAKER_OVERRIDE".
        if diarize and pyannote_segments:
            final_segments = self._assign_override_speakers(
                final_segments, pyannote_segments
            )
        elif not diarize:
            final_segments = [
                {"start": s["start"], "end": s["end"], "speaker": s.get("speaker", "SPEAKER_00")}
                for s in final_segments
            ]

        logger.info(f"[HybridVAD] Final: {len(final_segments)} segments")
        if on_progress:
            on_progress("diarizing", 1.0)  # VAD+diarization complete
        return final_segments

    @staticmethod
    def _assign_override_speakers(
        segments: List[Dict],
        pyannote_segments: List[Dict],
    ) -> List[Dict]:
        """
        Replace SPEAKER_OVERRIDE labels with the nearest known speaker.

        For each override segment, find the closest Pyannote segment by time
        distance (gap between endpoints) and adopt its speaker label. If no
        Pyannote segments exist, fall back to SPEAKER_00.
        """
        result = []
        for seg in segments:
            if seg.get("speaker") != "SPEAKER_OVERRIDE":
                result.append(seg)
                continue

            # Find the Pyannote segment closest in time
            best_speaker = "SPEAKER_00"
            best_distance = float("inf")
            mid = (seg["start"] + seg["end"]) / 2

            for p_seg in pyannote_segments:
                # Distance = gap between the two segments (0 if overlapping)
                p_mid = (p_seg["start"] + p_seg["end"]) / 2
                dist = abs(mid - p_mid)
                if dist < best_distance:
                    best_distance = dist
                    best_speaker = p_seg.get("speaker", "SPEAKER_00")

            assigned = dict(seg)
            assigned["speaker"] = best_speaker
            result.append(assigned)

        return result

    @staticmethod
    def _find_uncovered_regions(
        silero_seg: Dict,
        pyannote_segments: List[Dict],
        min_gap_duration: float = 0.3,
    ) -> List[Dict]:
        """
        Find sub-regions of a Silero segment that are NOT covered by any Pyannote segment.

        Instead of a binary "covered or not" check on the whole segment, this walks
        through the Silero time range and identifies every gap that Pyannote didn't
        claim. Gaps shorter than min_gap_duration (default 300ms) are ignored to
        avoid flooding the pipeline with micro-segments.

        Returns a list of {"start": float, "end": float} dicts for each gap, or
        an empty list if the Silero segment is fully covered.
        """
        s_start, s_end = silero_seg["start"], silero_seg["end"]
        if s_end - s_start <= 0:
            return []

        # Collect Pyannote segments that overlap with this Silero segment, sorted
        overlapping = []
        for p_seg in pyannote_segments:
            p_start = max(s_start, p_seg["start"])
            p_end = min(s_end, p_seg["end"])
            if p_end > p_start:
                overlapping.append((p_start, p_end))

        if not overlapping:
            # Pyannote has zero coverage of this entire Silero segment
            return [{"start": s_start, "end": s_end}]

        overlapping.sort(key=lambda x: x[0])

        # Merge overlapping Pyannote intervals within the Silero range
        merged = [overlapping[0]]
        for start, end in overlapping[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Walk the Silero range and collect gaps between merged Pyannote intervals
        gaps = []
        cursor = s_start

        for m_start, m_end in merged:
            if m_start > cursor:
                gap_dur = m_start - cursor
                if gap_dur >= min_gap_duration:
                    gaps.append({"start": round(cursor, 3), "end": round(m_start, 3)})
            cursor = max(cursor, m_end)

        # Trailing gap after the last Pyannote interval
        if cursor < s_end:
            gap_dur = s_end - cursor
            if gap_dur >= min_gap_duration:
                gaps.append({"start": round(cursor, 3), "end": round(s_end, 3)})

        return gaps


class UnifiedVAD:
    """
    Unified VAD Orchestrator.

    Supported modes:
        - "silero":   Fast, lightweight Silero VAD. Good default for speed.
        - "pyannote": High-quality Pyannote segmentation + optional diarization.
        - "hybrid":   Silero as a sensitive gate, Pyannote as a refiner, with a
                      confidence-based safety net. Best balance of recall and precision.
        - "none":     No VAD — treat the entire audio as one segment.

    See HybridVAD docstring for details on the Refined Gating strategy.
    """
    def __init__(self, mode: str = "silero", hf_token: Optional[str] = None,
                 silero_threshold: float = 0.35, override_threshold: float = 0.8):
        self.mode = mode
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.silero_threshold = silero_threshold
        self.override_threshold = override_threshold
        self.vad_model = None
        self._hybrid_fallback: Optional["HybridVAD"] = None

    def detect(self, audio: np.ndarray, sampling_rate: int = 16000, **kwargs) -> List[Dict]:
        """
        Orchestrate VAD based on the selected mode.

        Accepts an optional ``on_progress`` callback via kwargs.  The
        callback signature is ``(stage: str, fraction: float)`` where
        *stage* is a UI-friendly name ("vad", "diarizing") and *fraction*
        is 0.0–1.0 within the VAD phase.
        """
        on_progress: Optional[Callable] = kwargs.pop("on_progress", None)

        if self.mode == "silero":
            if kwargs.get("diarize", False):
                logger.warning(
                    "VAD mode 'silero' does not support speaker diarization — "
                    "automatically upgrading to 'hybrid' (Silero gate + pyannote) "
                    "to fulfil diarize=True. Set vad_mode='hybrid' or 'pyannote' "
                    "explicitly to silence this warning."
                )
                if self._hybrid_fallback is None:
                    self._hybrid_fallback = HybridVAD(
                        silero_threshold=self.silero_threshold,
                        override_threshold=self.override_threshold,
                        hf_token=self.hf_token,
                    )
                return self._hybrid_fallback.detect(
                    audio, sampling_rate, on_progress=on_progress, **kwargs
                )
            if not self.vad_model:
                self.vad_model = SileroVAD()
            if on_progress:
                on_progress("vad", 0.05)
            result = self.vad_model.detect(audio, sampling_rate)
            if on_progress:
                on_progress("vad", 1.0)
            return result

        elif self.mode == "pyannote":
            if not self.vad_model:
                self.vad_model = DiarizationAnalyzer(auth_token=self.hf_token)

            if on_progress:
                on_progress("diarizing", 0.05)

            # Check if we want full diarization or VAD-only
            diarize = kwargs.get("diarize", False)

            # Build a sub-progress callback for pyannote internal steps
            def _pyannote_progress(step_name: str, frac: float):
                if on_progress:
                    on_progress("diarizing", frac)

            if diarize:
                result = self.vad_model.diarize(
                    audio,
                    sampling_rate=sampling_rate,
                    on_progress=_pyannote_progress,
                    num_speakers=kwargs.get("num_speakers"),
                    min_speakers=kwargs.get("min_speakers"),
                    max_speakers=kwargs.get("max_speakers"),
                )
            else:
                result = self.vad_model.diarize(
                    audio, sampling_rate=sampling_rate,
                    on_progress=_pyannote_progress,
                )
            if on_progress:
                on_progress("diarizing", 1.0)
            return result

        elif self.mode == "hybrid":
            if not self.vad_model:
                self.vad_model = HybridVAD(
                    silero_threshold=self.silero_threshold,
                    override_threshold=self.override_threshold,
                    hf_token=self.hf_token,
                )
            # HybridVAD reads on_progress from kwargs
            return self.vad_model.detect(audio, sampling_rate, on_progress=on_progress, **kwargs)

        elif self.mode == "none":
            # Pass full audio as a single segment
            duration = len(audio) / sampling_rate
            if on_progress:
                on_progress("vad", 1.0)
            return [{"start": 0.0, "end": duration}]

        else:
            raise ValueError(f"Unknown VAD mode: {self.mode}")
