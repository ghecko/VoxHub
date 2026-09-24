import json
from typing import Any, Dict, List, Optional, Union
from fastapi.responses import JSONResponse, Response
from api.config import ResponseFormat
from core.format import OutputFormatter


def _unpack(result: Union[List[Dict[str, Any]], Dict[str, Any]]):
    """Accept both the legacy bare segment list and the result dict."""
    if isinstance(result, dict):
        return result.get("segments") or [], result
    return result or [], {}


def format_transcription(
    result: Union[List[Dict[str, Any]], Dict[str, Any]],
    response_format: ResponseFormat,
    include_words: bool = False,
) -> Response:
    """
    Format a transcription result into the requested format.

    ``result`` is the dict returned by ``TranscriptionService.transcribe``
    (``segments``, ``language``, ``duration``, ``pipeline``, ``warnings``) or,
    for backward compatibility, a bare list of segments.

    ``include_words`` adds the per-word timestamps/speakers to ``verbose_json``
    (OpenAI ``timestamp_granularities[]=word`` semantics). Words are only
    present when the wordalign pipeline produced them.
    """
    transcript_data, meta = _unpack(result)
    full_text = " ".join([seg["text"] for seg in transcript_data]).strip()
    warnings = meta.get("warnings") or []

    if response_format == ResponseFormat.JSON:
        body: Dict[str, Any] = {"text": full_text}
        if warnings:
            body["warnings"] = warnings
        return JSONResponse(content=body)

    elif response_format == ResponseFormat.VERBOSE_JSON:
        segments = []
        for i, seg in enumerate(transcript_data):
            item: Dict[str, Any] = {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker", "SPEAKER_00"),  # VoxHub extension
                # OpenAI fields we cannot compute for every backend. `confidence`
                # (VoxHub extension, 0-1, from CTC alignment) is the one to use.
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
            if "confidence" in seg:
                item["confidence"] = seg["confidence"]
            if include_words and seg.get("words"):
                item["words"] = [
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "speaker": w.get("speaker", item["speaker"]),
                        "score": w.get("score", 0.0),
                    }
                    for w in seg["words"]
                ]
            segments.append(item)

        duration = meta.get("duration")
        if duration is None:
            duration = transcript_data[-1].get("end", 0.0) if transcript_data else 0.0
        body = {
            "task": "transcribe",
            "language": meta.get("language") or "unknown",
            "duration": duration,
            "text": full_text,
            "segments": segments,
        }
        if include_words:
            body["words"] = [w for s in segments for w in s.get("words", [])]
        if meta.get("pipeline"):
            body["pipeline"] = meta["pipeline"]
        if warnings:
            body["warnings"] = warnings
        return JSONResponse(content=body)

    elif response_format == ResponseFormat.TEXT:
        return Response(content=full_text, media_type="text/plain")

    elif response_format == ResponseFormat.SRT:
        lines = []
        for i, entry in enumerate(transcript_data, 1):
            start = OutputFormatter._format_srt_time(entry.get("start", 0))
            end = OutputFormatter._format_srt_time(entry.get("end", 0))
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n\n")
        return Response(content="".join(lines), media_type="text/plain")

    elif response_format == ResponseFormat.VTT or response_format == ResponseFormat.VTT_JSON:
        lines = ["WEBVTT\n\n"]
        for i, entry in enumerate(transcript_data, 1):
            start = _format_vtt_time(entry.get("start", 0))
            end = _format_vtt_time(entry.get("end", 0))
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{start} --> {end}\n<{speaker}> {text}\n\n")

        if response_format == ResponseFormat.VTT_JSON:
            plain_segments = [
                {k: v for k, v in s.items() if k != "words" or include_words}
                for s in transcript_data
            ]
            body = {"text": full_text, "vtt": "".join(lines), "segments": plain_segments}
            if warnings:
                body["warnings"] = warnings
            return JSONResponse(content=body)
        return Response(content="".join(lines), media_type="text/vtt")

    return JSONResponse(content={"text": full_text})


def _format_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp format HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
