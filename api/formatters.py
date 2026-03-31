import json
from typing import List, Dict, Any
from fastapi.responses import JSONResponse, Response
from api.config import ResponseFormat
from core.format import OutputFormatter

def format_transcription(transcript_data: List[Dict[str, Any]], response_format: ResponseFormat) -> Response:
    """
    Format a list of transcription segments into the requested format.
    """
    full_text = " ".join([seg["text"] for seg in transcript_data]).strip()
    
    if response_format == ResponseFormat.JSON:
        return JSONResponse(content={"text": full_text})
    
    elif response_format == ResponseFormat.VERBOSE_JSON:
        # OpenAI verbose_json format refinement
        return JSONResponse(content={
            "task": "transcribe",
            "language": transcript_data[0].get("language", "unknown") if transcript_data else "unknown",
            "duration": transcript_data[-1].get("end", 0.0) if transcript_data else 0.0,
            "text": full_text,
            "segments": [
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": seg.get("speaker", "SPEAKER_00"), # VoxBench extension
                    # Mocking mandatory OpenAI fields if missing
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                }
                for i, seg in enumerate(transcript_data)
            ]
        })
    
    elif response_format == ResponseFormat.TEXT:
        return Response(content=full_text, media_type="text/plain")
    
    elif response_format == ResponseFormat.SRT:
        # We need a temporary way to get the SRT string.
        # core.format.OutputFormatter currently writes to a file.
        # Let's implement a string-based SRT formatter here.
        lines = []
        for i, entry in enumerate(transcript_data, 1):
            start = OutputFormatter._format_srt_time(entry.get("start", 0))
            end = OutputFormatter._format_srt_time(entry.get("end", 0))
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n\n")
        return Response(content="".join(lines), media_type="text/plain")
    
    elif response_format == ResponseFormat.VTT or response_format == ResponseFormat.VTT_JSON:
        # Basic VTT implementation
        lines = ["WEBVTT\n\n"]
        for i, entry in enumerate(transcript_data, 1):
            start = _format_vtt_time(entry.get("start", 0))
            end = _format_vtt_time(entry.get("end", 0))
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{start} --> {end}\n<{speaker}> {text}\n\n")
        
        if response_format == ResponseFormat.VTT_JSON:
            return JSONResponse(content={
                "text": full_text,
                "vtt": "".join(lines),
                "segments": transcript_data
            })
        return Response(content="".join(lines), media_type="text/vtt")

    return JSONResponse(content={"text": full_text})

def _format_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp format HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
