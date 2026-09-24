# VoxHub API Reference

Base URL: `http://localhost:8000`

---

## Authentication

If `VOXHUB_API_KEY` is set on the server, every request must include a Bearer token:

```
Authorization: Bearer <your-api-key>
```

If no key is configured, authentication is disabled and this header is ignored.

---

## Request Tracking

Every response includes an `X-Request-ID` header. You can also send your own `X-Request-ID` header on the request — the server will echo it back. If omitted, a UUID is generated automatically. Useful for correlating logs across services.

---

## Endpoints

### `POST /v1/audio/transcriptions`

Transcribe an audio file. OpenAI-compatible — any client that works with the OpenAI Audio API works here.

**Content-Type:** `multipart/form-data`

#### Parameters

| Field | Type | Required | Default | Description |
|:------|:-----|:---------|:--------|:------------|
| `file` | file | **yes** | — | Audio file to transcribe (`.mp3`, `.wav`, `.m4a`, `.flac`, etc.) |
| `model` | string | no | Server default (`whisper:turbo`) | Model specifier. See [Models](#models) below. |
| `language` | string | no | Auto-detect | ISO 639-1 language code (e.g. `fr`, `en`, `de`). Forces the model to transcribe in the given language instead of auto-detecting. Useful when auto-detect produces translations instead of transcriptions. |
| `prompt` | string | no | `null` | Optional text to guide the model's style or vocabulary. Works like OpenAI's `prompt` parameter. |
| `response_format` | string | no | `json` | Output format. One of: `json`, `verbose_json`, `text`, `srt`, `vtt`, `vtt_json`. See [Response Formats](#response-formats). |
| `temperature` | float | no | `0.0` | Sampling temperature (0.0–1.0). Higher values produce more varied output. |
| `timestamp_granularities[]` | string[] | no | `["segment"]` | Timestamp detail level. Values: `segment`, `word`. With `word`, `verbose_json` carries a `words` array per segment (and a flat top-level `words`). Word timestamps are produced by the `wordalign` pipeline only. |
| `diarize` | bool | no | Server default (`true`) | Enable/disable speaker diarization for this request. |
| `vad_mode` | string | no | Server default (`hybrid`) | **Legacy pipeline only.** VAD strategy: `silero`, `pyannote`, `hybrid`, `none`. See [VAD Modes](#vad-modes). With `diarize=true` and `vad_mode=silero` every segment is `SPEAKER_00` and a `warnings` entry is returned. |
| `pipeline` | string | no | Server default (`wordalign`) | `wordalign` (transcribe silence-bounded chunks, diarize in parallel, CTC-align words, attach a speaker per word) or `legacy` (diarize first, transcribe each speaker turn). See [Pipelines](#pipelines). |
| `num_speakers` | int | no | — | Exact number of speakers (pyannote hint). Mutually exclusive with `min_speakers`/`max_speakers`. |
| `min_speakers` | int | no | — | Lower bound on the number of speakers. |
| `max_speakers` | int | no | — | Upper bound on the number of speakers. |
| `return_speaker_embeddings` | string | no | `"false"` | `"true"` adds a per-speaker 512-d voice embedding (`speaker_embeddings`) plus `speaker_embedding_model` to JSON responses. Requires `diarize=true`. |

#### Example

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-key" \
  -F file=@meeting.mp3 \
  -F model=whisper:turbo \
  -F language=en \
  -F response_format=verbose_json \
  -F diarize=true \
  -F vad_mode=hybrid
```

#### Response (`json`)

```json
{
  "text": "Hello, welcome to the meeting."
}
```

#### Response (`verbose_json`)

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 42.5,
  "text": "Hello, welcome to the meeting. Let's begin.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.4,
      "text": "Hello, welcome to the meeting.",
      "speaker": "SPEAKER_00",
      "confidence": 0.91,
      "words": [
        {"word": "Hello,", "start": 0.02, "end": 0.41, "speaker": "SPEAKER_00", "score": 0.97},
        {"word": "welcome", "start": 0.48, "end": 0.90, "speaker": "SPEAKER_00", "score": 0.95}
      ],
      "avg_logprob": 0.0,
      "compression_ratio": 0.0,
      "no_speech_prob": 0.0
    },
    {
      "id": 1,
      "start": 3.1,
      "end": 4.8,
      "text": "Let's begin.",
      "speaker": "SPEAKER_01",
      "confidence": 0.88,
      "avg_logprob": 0.0,
      "compression_ratio": 0.0,
      "no_speech_prob": 0.0
    }
  ],
  "pipeline": "wordalign",
  "warnings": []
}
```

> **Note:** `speaker`, `confidence`, `words`, `pipeline` and `warnings` are VoxHub extensions not present in the OpenAI API. `language` is the code detected by the language probe (or the one you passed), `duration` is the real audio duration. `confidence` (0-1) is the mean CTC alignment score of the segment's words and is only present on the `wordalign` pipeline; `words` only when `timestamp_granularities[]=word` was requested. `avg_logprob`, `compression_ratio` and `no_speech_prob` are kept for OpenAI compatibility and return `0.0`.

#### Response (`text`)

```
Hello, welcome to the meeting. Let's begin.
```

Content-Type: `text/plain`

#### Response (`srt`)

```
1
00:00:00,000 --> 00:00:02,400
[SPEAKER_00] Hello, welcome to the meeting.

2
00:00:03,100 --> 00:00:04,800
[SPEAKER_01] Let's begin.
```

Content-Type: `text/plain`

#### Response (`vtt`)

```
WEBVTT

00:00:00.000 --> 00:00:02.400
<SPEAKER_00> Hello, welcome to the meeting.

00:00:03.100 --> 00:00:04.800
<SPEAKER_01> Let's begin.
```

Content-Type: `text/vtt`

#### Response (`vtt_json`)

Returns both the VTT string and the raw segment data:

```json
{
  "text": "Hello, welcome to the meeting. Let's begin.",
  "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:02.400\n<SPEAKER_00> Hello...\n\n",
  "segments": [
    {
      "start": 0.0,
      "end": 2.4,
      "text": "Hello, welcome to the meeting.",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

---

### `POST /v1/audio/translations`

OpenAI-compatible translation endpoint. Currently routes to the transcription pipeline (most backends focus on transcription rather than translation).

Takes the same `file`, `model`, and `response_format` parameters as `/v1/audio/transcriptions`.

---

### `POST /v1/audio/transcriptions/jobs`

Start an asynchronous transcription job. Use this for long audio files where you don't want to hold a connection open.

**Content-Type:** `multipart/form-data`

#### Parameters

Same as `/v1/audio/transcriptions`, except `temperature` and `timestamp_granularities[]` are not available on the jobs endpoint.

| Field | Type | Required | Default | Description |
|:------|:-----|:---------|:--------|:------------|
| `file` | file | **yes** | — | Audio file to transcribe |
| `model` | string | no | Server default | Model specifier |
| `language` | string | no | Auto-detect | ISO 639-1 language code |
| `prompt` | string | no | `null` | Style/vocabulary hint |
| `response_format` | string | no | `json` | Output format for the eventual result |
| `diarize` | bool | no | Server default | Enable/disable diarization |
| `vad_mode` | string | no | Server default | VAD strategy (legacy pipeline): `silero`, `pyannote`, `hybrid`, `none` |
| `pipeline` | string | no | Server default | `wordalign` or `legacy` |
| `num_speakers` / `min_speakers` / `max_speakers` | int | no | — | pyannote speaker-count hints (exact count *or* range) |
| `return_speaker_embeddings` | string | no | `"false"` | `"true"` to compute per-speaker voice embeddings |

#### Example

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions/jobs \
  -F file=@long_recording.mp3 \
  -F model=voxtral:mini-3b \
  -F vad_mode=pyannote \
  -F diarize=true
```

#### Response

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "links": {
    "status": "/v1/audio/transcriptions/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "result": "/v1/audio/transcriptions/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/result"
  }
}
```

---

### `GET /v1/audio/transcriptions/jobs`

List all transcription jobs currently held in memory, sorted newest-first. Useful for monitoring the queue, checking how many jobs are pending or in progress, and discovering job IDs.

#### Query Parameters

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `status` | string | — | Optional filter. One of: `pending`, `processing`, `completed`, `failed`, `cancelled`. When omitted, all jobs are returned. |

#### Example

```bash
# All jobs
curl http://localhost:8000/v1/audio/transcriptions/jobs \
  -H "Authorization: Bearer my-secret-key"

# Only jobs currently running
curl "http://localhost:8000/v1/audio/transcriptions/jobs?status=processing" \
  -H "Authorization: Bearer my-secret-key"
```

#### Response

```json
{
  "jobs": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "status": "processing",
      "stage": "transcribing",
      "progress": 42,
      "created_at": 1712300000.0,
      "completed_at": null,
      "error": null
    },
    {
      "id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
      "status": "pending",
      "stage": null,
      "progress": 0,
      "created_at": 1712299950.0,
      "completed_at": null,
      "error": null
    }
  ],
  "total": 5,
  "counts": {
    "pending": 1,
    "processing": 1,
    "completed": 2,
    "cancelled": 1
  }
}
```

| Field | Type | Description |
|:------|:-----|:------------|
| `jobs` | array | Job objects (without the `result` payload) matching the filter, sorted by `created_at` descending |
| `total` | int | Total number of jobs across all statuses |
| `counts` | object | Breakdown of jobs by status (`pending`, `processing`, `completed`, `failed`, `cancelled`) |

> **Note:** The `result` field is intentionally omitted from the list response to keep the payload small. Use `GET /v1/audio/transcriptions/jobs/{job_id}/result` to retrieve a completed job's transcription.

---

### `GET /v1/audio/transcriptions/jobs/{job_id}`

Poll the status of an async transcription job.

#### Response

```json
{
  "id": "a1b2c3d4-...",
  "status": "processing",
  "stage": "transcribing",
  "progress": 42.5,
  "chunks_done": 7,
  "chunks_total": 12
}
```

| Field | Type | Description |
|:------|:-----|:------------|
| `id` | string | The job identifier |
| `status` | string | One of: `pending`, `processing`, `completed`, `failed`, `cancelled` |
| `stage` | string or null | Current pipeline stage: `loading`, `detecting_language`, `vad`, `diarizing`, `transcribing`, `aligning`, `embeddings`, or `null` when done |
| `progress` | float | Percentage complete (0–100). Moves at least every ~5 s while a long step (pyannote) runs, so pollers can use "no change for N seconds" as a staleness signal. |
| `chunks_done` / `chunks_total` | int or null | `wordalign` pipeline only: transcription sub-progress. `chunks_total` can grow when a chunk is re-split after a dropped output. |

---

### `GET /v1/audio/transcriptions/jobs/{job_id}/result`

Retrieve the result of a completed job. Returns `400` if the job is not yet completed.

#### Query Parameters

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `response_format` | string | `json` | Output format (same options as the transcription endpoint) |
| `words` | bool | `false` | Add per-word timestamps and speakers to `verbose_json` (same as `timestamp_granularities[]=word` on the sync endpoint) |

#### Response

Same structure as the synchronous transcription endpoint, depending on the chosen `response_format`.

---

### `POST /v1/audio/transcriptions/jobs/{job_id}/cancel`

Cancel a pending or running transcription job. The cancellation is cooperative: a running job will stop at the next segment boundary. All partial results are discarded.

#### Response

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "cancelled"
}
```

Returns `404` if the job does not exist. Returns `409` if the job has already reached a terminal state (`completed`, `failed`, or `cancelled`).

#### Example

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/cancel \
  -H "Authorization: Bearer my-secret-key"
```

---

### `DELETE /v1/audio/transcriptions/jobs/{job_id}`

Delete a finished job from server memory, freeing the resources held by its result payload. Only jobs in a terminal state (`completed`, `failed`, `cancelled`) can be deleted. Running or pending jobs must be cancelled first.

#### Response

```json
{
  "job_id": "a1b2c3d4-...",
  "deleted": true
}
```

Returns `404` if the job does not exist. Returns `409` if the job is still running or pending (cancel it first).

#### Example

```bash
curl -X DELETE http://localhost:8000/v1/audio/transcriptions/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Authorization: Bearer my-secret-key"
```

---

### `GET /v1/models`

List all models registered in `models.yaml` (OpenAI-compatible format).

#### Response

```json
{
  "data": [
    { "id": "whisper:turbo", "object": "model", "owned_by": "voxhub", "permission": [] },
    { "id": "whisper:large-v3", "object": "model", "owned_by": "voxhub", "permission": [] },
    { "id": "voxtral:mini-3b", "object": "model", "owned_by": "voxhub", "permission": [] }
  ]
}
```

---

### `GET /models/list`

List models currently loaded in memory (VRAM).

#### Response

```json
{
  "models": ["whisper:turbo"]
}
```

---

### `POST /models/load`

Pre-load a model into memory so the first transcription request doesn't incur the loading delay.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `model` | string | **yes** | Model specifier to load (e.g. `whisper:turbo`) |

#### Response

```json
{ "status": "success", "model": "whisper:turbo" }
```

---

### `POST /models/unload`

Unload a model from memory to free VRAM.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `model` | string | **yes** | Model specifier to unload |

#### Response

```json
{ "status": "success", "model": "whisper:turbo" }
```

Returns `404` if the model is not currently loaded.

---

### `GET /health`

Health check. Also available at `GET /v1/health`.

#### Response

```json
{ "status": "healthy" }
```

---

## Reference

### Models

Models follow the `family:variant` naming convention. Available models (from `models.yaml`):

| Specifier | Family | Notes |
|:----------|:-------|:------|
| `whisper:turbo` | Whisper (OpenAI) | Fast, good quality. **Default.** |
| `whisper:large-v3` | Whisper (OpenAI) | Highest accuracy Whisper variant |
| `whisper:small` | Whisper (OpenAI) | Lightweight, lower VRAM |
| `whisper:medium` | Whisper (OpenAI) | Mid-range (disabled by default) |
| `voxtral:mini-3b` | Voxtral (Mistral, transformers) | Strong French/English, beats Whisper large-v3 on FLEURS, in-process |
| `voxtral:mini-3b-vllm` | Voxtral (Mistral, vLLM) | Same weights served by the `voxtral-vllm` container. Higher throughput; requires `docker compose --profile vllm up`. |
| `voxtral:small-24b` | Voxtral (Mistral, transformers) | High quality, large model (disabled by default) |
| `voxtral:small-24b-vllm` | Voxtral (Mistral, vLLM) | High-quality model served remotely via vLLM (disabled by default) |
| `moonshine:base` | Moonshine (UsefulSensors) | Ultra-low latency, **English only** |
| `moonshine:tiny` | Moonshine (UsefulSensors) | Minimal footprint (disabled by default) |
| `canary:1b` | Canary (NVIDIA NeMo) | SOTA accuracy, multi-task |

Models are loaded lazily on first request. Use `POST /models/load` to pre-warm.

### VAD Modes

The `vad_mode` parameter controls how audio is segmented before transcription:

| Value | Description | Best for |
|:------|:------------|:---------|
| `silero` | Lightweight Silero VAD | Speed-first workflows |
| `pyannote` | High-accuracy Pyannote segmentation + speaker labels | Production quality |
| `hybrid` | Silero as a sensitive gate, Pyannote as a refiner. Keeps high-confidence Silero segments even if Pyannote disagrees. | Best recall + precision (default) |
| `none` | No segmentation — process the entire audio as one chunk | Pre-segmented audio |

> **Note:** `pyannote` and `hybrid` modes require a valid `HF_TOKEN` environment variable (Hugging Face access token with permission for Pyannote models).

### Pipelines

| Value | How it works | Trade-offs |
|:------|:-------------|:-----------|
| `wordalign` (default) | 1. Silero finds silences and the audio is cut into speaker-agnostic chunks (~60 s, max 180 s). 2. The chunks are transcribed with full context (concurrently on vLLM backends) while pyannote diarizes the whole file in parallel. 3. Each chunk's text is force-aligned to its audio with a CTC model (`VOXHUB_ALIGN_MODEL`, default `MMS_FA`) to get word timestamps. 4. Every word takes the speaker of the turn it overlaps most; words are regrouped into segments on speaker change or pauses > 1 s. | Best ASR quality (context), per-word speakers, overlaps resolved word by word, word timestamps and confidence in the output. Needs the aligner model (~1.2 GB for MMS_FA). Word boundaries are as good as the CTC model (~50-100 ms). |
| `legacy` | pyannote (or Silero/hybrid) segments the audio into speaker turns first; each turn is transcribed on its own, then merged. | No extra model. Short turns (< 0.5 s) are dropped, interjections are absorbed into the surrounding turn, overlapping speech is trimmed, and each turn is transcribed without context. |

When the aligner cannot be loaded the server falls back to `legacy` and says so in `warnings`.

### Response Formats

| Value | Content-Type | Description |
|:------|:-------------|:------------|
| `json` | `application/json` | Plain text result: `{"text": "..."}` |
| `verbose_json` | `application/json` | Full segment-level detail with timestamps, speakers, and OpenAI-compatible metadata |
| `text` | `text/plain` | Raw transcript text, no structure |
| `srt` | `text/plain` | SubRip subtitle format with speaker labels |
| `vtt` | `text/vtt` | WebVTT subtitle format with speaker labels |
| `vtt_json` | `application/json` | Combined: the VTT string, the full text, and the raw segment array |

### Result Retention

Job results are stored in server memory and are **automatically purged** after a configurable time-to-live (TTL) measured from the moment the job reaches a terminal state (`completed`, `failed`, or `cancelled`).

| Environment Variable | Default | Description |
|:---------------------|:--------|:------------|
| `VOXHUB_RESULT_TTL` | `3600` (1 hour) | Seconds to keep finished job results. Set to `0` to disable automatic purging (results kept until server restart or manual deletion). |

The cleanup process runs every 60 seconds and removes all expired jobs silently. Once a job is purged, any subsequent `GET` on its status or result endpoint returns `404`.

To free memory before the TTL expires, use `DELETE /v1/audio/transcriptions/jobs/{job_id}` on any finished job.

> **Note:** All job state is in-memory. A server restart clears every job regardless of TTL.

---

## Error Responses

All errors return JSON:

```json
{
  "detail": "Descriptive error message"
}
```

| Status | Meaning |
|:-------|:--------|
| `400` | Job not yet 