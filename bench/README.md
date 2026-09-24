# VoxHub benchmark harness

Compares pipelines (`legacy` vs `wordalign`) and models on the same
recordings with the metrics that matter for meeting transcripts:

| Metric | What it measures | Needs |
|:-------|:-----------------|:------|
| **WER** | Word error rate of the full text, speakers ignored | `<stem>.txt` or `<stem>.ref.json` |
| **cpWER** | Speaker-attributed accuracy: WER computed per speaker after the best speaker mapping. A word given to the wrong person costs twice (insertion + deletion), so this is the number that moves when diarization or reconciliation is wrong. | `<stem>.ref.json` |
| **DER** | Diarization error rate (miss + false alarm + confusion, 0.25 s collar) | `<stem>.rttm` or `<stem>.ref.json` |
| **RTF** | Real-time factor (wall time / audio duration) | nothing |

Everything is implemented in `metrics.py` without dependencies so the harness
runs anywhere. `pyannote.metrics`, `meeteval` or `jiwer` give the same numbers
if you prefer the reference implementations.

## 1. Build a small reference set

Three or four HiDock recordings that are representative (a 2-person call, a
5-person meeting with interruptions, a noisy room) beat fifty clean samples.
For each recording `bench/data/<stem>.<ext>`:

* `<stem>.ref.json` is the easiest single reference and unlocks all metrics:

  ```json
  [
    {"start": 0.42, "end": 3.10, "speaker": "Jordan", "text": "Bon, on commence ?"},
    {"start": 3.30, "end": 5.85, "speaker": "Fabian", "text": "Oui, je partage mon écran."}
  ]
  ```

  The fastest way to produce it is to run VoxHub once, export the segments,
  then **fix them by hand in OpenHiNotes** (merge, split, rename, edit) and
  dump the corrected segments. Correcting is 10x faster than transcribing.

* `<stem>.rttm` (NIST format) if you have a diarization-only reference, e.g.
  from a labelling tool.
* `<stem>.txt` if you only have a plain transcript.

Speaker names in the reference do not need to match the hypothesis labels:
both cpWER and DER search the best 1:1 mapping.

## 2. Run

```bash
# both pipelines, server default model
python bench/run_bench.py --api http://localhost:8000 --data bench/data

# pipelines x models
python bench/run_bench.py --data bench/data \
  --pipeline legacy --pipeline wordalign \
  --model voxtral:mini-3b-vllm --model whisper:large-v3

# pass any extra form field to the API (speaker hints, VAD mode...)
python bench/run_bench.py --data bench/data --extra num_speakers=3
```

Raw `verbose_json` hypotheses land in `bench/results/last/` next to a
`results.json`; `--rescore bench/results/last` recomputes the table from them
without touching the GPU (useful after changing a metric or a reference).

## 3. Reading the table

* `wordalign` should lower **cpWER** noticeably (short turns and overlaps are
  no longer dropped/trimmed) and lower or keep **WER** (full-context chunks).
  If WER goes *up*, look at `warnings` (aligner fallback) and at the chunk
  logs: a hallucination loop that survived the repetition guard shows up as
  a chunk with a huge word count.
* **DER** is the same pyannote run in both pipelines; it moves only with
  `num_speakers` hints or a pyannote version change. A DER gap between the
  two pipelines means the legacy sanitizer (micro-turn absorption, overlap
  trimming) is dropping speech.
* **RTF** on vLLM should drop with `wordalign` thanks to concurrent chunks;
  on the in-process transformers backend it is roughly unchanged (the
  aligner adds a little, the shorter segment list removes a little).

## 4. Benchmarking another backend (VibeVoice-ASR, ...)

A joint STT + diarization model is just another `models.yaml` entry to this
harness. Register the backend class, then:

```bash
python bench/run_bench.py --data bench/data --model vibevoice:7b-asr --pipeline legacy --extra vad_mode=none
```

The backend's own speaker labels end up in `segments[].speaker` and are scored
with the same cpWER / DER as Voxtral + pyannote. Two things to keep in mind
when wiring such a backend into `api/transcriber.py`:

1. It wants the **whole file**, not chunks: add a `supports_whole_file` flag
   on the transcriber and short-circuit the chunk loop when it is set (the
   `_run_legacy` path with `vad_mode=none` already gives one big segment,
   which is enough for a first measurement).
2. Keep **pyannote/embedding** for voice profiles even if the model diarizes
   by itself, otherwise every enrolled `VoiceProfile` in OpenHiNotes silently
   stops matching (different embedding space).
