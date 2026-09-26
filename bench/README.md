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

# speaker-count hint taken from each file's reference (recommended)
python bench/run_bench.py --data bench/data --speakers-from-ref

# pass any extra form field to the API (VAD mode, a global speaker hint...)
python bench/run_bench.py --data bench/data --extra vad_mode=none
```

Raw `verbose_json` hypotheses land in `bench/results/last/` next to a
`results.json`; `--rescore bench/results/last` recomputes the table from them
without touching the GPU (useful after changing a metric or a reference).

Every hypothesis also carries `diarization`, the raw pyannote turns before
either pipeline touches them. The table scores them as `der_turns` (identical
for both pipelines by construction), and `bench/reassign.py` re-runs the
word→speaker rule of `core/align.py` on the saved `words` + `diarization`
with other thresholds, so the speaker-attribution rule is tuned offline:

```bash
python bench/reassign.py bench/results/last --out bench/results/last_maxoverlap --continuity-window 0
python bench/run_bench.py --data bench/data --rescore bench/results/last_maxoverlap --model voxtral:mini-3b-vllm
```

The same loop calibrates the **speaker cluster merge** (`core/speaker_merge.py`:
pyannote clusters whose `pyannote/embedding` vectors are closer than
`VOXHUB_SPEAKER_MERGE_THRESHOLD`, cosine distance, are folded into one). The
space is tight, so do not guess the threshold: a first try at 0.45 merged every
recording down to one speaker. Measured on two HiDock meetings forced to
4 clusters (2026-09-26): clusters of the same voice at 0.10, 0.13, 0.13 (and
0.28 for a short cluster on a degraded phone call), clean clusters of
different people at 0.35, 0.39, 0.43, 0.48. Hence the default of **0.25**
(catches the usual split with 0.1 of margin) and 0.3 as the aggressive
setting (also fixes the phone case, 0.05 from a different-voice pair).
Re-check on your own recordings; you need both kinds of pairs in one matrix,
and the way to get same-voice pairs is to *force* pyannote to over-split with
an exact `num_speakers` above the true count:

```bash
# the matrix is emitted whatever the threshold (VOXHUB_SPEAKER_DISTANCES=true);
# exact over-count hint → pyannote splits real voices
python bench/run_bench.py --data bench/data --extra num_speakers=4 --out bench/results/split4
# every hypothesis now carries the 4x4 diarization_distances; sweep offline,
# --merge-floor 1 so the exact hint does not block the merge during the sweep
for t in 0.15 0.2 0.25 0.3; do
  python bench/reassign.py bench/results/split4 --out bench/results/split4_m$t --merge-threshold $t --merge-floor 1
  python bench/run_bench.py --data bench/data --rescore bench/results/split4_m$t --model voxtral:mini-3b-vllm
done
```

Read the matrices: the same-voice pairs must sit clearly below the
different-voice pairs, the threshold goes in the gap, and a file with the
true speaker count must come out unmerged at that threshold.

Re-checked on 2026-09-27 in the current embedding space (the diarization
pipeline's own model), four recordings forced to 4 clusters and swept from
0.15 to 0.35: every same-voice split merges back by 0.20 (a 27-minute
two-person meeting goes from cpWER 0.455 to 0.051, a two-person recording
from 0.780 to 0.200, the degraded Teams call from 0.676 to 0.465) and no
different-voice pair merges up to 0.35. The 0.25 default sits in the middle
of that gap. Only then set
`VOXHUB_SPEAKER_MERGE_THRESHOLD` in `.env` (it is read through
`docker-compose.yaml`; a variable exported in the shell is not passed to the
container). `num_speakers` / `min_speakers` are a floor the merge never
crosses, so an exact hint disables it for that file.

## 3. Reading the table

* `wordalign` should lower **cpWER** noticeably (short turns and overlaps are
  no longer dropped/trimmed) and lower or keep **WER** (full-context chunks).
  If WER goes *up*, look at `warnings` (aligner fallback) and at the chunk
  logs: a hallucination loop that survived the repetition guard shows up as
  a chunk with a huge word count.
* **DER** is scored on the hypothesis *segments*, not on the raw pyannote
  turns, so it is not identical across pipelines even though both run the
  same diarization: `wordalign` segments hug the words, `legacy` segments
  (and a reference corrected from a legacy draft) span whole turns including
  their pauses. Expect a few points of `der_miss` on `wordalign` from that
  alone; `--collar 0.5` narrows it. A large `der_conf` moves with
  `num_speakers` (see `--speakers-from-ref`) or a pyannote version change,
  and a `der_miss` gap that survives a wide collar means real speech was
  dropped (legacy sanitizer, or a lost chunk).
* Pyannote under-counts speakers on short clips (< 1 min) and over-splits
  when forced above the true count, so pass the hint per file with
  `--speakers-from-ref` rather than a global `--extra num_speakers=N`.
* Beware of reference bias: a `.ref.json` corrected from one pipeline's draft
  inherits its segmentation, its written-French normalisation (`je ne sais
  pas` vs the spoken `je sais pas`) and its omissions. Speech the draft
  dropped (overlaps, short interjections) is usually *not* re-added by the
  corrector, so the other pipeline gets charged insertions for being right.
  Spot-check the biggest insertions by listening at their word timestamps
  before trusting a WER gap.
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

## 5. Choosing the speaker-embedding model

Voice profiles live or die by one property of the embedding space: the same
person recorded twice must be closer than two different people, across
microphones. `pyannote/embedding` (the historical default) fails that on
HiDock meetings: the same speaker on a table mic and on a headset lands
0.36-0.58 apart while a Microsoft Teams synthetic voice lands 0.22 from a real
speaker. `bench/embedding_models.py` measures exactly this on your references
(the human names in `<stem>.ref.json` must be consistent across files):

```bash
docker compose exec voxhub-api python bench/embedding_models.py --data bench/data \
  --model pyannote/embedding \
  --model pyannote/wespeaker-voxceleb-resnet34-LM \
  --model speechbrain/spkrec-ecapa-voxceleb \
  --model diarization --alias MS2=MS --out bench/results/embedding_models.json
```

`diarization` is the model the diarization pipeline clusters with (in
pyannote.audio 4 it lives inside the pipeline repo, reported as
`<pipeline>#embedding`); ECAPA needs `pip install speechbrain` in the
container. Result on 2026-09-27 (six recordings, 17 same-voice pairs, 61 different-voice
pairs): `pyannote/embedding` same ≤ 0.306 vs different ≥ 0.334 (margin
+0.03, and it had already failed on headset/phone recordings in production);
`speechbrain/spkrec-ecapa-voxceleb` 0.295 vs 0.373 (+0.08);
`pyannote/wespeaker-voxceleb-resnet34-LM` and `diarization` (the same
model inside community-1) 0.227 vs 0.369 (+0.14). `diarization` is now the
default: same space for clustering, cluster merge and voice profiles, no
extra model to load. Read the `margin` line: positive means the largest same-voice
distance is below the smallest different-voice one and the matching
threshold goes in between. Two same-voice pairs is not a measurement: feed
it every recording where the same people appear, including headset and
phone ones. `fetch_openhinotes_ref.py` builds a usable `.ref.json` from any
OpenHiNotes transcription whose speakers are named, no text correction
needed for this test (only the speaker segments are used). Then set `VOXHUB_EMBEDDING_MODEL`, and on the
OpenHiNotes side update `EXPECTED_EMBEDDING_MODEL` / `EXPECTED_EMBEDDING_DIM`,
purge the stored profiles and transcription embeddings (admin endpoints) and
re-enrol: vectors from two models are never comparable.

