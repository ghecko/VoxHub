#!/usr/bin/env python3
"""
VoxHub benchmark harness: WER / cpWER / DER / speed per pipeline and model.

Usage
-----
    # Compare the two pipelines on every audio in ./data with the default model
    python bench/run_bench.py --api http://localhost:8000 --data bench/data

    # Compare models too (cartesian product with pipelines)
    python bench/run_bench.py --data bench/data \\
        --pipeline legacy --pipeline wordalign \\
        --model voxtral:mini-3b-vllm --model whisper:large-v3

    # Re-score saved hypotheses without calling the API
    python bench/run_bench.py --data bench/data --rescore bench/results/last

Data layout (one stem per recording, any subset of the references)
----------------------------------------------------------------
    bench/data/
      meeting1.mp3
      meeting1.rttm        diarization reference (NIST RTTM)            → DER
      meeting1.ref.json    [{"start","end","speaker","text"}, ...]      → cpWER (+ DER if no .rttm)
      meeting1.txt         plain reference transcript                   → WER

Hypotheses are saved under ``--out`` as ``<stem>.<pipeline>.<model>.json``
(the raw verbose_json) so they can be re-scored or inspected later.

Adding a backend (e.g. VibeVoice-ASR)
-------------------------------------
Register it in ``models.yaml`` and pass ``--model vibevoice:...``. A joint
STT+diarization backend should return segments that already carry a
``speaker``; run it with ``--pipeline legacy --extra vad_mode=none`` (or add a
whole-file branch in ``api/transcriber.py``) and this harness scores it like
any other hypothesis, so the comparison with Voxtral+pyannote is apples to
apples: same audio, same references, same metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cpwer, der, read_rttm, wer  # noqa: E402

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".hda", ".webm")


def _post(api: str, api_key: Optional[str], audio_path: str, form: Dict[str, str]) -> Dict:
    import urllib.request
    import uuid

    boundary = f"----voxhubbench{uuid.uuid4().hex}"
    body = bytearray()
    for k, v in form.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    with open(audio_path, "rb") as f:
        data = f.read()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{os.path.basename(audio_path)}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{api.rstrip('/')}/v1/audio/transcriptions", data=bytes(body), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read().decode())


def _load_refs(stem: str):
    rttm = f"{stem}.rttm"
    refjson = f"{stem}.ref.json"
    txt = f"{stem}.txt"
    ref_segments = None
    if os.path.exists(refjson):
        with open(refjson, encoding="utf-8") as f:
            ref_segments = json.load(f)
    diar_ref = read_rttm(rttm) if os.path.exists(rttm) else ref_segments
    text_ref = None
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            text_ref = f.read()
    elif ref_segments:
        text_ref = " ".join(s.get("text", "") for s in sorted(ref_segments, key=lambda s: s["start"]))
    return ref_segments, diar_ref, text_ref


def score(hyp: Dict, ref_segments, diar_ref, text_ref, collar: float) -> Dict:
    hyp_segments = hyp.get("segments") or []
    out: Dict[str, object] = {
        "n_segments": len(hyp_segments),
        "n_speakers": len({s.get("speaker") for s in hyp_segments}),
    }
    if text_ref is not None:
        out["wer"] = round(wer(text_ref, hyp.get("text", ""))["wer"], 4)
    if ref_segments:
        try:
            out["cpwer"] = round(cpwer(ref_segments, hyp_segments)["cpwer"], 4)
        except ValueError as e:
            out["cpwer"] = str(e)
    if diar_ref:
        try:
            d = der(diar_ref, hyp_segments, collar=collar)
            out["der"] = round(d["der"], 4)
            out["der_miss"] = round(d["miss"], 4)
            out["der_fa"] = round(d["false_alarm"], 4)
            out["der_conf"] = round(d["confusion"], 4)
        except ValueError as e:
            out["der"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.environ.get("VOXHUB_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("VOXHUB_API_KEY"))
    ap.add_argument("--data", required=True, help="Folder with audio files and references")
    ap.add_argument("--pipeline", action="append", help="legacy | wordalign (repeatable; default both)")
    ap.add_argument("--model", action="append", help="Model spec (repeatable; default server default)")
    ap.add_argument("--language", default=None)
    ap.add_argument("--extra", action="append", default=[], help="Extra form field key=value (repeatable)")
    ap.add_argument("--collar", type=float, default=0.25, help="DER collar in seconds")
    ap.add_argument("--out", default="bench/results/last", help="Where hypotheses + results.json go")
    ap.add_argument("--rescore", default=None, help="Re-score saved hypotheses from this folder instead of calling the API")
    args = ap.parse_args()

    pipelines = args.pipeline or ["legacy", "wordalign"]
    models = args.model or [None]
    extra = dict(kv.split("=", 1) for kv in args.extra)
    os.makedirs(args.out, exist_ok=True)

    audios = sorted(p for p in glob.glob(os.path.join(args.data, "*")) if p.lower().endswith(AUDIO_EXT))
    if not audios:
        print(f"No audio files in {args.data}", file=sys.stderr)
        return 1

    rows: List[Dict] = []
    for audio in audios:
        stem = os.path.splitext(audio)[0]
        ref_segments, diar_ref, text_ref = _load_refs(stem)
        if not (ref_segments or diar_ref or text_ref):
            print(f"[skip] {os.path.basename(audio)}: no reference (.rttm / .ref.json / .txt)")
            continue
        for pipeline in pipelines:
            for model in models:
                tag = f"{os.path.basename(stem)}.{pipeline}.{(model or 'default').replace(':', '_').replace('/', '_')}"
                hyp_path = os.path.join(args.rescore or args.out, tag + ".json")
                if args.rescore:
                    if not os.path.exists(hyp_path):
                        print(f"[skip] {tag}: no saved hypothesis")
                        continue
                    with open(hyp_path, encoding="utf-8") as f:
                        hyp = json.load(f)
                    elapsed = hyp.get("_bench", {}).get("elapsed_s")
                else:
                    form = {
                        "response_format": "verbose_json",
                        "diarize": "true",
                        "pipeline": pipeline,
                        "timestamp_granularities[]": "word",
                        **extra,
                    }
                    if model:
                        form["model"] = model
                    if args.language:
                        form["language"] = args.language
                    print(f"[run ] {tag} ...", end="", flush=True)
                    t0 = time.time()
                    try:
                        hyp = _post(args.api, args.api_key, audio, form)
                    except Exception as e:
                        print(f" FAILED: {e}")
                        rows.append({"file": os.path.basename(audio), "pipeline": pipeline, "model": model or "default", "error": str(e)})
                        continue
                    elapsed = round(time.time() - t0, 1)
                    hyp["_bench"] = {"elapsed_s": elapsed, "audio": os.path.basename(audio), "model": model, "pipeline": pipeline}
                    with open(hyp_path, "w", encoding="utf-8") as f:
                        json.dump(hyp, f, ensure_ascii=False, indent=1)
                    print(f" {elapsed}s")
                row = {"file": os.path.basename(audio), "pipeline": hyp.get("pipeline", pipeline), "model": model or "default"}
                row.update(score(hyp, ref_segments, diar_ref, text_ref, args.collar))
                dur = hyp.get("duration")
                if elapsed and dur:
                    row["elapsed_s"] = elapsed
                    row["rtf"] = round(elapsed / dur, 3)
                if hyp.get("warnings"):
                    row["warnings"] = "; ".join(hyp["warnings"])
                rows.append(row)

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    cols = ["file", "pipeline", "model", "wer", "cpwer", "der", "der_miss", "der_fa", "der_conf", "n_speakers", "n_segments", "rtf", "elapsed_s"]
    present = [c for c in cols if any(c in r for r in rows)]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in present}
    print()
    print("  ".join(c.ljust(widths[c]) for c in present))
    print("  ".join("-" * widths[c] for c in present))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in present))
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} run(s) failed, see results.json")
    print(f"\nHypotheses and results.json written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
