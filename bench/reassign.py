#!/usr/bin/env python3
"""
Re-run the word→speaker rule offline on saved wordalign hypotheses.

A wordalign ``verbose_json`` saved by ``run_bench.py`` carries the aligned
``words`` and, since the ``diarization`` field exists, the raw pyannote
turns. That is everything ``assign_word_speakers`` needs, so a change to the
rule (or to its thresholds) can be scored without touching the GPU:

    python bench/reassign.py bench/results/20260927 --out bench/results/20260927_cont0 \\
        --continuity-window 0
    python bench/run_bench.py --data bench/data --rescore bench/results/20260927_cont0 \\
        --model voxtral:mini-3b-vllm

Only ``*.wordalign.*.json`` files with both ``words`` and ``diarization`` are
rewritten; the others are copied unchanged so ``--rescore`` still sees the
whole run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.align import assign_word_speakers, words_to_segments  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Folder with saved hypotheses (a run_bench --out)")
    ap.add_argument("--out", required=True, help="Folder for the rewritten hypotheses")
    ap.add_argument("--max-gap", type=float, default=1.0)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--ambiguity", type=float, default=0.5)
    ap.add_argument("--continuity-window", type=float, default=2.0, help="0 = plain max-overlap rule")
    ap.add_argument(
        "--merge-floor", type=int, default=None,
        help="Never merge below this many speakers (default: the run's num_speakers/min_speakers hint; "
        "pass 1 to sweep freely on a run that forced num_speakers to over-split)",
    )
    ap.add_argument(
        "--merge-threshold", type=float, default=None,
        help="Re-plan the speaker cluster merge at this cosine distance from the saved "
        "diarization_distances (run the server with VOXHUB_SPEAKER_MERGE_THRESHOLD=0 to "
        "get unmerged turns + the full matrix, then sweep this offline)",
    )
    ap.add_argument("--max-pause", type=float, default=1.0, help="words_to_segments: silence that starts a segment")
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_done = 0
    for path in sorted(glob.glob(os.path.join(args.run, "*.json"))):
        name = os.path.basename(path)
        dst = os.path.join(args.out, name)
        with open(path, encoding="utf-8") as f:
            hyp = json.load(f)
        if not isinstance(hyp, dict):  # results.json is a list of rows
            shutil.copyfile(path, dst)
            continue
        words = hyp.get("words") or [w for s in hyp.get("segments", []) for w in (s.get("words") or [])]
        turns = hyp.get("diarization")
        if hyp.get("pipeline") != "wordalign" or not words or not turns:
            shutil.copyfile(path, dst)
            continue
        if args.merge_threshold is not None and hyp.get("diarization_distances"):
            from core.speaker_merge import plan_cluster_merges
            durations: dict = {}
            for t in turns:
                durations[t["speaker"]] = durations.get(t["speaker"], 0.0) + (t["end"] - t["start"])
            form = hyp.get("_bench", {}).get("form") or {}
            floor = args.merge_floor if args.merge_floor is not None else (
                form.get("num_speakers") or form.get("min_speakers"))
            mapping = plan_cluster_merges(
                hyp["diarization_distances"], dict(durations), args.merge_threshold,
                int(floor) if floor else None,
            )
            turns = [{**t, "speaker": mapping.get(t["speaker"], t["speaker"])} for t in turns]
            hyp["diarization"] = turns
            hyp["diarization_merges"] = [
                {"from": a, "into": b, "distance": hyp["diarization_distances"][a][b]}
                for a, b in sorted(mapping.items())
            ]
        words = sorted(({k: w[k] for k in ("word", "start", "end", "score") if k in w} for w in words),
                       key=lambda w: w["start"])
        words = assign_word_speakers(
            words, turns,
            max_gap=args.max_gap, min_overlap=args.min_overlap,
            ambiguity=args.ambiguity, continuity_window=args.continuity_window,
        )
        segments = words_to_segments(words, max_pause=args.max_pause, max_duration=args.max_duration)
        hyp["segments"] = [
            {"id": i, "start": s["start"], "end": s["end"], "text": s["text"], "speaker": s["speaker"],
             "confidence": s.get("confidence", 0.0), "words": s["words"]}
            for i, s in enumerate(segments)
        ]
        hyp["words"] = [w for s in hyp["segments"] for w in s["words"]]
        hyp["text"] = " ".join(s["text"] for s in hyp["segments"]).strip()
        hyp.setdefault("_bench", {})["reassign"] = vars(args)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(hyp, f, ensure_ascii=False, indent=1)
        n_done += 1
        print(f"[reassign] {name}: {len(words)} words → {len(segments)} segments")
    print(f"{n_done} hypothesis file(s) rewritten into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
