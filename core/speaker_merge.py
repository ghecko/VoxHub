"""
Merge diarization clusters that are the same voice.

pyannote's most common failure on meetings is to split one person in two
(or more) clusters, which doubles the speaker count and, downstream, leaves
half of that person's speech unmatched to their voice profile. The
decision logic here is pure numpy (unit-testable without torch); the
embeddings come from :func:`core.embeddings.cluster_embeddings`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def pairwise_cosine_distances(embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """``{a: {b: 1 - cos(a, b)}}`` for every pair of labels (symmetric, no diagonal)."""
    labels = sorted(embeddings)
    out: Dict[str, Dict[str, float]] = {a: {} for a in labels}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            d = float(1.0 - np.dot(embeddings[a], embeddings[b]))
            out[a][b] = d
            out[b][a] = d
    return out


def plan_cluster_merges(
    distances: Dict[str, Dict[str, float]],
    durations: Dict[str, float],
    threshold: float,
    min_speakers: Optional[int] = None,
) -> Dict[str, str]:
    """Decide which speaker labels to fold into which.

    Greedy agglomerative: repeatedly merge the closest pair below
    ``threshold`` (average linkage on the stored distances), never going
    below ``min_speakers`` labels. The label with more speech survives.
    Returns ``{absorbed_label: surviving_label}`` with chains already
    resolved, so it can be applied with a single dict lookup per turn.
    """
    if not distances or threshold <= 0:
        return {}
    alive = {a: {a} for a in distances}          # surviving label -> member labels
    dist = {a: dict(v) for a, v in distances.items()}
    floor = max(int(min_speakers or 0), 1)
    mapping: Dict[str, str] = {}
    while len(alive) > floor:
        best = None
        for a in alive:
            for b, d in dist[a].items():
                if b in alive and a < b and d < threshold and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        keep, drop = (a, b) if durations.get(a, 0.0) >= durations.get(b, 0.0) else (b, a)
        # Average-linkage update: distance from the merged cluster to any
        # other one is the size-weighted mean of the two members' distances.
        na, nb = len(alive[keep]), len(alive[drop])
        for other in alive:
            if other in (keep, drop):
                continue
            d = (dist[keep][other] * na + dist[drop][other] * nb) / (na + nb)
            dist[keep][other] = d
            dist[other][keep] = d
        alive[keep] |= alive.pop(drop)
        durations[keep] = durations.get(keep, 0.0) + durations.get(drop, 0.0)
        for m in alive[keep]:
            if m != keep:
                mapping[m] = keep
    return mapping


def merge_speaker_clusters(
    audio: np.ndarray,
    turns: List[Dict],
    threshold: float,
    sample_rate: int = 16000,
    min_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Dict:
    """Merge diarization clusters that are the same voice.

    pyannote's most common failure on meetings is to split one person in
    two (or more) clusters, which doubles the speaker count and, for the
    consumer, leaves half of that person's speech unmatched to their voice
    profile. Clusters whose ``pyannote/embedding`` vectors are closer than
    ``threshold`` (cosine distance) are relabelled to the one with more
    speech. ``min_speakers`` (or an exact ``num_speakers``) is a floor the
    merge never crosses.

    Returns ``{"turns": relabelled turns, "merges": [{from, into, distance}],
    "distances": pairwise matrix}``; the matrix is what you look at to set
    the threshold for your recordings (``diarization_distances`` in the
    verbose_json output).
    """
    labels = {t.get("speaker") for t in turns if t.get("speaker")}
    if len(labels) < 2:
        return {"turns": turns, "merges": [], "distances": {}}
    from core.embeddings import cluster_embeddings  # torch, loaded lazily

    embeddings = cluster_embeddings(audio, turns, sample_rate, hf_token)
    if len(embeddings) < 2:
        return {"turns": turns, "merges": [], "distances": {}}
    distances = pairwise_cosine_distances(embeddings)
    durations: Dict[str, float] = {}
    for t in turns:
        if t.get("speaker"):
            durations[t["speaker"]] = durations.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    # Labels too short to embed still count towards the floor.
    floor = (min_speakers or 0) - (len(labels) - len(embeddings))
    # threshold <= 0: measure only (the matrix is what calibrates the threshold).
    mapping = plan_cluster_merges(distances, dict(durations), threshold, floor) if threshold > 0 else {}
    merges = [
        {"from": a, "into": b, "distance": round(distances[a][b], 3)}
        for a, b in sorted(mapping.items())
    ]
    if not mapping:
        return {"turns": turns, "merges": [], "distances": distances}
    merged = [{**t, "speaker": mapping.get(t.get("speaker"), t.get("speaker"))} for t in turns]
    return {"turns": merged, "merges": merges, "distances": distances}

