import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.speaker_merge import pairwise_cosine_distances, plan_cluster_merges  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_pairwise_cosine_distances_symmetric_no_diagonal():
    d = pairwise_cosine_distances({"A": _unit([1, 0]), "B": _unit([0, 1]), "C": _unit([1, 0.1])})
    assert set(d) == {"A", "B", "C"} and "A" not in d["A"]
    assert abs(d["A"]["B"] - 1.0) < 1e-6 and d["A"]["B"] == d["B"]["A"]
    assert d["A"]["C"] < 0.01


def test_plan_merges_closest_pair_first_and_keeps_longer_label():
    # A and C are the same voice (0.2 apart), B is someone else.
    dist = {"A": {"B": 0.8, "C": 0.2}, "B": {"A": 0.8, "C": 0.7}, "C": {"A": 0.2, "B": 0.7}}
    dur = {"A": 30.0, "B": 40.0, "C": 50.0}
    assert plan_cluster_merges(dist, dict(dur), threshold=0.45) == {"A": "C"}
    # Below threshold nothing happens; threshold 0 disables.
    assert plan_cluster_merges(dist, dict(dur), threshold=0.1) == {}
    assert plan_cluster_merges(dist, dict(dur), threshold=0.0) == {}


def test_plan_merges_chains_resolve_to_final_survivor():
    # A~B and B~C: after A→B the merged cluster is still close to C, and C
    # has the most speech, so everything ends up under C.
    dist = {"A": {"B": 0.1, "C": 0.3}, "B": {"A": 0.1, "C": 0.3}, "C": {"A": 0.3, "B": 0.3}}
    dur = {"A": 10.0, "B": 20.0, "C": 60.0}
    assert plan_cluster_merges(dist, dict(dur), threshold=0.45) == {"A": "C", "B": "C"}


def test_plan_merges_respects_min_speakers_floor():
    dist = {"A": {"B": 0.1, "C": 0.1}, "B": {"A": 0.1, "C": 0.1}, "C": {"A": 0.1, "B": 0.1}}
    dur = {"A": 10.0, "B": 20.0, "C": 30.0}
    out = plan_cluster_merges(dist, dict(dur), threshold=0.45, min_speakers=2)
    assert len(out) == 1  # three labels, floor two: exactly one merge
    assert plan_cluster_merges(dist, dict(dur), threshold=0.45, min_speakers=3) == {}
