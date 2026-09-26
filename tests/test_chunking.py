import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chunking import build_chunks, split_chunk  # noqa: E402


def _regions(pairs):
    return [{"start": s, "end": e} for s, e in pairs]


def test_empty():
    assert build_chunks([], 10.0) == []


def test_single_short_region_is_one_chunk():
    chunks = build_chunks(_regions([(1.0, 5.0)]), 10.0, edge_pad=0.0)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.75 and chunks[0]["end"] == 5.25
    assert chunks[0]["index"] == 0


def test_edge_pad_reaches_file_edges_only_when_close():
    # First region starts 2 s in, last one ends 3 s before the end: both
    # chunks are pulled to the file edges (Silero missed a short utterance).
    chunks = build_chunks(_regions([(2.0, 5.0)]), 8.0, edge_pad=5.0)
    assert chunks[0]["start"] == 0.0 and chunks[0]["end"] == 8.0
    # 20 s of leading silence is more than edge_pad: left alone.
    chunks = build_chunks(_regions([(20.0, 25.0)]), 60.0, edge_pad=5.0)
    assert chunks[0]["start"] == 19.75 and chunks[0]["end"] == 25.25


def test_cuts_at_longest_silence_near_target():
    # 12 regions of 9 s separated by 1 s silences, except a 4 s silence
    # after the 6th region (ending at 59 s).
    pairs, t = [], 0.0
    for k in range(12):
        pairs.append((t, t + 9.0))
        t += 9.0 + (4.0 if k == 5 else 1.0)
    chunks = build_chunks(_regions(pairs), t, target_duration=60, max_duration=180, min_duration=8)
    assert len(chunks) == 2
    # First chunk ends at the 4 s silence (region 6 ends at 5*10+9 = 59)
    assert abs(chunks[0]["end"] - (59.0 + 0.25)) < 1e-6
    # No overlap between chunks and padding never crosses the neighbour
    assert chunks[0]["end"] <= chunks[1]["start"]
    assert chunks[-1]["end"] <= t


def test_hard_ceiling_when_no_silence():
    chunks = build_chunks(_regions([(0.0, 400.0)]), 400.0, target_duration=60, max_duration=180)
    assert len(chunks) == 3
    for c in chunks:
        assert c["end"] - c["start"] <= 180.0 + 0.5
    assert chunks[-1]["end"] == 400.0


def test_tail_orphan_is_merged():
    pairs = [(0, 30), (31, 59), (60, 62)]  # 2 s orphan at the end
    chunks = build_chunks(_regions(pairs), 62.0, target_duration=60, min_duration=8)
    assert len(chunks) == 1
    assert chunks[0]["end"] == 62.0


def test_chunks_are_sorted_and_cover_all_speech():
    pairs = [(i * 7.0, i * 7.0 + 5.0) for i in range(40)]  # 280 s of audio
    chunks = build_chunks(_regions(pairs), 280.0)
    assert [c["index"] for c in chunks] == list(range(len(chunks)))
    for a, b in zip(chunks, chunks[1:]):
        assert a["end"] <= b["start"]
    for s, e in pairs:
        assert any(c["start"] <= s and c["end"] >= e for c in chunks), (s, e)


def test_split_chunk_prefers_internal_silence():
    regions = _regions([(0, 20), (23, 40)])
    chunk = {"start": 0.0, "end": 40.0, "index": 0}
    parts = split_chunk(chunk, regions)
    assert len(parts) == 2
    assert parts[0]["end"] == 21.5 and parts[1]["start"] == 21.5


def test_split_chunk_midpoint_fallback_and_too_short():
    chunk = {"start": 10.0, "end": 30.0}
    parts = split_chunk(chunk, [])
    assert [p["start"] for p in parts] == [10.0, 20.0]
    assert split_chunk({"start": 0.0, "end": 5.0}, []) == [{"start": 0.0, "end": 5.0}]
