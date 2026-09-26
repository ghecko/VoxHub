import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.align import (  # noqa: E402
    _interpolate_missing,
    _merge_alignment,
    assign_word_speakers,
    normalize_word,
    split_words,
    uniform_word_times,
    words_to_segments,
)

ASCII_VOCAB = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'")}


def test_normalize_word_ascii_vocab_strips_accents_and_punct():
    assert normalize_word("Élève,", ASCII_VOCAB, True, True) == "eleve"
    assert normalize_word("cœur…", ASCII_VOCAB, True, True) == "coeur"
    assert normalize_word("l’homme", ASCII_VOCAB, True, True) == "l'homme"
    assert normalize_word("2026", ASCII_VOCAB, True, True) == ""
    assert normalize_word("(ok)", ASCII_VOCAB, True, True) == "ok"


def test_normalize_word_accented_vocab_keeps_accents():
    vocab = dict(ASCII_VOCAB)
    vocab.update({"é": 40, "è": 41})
    assert normalize_word("élève", vocab, True, False) == "élève"
    assert normalize_word("çà", vocab, True, False) == "ca"  # ç, à not in vocab → stripped


def test_merge_alignment_collapses_runs():
    # frames: b b 0 a a a 0 0 a  (blank=0) for tokens [b, a, a]
    ali = [2, 2, 0, 1, 1, 1, 0, 0, 1]
    probs = [0.9, 0.8, 1, 0.5, 0.6, 0.7, 1, 1, 0.4]
    spans = _merge_alignment(ali, probs, [2, 1, 1], blank=0)
    assert spans[0][:2] == (0, 1)
    assert spans[1][:2] == (3, 5)
    assert spans[2][:2] == (8, 8)
    assert abs(spans[1][2] - 0.6) < 1e-9


def test_interpolate_missing_fills_from_neighbours():
    words = [
        {"word": "a", "start": 0.0, "end": 1.0, "score": 0.9},
        {"word": "2026", "start": None, "end": None, "score": 0.0},
        {"word": "42", "start": None, "end": None, "score": 0.0},
        {"word": "b", "start": 3.0, "end": 4.0, "score": 0.9},
        {"word": "…", "start": None, "end": None, "score": 0.0},
    ]
    _interpolate_missing(words, 0.0, 5.0)
    assert (words[1]["start"], words[1]["end"]) == (1.0, 2.0)
    assert (words[2]["start"], words[2]["end"]) == (2.0, 3.0)
    assert (words[4]["start"], words[4]["end"]) == (4.0, 5.0)


def test_uniform_word_times():
    w = uniform_word_times(["a", "b", "c", "d"], 10.0, 12.0)
    assert [x["start"] for x in w] == [10.0, 10.5, 11.0, 11.5]
    assert w[-1]["end"] == 12.0


def _w(word, s, e, score=0.9):
    return {"word": word, "start": s, "end": e, "score": score}


def test_assign_word_speakers_max_overlap_and_fallbacks():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 4.5, "end": 10.0, "speaker": "B"},   # overlaps A on [4.5, 5]
        {"start": 20.0, "end": 25.0, "speaker": "A"},
    ]
    words = [
        _w("hello", 0.5, 1.0),        # A
        _w("yes", 4.6, 4.9),          # inside both turns → ambiguous; "hello" is
                                      # out of the continuity window, the next
                                      # confident word ("right", B) decides → B
        _w("right", 4.7, 5.5),        # overlap A=0.3, B=0.8 → confident B
        _w("gap", 10.4, 10.8),        # no overlap, 0.4 s after B → B (nearest within 1 s)
        _w("far", 14.0, 14.5),        # no turn within 1 s → inherits previous (B)
        _w("back", 20.5, 21.0),       # A
    ]
    out = assign_word_speakers(words, turns, max_gap=1.0)
    assert [w["speaker"] for w in out] == ["A", "B", "B", "B", "B", "A"]


def test_assign_word_speakers_boundary_and_overlap_follow_continuity():
    # A talks 0-10, B interjects 3-4 (overlapping) and then takes over at 10,
    # but pyannote put the boundary 0.3 s late.
    turns = [
        {"start": 0.0, "end": 10.3, "speaker": "A"},
        {"start": 3.0, "end": 4.0, "speaker": "B"},
        {"start": 10.0, "end": 15.0, "speaker": "B"},
    ]
    words = [
        _w("a1", 1.0, 1.5),          # confident A
        _w("x", 3.1, 3.4),           # inside A and B → ambiguous → previous resolved (A)
        _w("y", 3.5, 3.9),           # ambiguous → previous resolved word "x" (A)
        _w("a2", 5.0, 5.5),          # confident A
        _w("b1", 9.9, 10.5),         # A covers 0.4/0.6, B 0.5/0.6 → both ≥ 0.5 → ambiguous;
                                     # previous resolved is "a2" (A) but 4.4 s away,
                                     # next confident "b2" (B) → B
        _w("b2", 11.0, 11.5),        # confident B
    ]
    out = assign_word_speakers(words, turns, max_gap=1.0)
    assert [w["speaker"] for w in out] == ["A", "A", "A", "A", "B", "B"]
    # Continuity never invents a speaker: with the window at 0 the
    # max-overlap rule is back (B wins "b1" by 0.1 s, "x"/"y" tie → A first).
    out = assign_word_speakers(words, turns, max_gap=1.0, continuity_window=0.0)
    assert [w["speaker"] for w in out] == ["A", "A", "A", "A", "B", "B"]


def test_assign_word_speakers_split_turns_are_summed_per_speaker():
    # A's turn split in two by pyannote around the word; B has one turn
    # covering it. Per-turn max-overlap would give B; per-speaker gives A.
    turns = [
        {"start": 0.0, "end": 1.2, "speaker": "A"},
        {"start": 1.2, "end": 3.0, "speaker": "A"},
        {"start": 0.9, "end": 1.5, "speaker": "B"},
    ]
    out = assign_word_speakers([_w("w", 1.0, 1.4)], turns, continuity_window=0.0)
    assert out[0]["speaker"] == "A"


def test_assign_word_speakers_leading_hole_takes_next_and_no_turns_default():
    turns = [{"start": 10.0, "end": 12.0, "speaker": "Z"}]
    out = assign_word_speakers([_w("x", 0.0, 0.5), _w("y", 10.5, 11.0)], turns)
    assert [w["speaker"] for w in out] == ["Z", "Z"]
    out = assign_word_speakers([_w("x", 0.0, 0.5)], [], default_speaker="S0")
    assert out[0]["speaker"] == "S0"


def test_words_to_segments_breaks_on_speaker_pause_and_duration():
    words = [
        {**_w("Bonjour", 0.0, 0.5), "speaker": "A"},
        {**_w("à", 0.5, 0.6), "speaker": "A"},
        {**_w("tous.", 0.6, 1.0), "speaker": "A"},
        {**_w("Merci.", 1.2, 1.6), "speaker": "B"},       # speaker change
        {**_w("Donc", 3.5, 3.8), "speaker": "B"},         # pause 1.9 s > 1.0
        {**_w("voilà", 3.8, 4.2), "speaker": "B"},
    ]
    segs = words_to_segments(words, max_pause=1.0)
    assert [(s["speaker"], s["text"]) for s in segs] == [
        ("A", "Bonjour à tous."),
        ("B", "Merci."),
        ("B", "Donc voilà"),
    ]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.0
    assert segs[0]["confidence"] == 0.9
    assert [s["id"] for s in segs] == [0, 1, 2]
    assert len(segs[0]["words"]) == 3

    # Hard ceiling: 40 words of 1 s each from one speaker without pause → split at 30 s
    long_words = [{**_w(f"w{i}", float(i), i + 1.0), "speaker": "A"} for i in range(40)]
    segs = words_to_segments(long_words, max_pause=1.0, max_duration=30.0, soft_duration=15.0)
    assert len(segs) == 2
    assert all(s["end"] - s["start"] <= 31.0 for s in segs)

    # Soft ceiling: sentence end after 15 s triggers a break
    soft = [{**_w("w", float(i), i + 1.0), "speaker": "A"} for i in range(20)]
    soft[16]["word"] = "fin."
    segs = words_to_segments(soft, max_pause=1.0, max_duration=30.0, soft_duration=15.0)
    assert len(segs) == 2 and segs[0]["words"][-1]["word"] == "fin."


def test_split_words_glues_french_punctuation_to_previous_word():
    # "c'est ça ?" must not yield a standalone "?" that could drift to the
    # next speaker's segment once interpolated.
    assert split_words("Tu as pris deux semaines, c'est ça ? Trois semaines même !") == [
        "Tu", "as", "pris", "deux", "semaines,", "c'est", "ça ?", "Trois", "semaines", "même !",
    ]
    assert split_words("... Oui") == ["... Oui"]
    assert split_words("?") == ["?"]
    assert split_words("") == []
    # normalize_word still sees only the letters
    assert normalize_word("ça ?", ASCII_VOCAB, True, True) == "ca"
