import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))

from metrics import cpwer, der, normalize_text, wer  # noqa: E402


def test_normalize_text():
    assert normalize_text("Bonjour, c’est l'équipe !") == ["bonjour", "c'est", "l'équipe"]
    assert normalize_text("Élève", strip_accents=True) == ["eleve"]


def test_wer_counts():
    r = wer("le chat dort sur le canapé", "le chat dort sur canapé rouge")
    # Two edits (del "le" + ins "rouge", or two substitutions: same cost)
    assert r["N"] == 6 and r["S"] + r["D"] + r["I"] == 2
    assert abs(r["wer"] - 2 / 6) < 1e-9
    assert wer("a b c", "a b c")["wer"] == 0.0
    assert wer("", "x y")["wer"] == 1.0


def test_cpwer_finds_speaker_permutation():
    ref = [
        {"start": 0, "end": 1, "speaker": "Alice", "text": "bonjour à tous"},
        {"start": 1, "end": 2, "speaker": "Bob", "text": "merci beaucoup"},
    ]
    # Same words, labels swapped: cpWER must be 0 with mapping S1→Alice, S0→Bob
    hyp = [
        {"start": 0, "end": 1, "speaker": "S1", "text": "bonjour à tous"},
        {"start": 1, "end": 2, "speaker": "S0", "text": "merci beaucoup"},
    ]
    r = cpwer(ref, hyp)
    assert r["cpwer"] == 0.0 and r["mapping"] == {"S1": "Alice", "S0": "Bob"}
    # Speaker attribution error: Bob's words given to Alice count twice in
    # cpWER (inserted on Alice's stream, deleted from Bob's) → 4 errors / 5
    hyp2 = [{"start": 0, "end": 2, "speaker": "S0", "text": "bonjour à tous merci beaucoup"}]
    r2 = cpwer(ref, hyp2)
    assert r2["errors"] == 4 and abs(r2["cpwer"] - 4 / 5) < 1e-9


def test_der_perfect_and_confusion():
    ref = [
        {"start": 0.0, "end": 10.0, "speaker": "A"},
        {"start": 10.0, "end": 20.0, "speaker": "B"},
    ]
    hyp_perfect = [
        {"start": 0.0, "end": 10.0, "speaker": "X"},
        {"start": 10.0, "end": 20.0, "speaker": "Y"},
    ]
    r = der(ref, hyp_perfect, collar=0.0)
    assert r["der"] == 0.0 and r["mapping"] == {"X": "A", "Y": "B"}

    # Second half attributed to the first speaker → 50% confusion
    hyp_conf = [{"start": 0.0, "end": 20.0, "speaker": "X"}]
    r = der(ref, hyp_conf, collar=0.0)
    assert abs(r["der"] - 0.5) < 0.01 and abs(r["confusion"] - 0.5) < 0.01

    # Missed speech and false alarm
    hyp_miss = [{"start": 0.0, "end": 5.0, "speaker": "X"}, {"start": 25.0, "end": 30.0, "speaker": "X"}]
    r = der(ref, hyp_miss, collar=0.0)
    assert abs(r["miss"] - 0.75) < 0.01 and abs(r["false_alarm"] - 0.25) < 0.01


def test_der_collar_ignores_boundaries():
    ref = [{"start": 0.0, "end": 10.0, "speaker": "A"}]
    hyp = [{"start": 0.2, "end": 9.8, "speaker": "X"}]  # 0.2 s off at each end
    assert der(ref, hyp, collar=0.0)["der"] > 0.03
    assert der(ref, hyp, collar=0.25)["der"] == 0.0
