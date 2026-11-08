"""
Integration tests against the frozen model (blueprint P3.4, acceptance tests C2, C3, C5).
Skipped automatically when the model files have not been built yet.
"""

import random
import time

import numpy as np
import pytest

from src.config import ROOT, path

MODELS_READY = (path("models_dir") / "thresholds.json").exists()
pytestmark = pytest.mark.skipif(not MODELS_READY, reason="frozen model not built yet")
SAMPLES = sorted((ROOT / "app" / "sample_contracts").glob("*.sol"))
SCHEMA_KEYS = {"probabilities", "flags", "risk_level", "latency_ms"}


@pytest.fixture(scope="module")
def predictor():
    from src.predict import Predictor

    return Predictor()


# ------------------------------------------------------------- C2: inputs --
def test_six_real_contracts_give_sensible_output(predictor):
    assert len(SAMPLES) == 6
    out = {f.name: predictor.predict(f.read_text(encoding="utf-8")) for f in SAMPLES}
    for name, r in out.items():
        assert SCHEMA_KEYS <= set(r) and set(r["probabilities"]) == set(predictor.classes), name
        assert r["risk_level"] in ("low", "medium", "high")
        assert all(0.0 <= p <= 1.0 for p in r["probabilities"].values())
    # the classic reentrancy example must score higher on reentrancy than the clean token
    if "reentrancy" in predictor.classes:
        assert out["02_reentrancy_bank.sol"]["probabilities"]["reentrancy"] > \
            out["01_clean_token.sol"]["probabilities"]["reentrancy"]


@pytest.mark.parametrize("bad, fragment", [
    ("", "empty"),
    ("   \n\t  ", "empty"),
    ("contract A { uint x; }" * 80000, "too large"),
    ("x", "too short"),
])
def test_edge_cases_give_clear_messages(predictor, bad, fragment):
    from src.predict import InputError

    with pytest.raises(InputError, match=fragment):
        predictor.predict(bad)


def test_non_solidity_text_warns_but_does_not_crash(predictor):
    r = predictor.predict("The quick brown fox jumps over the lazy dog. " * 20)
    assert r["warnings"] and "Solidity" in r["warnings"][0]


# ------------------------------------------------------------- fuzzing ----
def test_fuzz_inputs_never_crash(predictor):
    from src.explain import explain_all
    from src.predict import InputError

    rng = random.Random(7)
    alphabet = "abcdefxyz{}();=+-*/<>!&|\"'\\\n\r\t 0123456789.@#$\x00é漢🙂"
    inputs = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 3000))) for _ in range(25)]
    inputs += [bytes(rng.randrange(256) for _ in range(2000)).decode("latin-1"),
               "a" * 200_000,                                      # one extremely long line
               "\n".join(f"uint v{i} = {i};" for i in range(5000))]  # thousands of lines
    for text in inputs:
        try:
            r = predictor.predict(text)
            assert SCHEMA_KEYS <= set(r)
            explain_all(text, predictor)
        except InputError:
            pass


# ------------------------------------------------------- C3: consistency --
def test_app_shows_exactly_the_predict_output(predictor):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.selectbox[0].set_value("02_reentrancy_bank.sol").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    shown = at.session_state["scan"]["result"]
    direct = predictor.predict((ROOT / "app" / "sample_contracts" / "02_reentrancy_bank.sol").read_text("utf-8"))
    assert shown["risk_level"] == direct["risk_level"]
    for c in predictor.classes:
        assert shown["probabilities"][c] == pytest.approx(direct["probabilities"][c], abs=1e-9)


def test_paste_and_upload_paths_are_identical(predictor):
    """F1: the app decodes uploads to the same text that is pasted."""
    src = SAMPLES[1].read_text(encoding="utf-8")
    decoded = src.encode("utf-8").decode("utf-8", errors="replace")
    a, b = predictor.predict(src), predictor.predict(decoded)
    assert a["probabilities"] == b["probabilities"]


def test_sensitivity_slider_is_monotone(predictor):
    """F10: lower scale -> at least as many flags as a higher scale."""
    src = SAMPLES[2].read_text(encoding="utf-8")
    counts = [sum(predictor.predict(src, threshold_scale=s)["flags"].values()) for s in (0.5, 0.75, 1.0, 1.25, 1.5)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_both_app_models_work(predictor):
    src = SAMPLES[0].read_text(encoding="utf-8")
    for m in predictor.available_models:
        assert set(predictor.predict(src, model=m)["probabilities"]) == set(predictor.classes)


def test_similar_returns_three_neighbours(predictor):
    sim = predictor.similar(SAMPLES[1].read_text(encoding="utf-8"), k=3)
    assert len(sim) == 3 and all(-1.0 <= s["similarity"] <= 1.0001 for s in sim)
    assert sim[0]["similarity"] >= sim[-1]["similarity"]


def test_fast_occlusion_matches_full_revectorisation(predictor):
    """The incremental count trick must equal re-vectorising the shortened text."""
    _, toks, owner = predictor.line_tokens(SAMPLES[1].read_text(encoding="utf-8"))
    spans = [(5, 12), (20, 21), (0, 3), (len(toks) - 4, len(toks))]
    base, rem = predictor.removal_probabilities(toks, spans)
    full = predictor.proba_docs([" ".join(toks)] + [" ".join(toks[:a] + toks[b:]) for a, b in spans])
    assert np.allclose(base, full[0], atol=1e-6) and np.allclose(rem, full[1:], atol=1e-6)


# ---------------------------------------------------------- C5: latency --
def test_latency_is_a_few_seconds(predictor):
    from src.explain import explain_all

    src = SAMPLES[1].read_text(encoding="utf-8")
    t = time.perf_counter()
    predictor.predict(src)
    explain_all(src, predictor)
    assert time.perf_counter() - t < 5.0
