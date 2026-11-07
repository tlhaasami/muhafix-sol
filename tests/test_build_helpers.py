"""
Unit tests for helpers in src/build_data.py, including a regression test for a
quadratic regular expression found while building the real data pack.
"""

import time

from src.build_data import main_contract_snippet, solidity_minor, split_hash
from src.preprocess import clean_source


def test_main_contract_is_the_last_declared_contract():
    src = "library SafeMath {}\ncontract Base {}\n  abstract contract Mid {}\ncontract Token is Base {\n uint x;\n}\n"
    name, snippet = main_contract_snippet(src)
    assert name == "Token" and snippet.startswith("contract Token")


def test_snippet_regex_is_linear_on_long_blank_runs():
    """A 600 KB block comment becomes one long whitespace run after cleaning. The old
    pattern ^\\s*contract (re.M) re-scanned that run from every line start (quadratic,
    90 s for 200 KB); the fixed pattern must stay fast."""
    src = "/*" + ("x" * 79 + "\n") * 7500 + "*/\ncontract Real { }\n"
    clean = clean_source(src)
    t = time.perf_counter()
    name, _ = main_contract_snippet(clean)
    assert name == "Real"
    assert time.perf_counter() - t < 1.0


def test_solidity_minor_parsing():
    assert solidity_minor("^0.4.24") == "0.4"
    assert solidity_minor(">=0.6.0 <0.8.0") == "0.6"
    assert solidity_minor("none") == "unknown"


def test_split_hash_is_order_independent():
    import pandas as pd

    assert split_hash(pd.Series(["b", "a"])) == split_hash(pd.Series(["a", "b"]))
