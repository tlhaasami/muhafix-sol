"""
Unit tests for src/explain.py with a tiny fake predictor (no trained model needed).
"""

import numpy as np

from src.explain import explain_all, find_blocks
from src.preprocess import tokenize_lines, truncate_tokens

BANK = """pragma solidity ^0.4.24;
contract Bank {
    mapping(address => uint) balances;
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    // a comment with { braces } and function keyword
    function withdraw() public {
        uint amount = balances[msg.sender];
        require(msg.sender.call.value(amount)());
        balances[msg.sender] = 0;
    }
    function () payable {}
    function abstractOne() public;
}
"""


class KeywordPredictor:
    """Probability of 'reentrancy' = share of tokens equal to 'call' (scaled); toy but monotone."""

    classes = ["reentrancy", "other"]

    def line_tokens(self, source):
        lines = tokenize_lines(source)
        flat, owner = [], []
        for i, t in enumerate(lines):
            flat += t
            owner += [i + 1] * len(t)
        kept, _ = truncate_tokens(flat, 10_000)
        return lines, kept, owner

    def _p(self, toks):
        n = max(len(toks), 1)
        return np.array([min(1.0, 20 * toks.count("call") / n), 0.1])

    def removal_probabilities(self, tokens, spans, model=None):
        base = self._p(tokens)
        rem = np.array([self._p(tokens[:a] + tokens[b:]) for a, b in spans]).reshape(len(spans), 2)
        return base, rem


def test_find_blocks_ignores_comments_and_bodiless_declarations():
    blocks = find_blocks(BANK)
    names = [(b.name, b.start_line, b.end_line) for b in blocks]
    assert ("deposit", 4, 6) in names
    assert ("withdraw", 8, 12) in names
    assert ("fallback", 13, 13) in names
    assert all(b.name != "abstractOne" for b in blocks)


def test_explain_points_at_the_call_line():
    out = explain_all(BANK, KeywordPredictor(), top_k=3)
    top = out["lines"]["reentrancy"]
    assert top and top[0]["line"] == 10 and top[0]["function"] == "withdraw"
    assert out["functions"]["reentrancy"][0]["function"] == "withdraw"
    assert all(it["score"] > 0 for it in top)


def test_explain_never_raises_on_odd_input():
    p = KeywordPredictor()
    for junk in ["", "}}}{{{", "function f( {", "\x00" * 10, "contract A {" * 500]:
        out = explain_all(junk, p)
        assert set(out["lines"]) == set(p.classes)
