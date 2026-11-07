"""
Unit tests for src/preprocess.py (blueprint P1.4, acceptance test A3).
"""

import random
import time

import pytest

from src.preprocess import (
    PROTECTED,
    clean_source,
    count_lines,
    normalise,
    normalise_identifiers,
    prepare_tokens,
    split_lines,
    tokenize,
    tokenize_lines,
    truncate_tokens,
)


def _same_layout(a: str, b: str) -> bool:
    """Same length, and newlines at exactly the same positions."""
    return len(a) == len(b) and all(
        (x in "\r\n") == (y in "\r\n") and (x not in "\r\n" or x == y) for x, y in zip(a, b)
    )


# ----------------------------------------------------------------- cleaning --
def test_empty_input():
    assert clean_source("") == ""
    assert normalise("") == ""
    assert tokenize("") == []
    assert prepare_tokens("") == []
    assert truncate_tokens([], 10) == ([], False)


def test_line_comment_blanked_and_layout_kept():
    code = "uint x = 1; // a line comment\nuint y = 2;"
    out = clean_source(code)
    assert "line comment" not in out and "uint x = 1;" in out and "uint y = 2;" in out
    assert _same_layout(code, out)


def test_block_comment_blanked_and_layout_kept():
    code = "/* multi-line\n   comment here\n*/\ncontract Token {}"
    out = clean_source(code)
    assert "multi-line" not in out and "contract Token {}" in out
    assert _same_layout(code, out)


def test_comment_markers_inside_double_quoted_string_are_kept():
    code = 'string s = "http://x.com/*not a comment*/"; // real comment\n'
    out = clean_source(code)
    assert "http://x.com/*not a comment*/" in out and "real comment" not in out


def test_comment_markers_inside_single_quoted_string_are_kept():
    out = clean_source("bytes b = 'test // not a comment';\n")
    assert "test // not a comment" in out


def test_escaped_quote_inside_string():
    code = 'string s = "say \\"hi\\" // still string"; // gone\n'
    out = clean_source(code)
    assert '// still string"' in out and "gone" not in out


def test_unterminated_string_stops_at_end_of_line():
    code = 'string s = "oops\n// this is a real comment\nuint a;'
    out = clean_source(code)
    assert "real comment" not in out and "uint a;" in out
    assert count_lines(out) == count_lines(code)


def test_unterminated_block_comment_runs_to_eof():
    code = "uint a;\n/* never closed\nuint b;"
    out = clean_source(code)
    assert "uint a;" in out and "uint b;" not in out and _same_layout(code, out)


def test_nested_looking_comment_markers():
    # Solidity block comments do not nest: the first */ closes the comment.
    out = clean_source("/* outer /* inner */ code_after */\nuint z = 5;")
    assert "outer" not in out and "code_after" in out and "uint z = 5;" in out


def test_windows_crlf_and_old_mac_line_endings():
    code = "contract A {\r\n    // comment\r\n    uint256 a;\r}\r\n"
    out = clean_source(code)
    assert "comment" not in out and _same_layout(code, out)
    assert count_lines(out) == count_lines(code) == 5


def test_unicode_preserved_in_strings_removed_in_comments():
    code = "// تبصرہ\nstring name = 'محفوظ';\n"
    out = clean_source(code)
    assert "تبصرہ" not in out and "محفوظ" in out and _same_layout(code, out)


def test_smartbugs_annotation_markers_removed():
    code = ("/*\n * @source: x\n * @vulnerable_at_lines: 4\n */\ncontract Buggy {\n"
            "    // <yes> <report> REENTRANCY\n    function withdraw() public {}\n}")
    out = clean_source(code)
    for marker in ("@vulnerable_at_lines", "<yes>", "<report>", "REENTRANCY", "@source"):
        assert marker not in out
    assert "function withdraw() public {}" in out and _same_layout(code, out)


def test_division_operator_is_not_a_comment():
    out = clean_source("uint a = b / c; uint d = e /f;\n")
    assert "b / c" in out and "e /f" in out


# --------------------------------------------------------- normalisation --
def test_literals_replaced():
    code = ("address o = 0x1234567890abcdef1234567890abcdef12345678;\nuint v = 1_000;\n"
            "uint w = 1e18; uint z = .5; string m = 'hello'; bytes h = hex\"00ff\";")
    out = normalise(code)
    for gone in ("0x1234", "1_000", "1e18", ".5", "hello", "00ff"):
        assert gone not in out
    assert out.count("<STR>") == 2 and "<NUM>" in out
    assert count_lines(out) == count_lines(code)


def test_numbers_inside_identifiers_untouched():
    assert normalise("uint256 x2 = y;") == "uint256 x2 = y;"


def test_identifier_normalisation_keeps_security_builtins():
    code = ("function withdraw(uint amount) public { require(balances[msg.sender] >= amount);"
            " msg.sender.call.value(amount)(\"\"); tx.origin; block.timestamp; selfdestruct(owner); }")
    toks = prepare_tokens(code, ids=True)
    for keep in ("msg.sender", "call", "value", "tx.origin", "block.timestamp",
                 "selfdestruct", "require", "function", "public"):
        assert keep in toks, keep
    for gone in ("withdraw", "amount", "balances", "owner"):
        assert gone not in toks, gone
    assert "<ID>" in toks and "<STR>" in toks


def test_identifier_normalisation_handles_comparison_operators():
    toks = prepare_tokens("if (a<b && c>limit) {}", ids=True)
    assert toks == ["if", "(", "<ID>", "<", "<ID>", "&&", "<ID>", ">", "<ID>", ")", "{", "}"]


def test_member_names_after_dot_are_kept():
    toks = prepare_tokens("token.transferFrom(a, b, c); SafeMath.add(x, y);", ids=True)
    assert "transferFrom" in toks and "add" in toks and "token" not in toks


def test_protected_set_contains_blueprint_tokens():
    for t in ("call", "send", "transfer", "delegatecall", "selfdestruct",
              "msg.sender", "tx.origin", "block.timestamp"):
        assert t in PROTECTED


# --------------------------------------------------------------- tokens --
def test_tokenize_composite_and_operators():
    toks = prepare_tokens("require(tx.origin == msg.sender, 'no'); x += 1; y >>= 2;")
    assert toks[:6] == ["require", "(", "tx.origin", "==", "msg.sender", ","]
    assert "+=" in toks and ">>=" in toks and "<STR>" in toks and "<NUM>" in toks


def test_composite_token_requires_word_boundary():
    assert "msg.sender" not in tokenize("msg.senderX")


@pytest.mark.parametrize("ids", [False, True])
def test_tokenize_lines_concatenates_to_prepare_tokens(ids):
    code = ("pragma solidity ^0.4.24;\n// c\ncontract A {\n  function f() public { x = \"//\";"
            " token\n    .transfer(a); }\n}\n")
    per_line = tokenize_lines(code, ids=ids)
    assert len(per_line) == count_lines(code)
    assert [t for line in per_line for t in line] == prepare_tokens(code, ids=ids)
    assert per_line[1] == []  # the comment line is empty after cleaning


def test_truncation_keeps_head_and_tail():
    toks = [str(i) for i in range(100)]
    out, flag = truncate_tokens(toks, max_len=20, head_fraction=0.25)
    assert flag and len(out) == 20 and out[:5] == toks[:5] and out[5:] == toks[-15:]
    assert truncate_tokens(toks, max_len=150) == (toks, False)


# ------------------------------------------------- invariants / speed --
def test_a3_line_count_invariant_on_random_inputs():
    """Acceptance test A3 on 1,000 randomly generated snippets."""
    rng = random.Random(42)
    pieces = ["//", "/*", "*/", '"', "'", "\\", "\n", "\r\n", "\r", "a", " ", "x=1;", "محفوظ", "/", "*"]
    for _ in range(1000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 60)))
        out = clean_source(text)
        assert _same_layout(text, out), repr(text)
        assert split_lines(out) and count_lines(out) == count_lines(text)


def test_very_long_input_is_fast_and_safe():
    sample = "contract A {\n // line comment\n /* block */\n uint x = 1; string s = \"a//b\";\n}\n" * 20000
    start = time.perf_counter()
    out = clean_source(sample)
    toks = prepare_tokens(sample)
    elapsed = time.perf_counter() - start
    assert count_lines(out) == count_lines(sample)
    assert len(toks) > 100000
    assert elapsed < 5.0, f"too slow: {elapsed:.2f}s"


@pytest.mark.parametrize("junk", ["\x00\x01\x02", "🙂" * 50, "}}}{{{", "\\" * 99, "'" * 7])
def test_garbage_never_raises(junk):
    out = clean_source(junk)
    assert count_lines(out) == count_lines(junk)
    normalise_identifiers(tokenize(normalise(out)))
