"""
Unit test suite for src/preprocess.py
Mandated by Phase 1 Work Package P1.4 and Acceptance Test A3.
"""

import pytest
import time
from src.preprocess import clean_source, normalise, tokenize, truncate_tokens, PROTECTED_KEYWORDS


def test_empty_input():
    assert clean_source("") == ""
    assert normalise("") == ""
    assert tokenize("") == []
    tokens, trunc = truncate_tokens([], max_len=10)
    assert tokens == []
    assert not trunc


def test_single_line_comment_stripped_preserves_lines():
    code = "uint x = 1; // this is a line comment\nuint y = 2;"
    cleaned = clean_source(code)
    assert "// this is a line comment" not in cleaned
    assert "uint x = 1;" in cleaned
    assert "uint y = 2;" in cleaned
    assert code.count("\n") == cleaned.count("\n")


def test_block_comment_stripped_preserves_lines():
    code = "/* multi-line\n   comment here\n*/\ncontract Token {}"
    cleaned = clean_source(code)
    assert "multi-line" not in cleaned
    assert "contract Token {}" in cleaned
    assert code.count("\n") == cleaned.count("\n")
    assert len(code.splitlines()) == len(cleaned.splitlines())


def test_comment_markers_inside_double_quoted_string():
    code = 'string memory s = "http://example.com/*not a comment*/"; // real comment\n'
    cleaned = clean_source(code)
    assert "http://example.com/*not a comment*/" in cleaned
    assert "real comment" not in cleaned
    assert code.count("\n") == cleaned.count("\n")


def test_comment_markers_inside_single_quoted_string():
    code = "bytes memory b = 'test // not a comment';\n"
    cleaned = clean_source(code)
    assert "test // not a comment" in cleaned
    assert code.count("\n") == cleaned.count("\n")


def test_nested_looking_comment_markers():
    code = "/* outer comment /* nested lookalike */ still code */\nuint z = 5;"
    cleaned = clean_source(code)
    assert "outer comment" not in cleaned
    assert "uint z = 5;" in cleaned
    assert code.count("\n") == cleaned.count("\n")


def test_windows_crlf_line_endings():
    code = "contract A {\r\n    // comment\r\n    uint256 a;\r\n}"
    cleaned = clean_source(code)
    assert "// comment" not in cleaned
    assert cleaned.count("\r\n") == code.count("\r\n")
    assert len(code.split("\r\n")) == len(cleaned.split("\r\n"))


def test_unicode_preservation():
    code = "// Urdu comment: محفوظ سمارٹ کانٹریکٹ\nstring name = 'محفوظ';\n"
    cleaned = clean_source(code)
    assert "محفوظ سمارٹ کانٹریکٹ" not in cleaned
    assert "محفوظ" in cleaned
    assert code.count("\n") == cleaned.count("\n")


def test_annotation_markers_stripped():
    code = "// @vulnerable_at_lines: 3,4\ncontract Buggy {\n    // <yes> <report> REENTRANCY\n    function withdraw() public {}\n}"
    cleaned = clean_source(code)
    assert "@vulnerable_at_lines" not in cleaned
    assert "<yes>" not in cleaned
    assert "<report>" not in cleaned
    assert "function withdraw() public {}" in cleaned


def test_normalise_strings_and_numbers():
    code = "address owner = 0x1234567890abcdef1234567890abcdef12345678;\nuint256 val = 1000;\nstring msg = 'hello world';"
    norm = normalise(code)
    assert "<NUM>" in norm
    assert "<STR>" in norm
    assert "1000" not in norm
    assert "hello world" not in norm


def test_protected_security_keywords_preserved():
    code = "msg.sender.transfer(msg.value);\ntx.origin == owner;\nselfdestruct(payable(msg.sender));\ndelegatecall(data);"
    norm = normalise(code)
    assert "msg.sender" in norm
    assert "transfer" in norm
    assert "tx.origin" in norm
    assert "selfdestruct" in norm
    assert "delegatecall" in norm


def test_tokenize_composite_identifiers():
    code = "require(tx.origin == msg.sender, 'unauthorized');"
    cleaned = clean_source(code)
    norm = normalise(cleaned)
    tokens = tokenize(norm)
    assert "require" in tokens
    assert "tx.origin" in tokens
    assert "==" in tokens
    assert "msg.sender" in tokens
    assert "<STR>" in tokens


def test_truncation_policy():
    tokens = [f"tok_{i}" for i in range(100)]
    t_tokens, truncated = truncate_tokens(tokens, max_len=50)
    assert len(t_tokens) == 50
    assert truncated is True

    short_tokens, not_truncated = truncate_tokens(tokens, max_len=150)
    assert len(short_tokens) == 100
    assert not_truncated is False


def test_line_count_invariance_on_complex_contract():
    raw_contract = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/*
 * @title Complex Token
 * @author Audit Team
 */
contract ComplexToken {
    // State variables
    mapping(address => uint256) public balances;
    address public owner;

    constructor() {
        owner = msg.sender; // Set owner
    }

    function transfer(address to, uint256 amount) public {
        /* Check balance */
        require(balances[msg.sender] >= amount, "Insufficient");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}
"""
    cleaned = clean_source(raw_contract)
    assert len(raw_contract.splitlines()) == len(cleaned.splitlines())
    assert raw_contract.count("\n") == cleaned.count("\n")


def test_clean_source_benchmark_speed():
    sample = "contract A {\n // line comment\n /* block comment */\n uint x = 1;\n}\n" * 100
    start = time.time()
    for _ in range(50):
        _ = clean_source(sample)
    duration = time.time() - start
    assert duration < 1.0  # Processing 5000 lines should take well under 1 second
