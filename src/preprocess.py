"""
Muhafiz.sol - Preprocessing Module
Phase 1: Data Foundation and Feature Pipeline
Owner: Talha Sami (24I-3118)

Implements line-preserving source cleaning, token normalization,
and tokenization according to the Master Project Blueprint.
"""

import re
from typing import List, Tuple

PROTECTED_KEYWORDS = {
    # Solidity Core
    "contract", "interface", "library", "is", "struct", "enum", "event",
    "function", "modifier", "constructor", "fallback", "receive",
    "public", "private", "internal", "external", "pure", "view", "payable",
    "constant", "immutable", "anonymous", "indexed", "virtual", "override",
    "storage", "memory", "calldata", "returns", "return", "if", "else",
    "for", "while", "do", "break", "continue", "throw", "revert",
    "require", "assert", "emit", "new", "delete", "try", "catch",
    "assembly", "pragma", "solidity", "import", "using",
    # Types
    "address", "bool", "string", "bytes", "byte",
    "int", "uint", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "int128", "uint128", "int256", "uint256",
    "bytes1", "bytes2", "bytes3", "bytes4", "bytes32", "mapping",
    # Built-in variables and Security-sensitive members
    "msg", "msg.sender", "msg.value", "msg.data", "msg.sig",
    "tx", "tx.origin", "tx.gasprice",
    "block", "block.timestamp", "block.number", "block.coinbase",
    "block.difficulty", "block.gaslimit", "block.chainid", "now",
    "this", "super", "selfdestruct", "suicide",
    "call", "delegatecall", "callcode", "staticcall",
    "send", "transfer", "balance", "gasleft",
    "keccak256", "sha256", "sha3", "ripemd160", "ecrecover",
    "addmod", "mulmod"
}


def clean_source(text: str) -> str:
    """
    Clean Solidity source code by stripping single-line and multi-line comments.
    
    Guarantees:
    - Exactly preserves total line count and line breaks (comments are replaced with spaces).
    - Ignores comment markers inside string literals ('...', "...").
    - Preserves UTF-8 and CRLF/LF line endings accurately.
    """
    if not text:
        return ""

    chars = list(text)
    n = len(chars)
    i = 0

    STATE_NORMAL = 0
    STATE_SINGLE_QUOTE = 1
    STATE_DOUBLE_QUOTE = 2
    STATE_LINE_COMMENT = 3
    STATE_BLOCK_COMMENT = 4

    state = STATE_NORMAL

    while i < n:
        c = chars[i]
        next_c = chars[i + 1] if i + 1 < n else ""

        if state == STATE_NORMAL:
            if c == "'" and not (i > 0 and chars[i - 1] == "\\"):
                state = STATE_SINGLE_QUOTE
            elif c == '"' and not (i > 0 and chars[i - 1] == "\\"):
                state = STATE_DOUBLE_QUOTE
            elif c == "/" and next_c == "/":
                state = STATE_LINE_COMMENT
                chars[i] = " "
                chars[i + 1] = " "
                i += 1
            elif c == "/" and next_c == "*":
                state = STATE_BLOCK_COMMENT
                chars[i] = " "
                chars[i + 1] = " "
                i += 1

        elif state == STATE_SINGLE_QUOTE:
            if c == "\\" and i + 1 < n:
                i += 1  # Skip escaped character
            elif c == "'":
                state = STATE_NORMAL

        elif state == STATE_DOUBLE_QUOTE:
            if c == "\\" and i + 1 < n:
                i += 1  # Skip escaped character
            elif c == '"':
                state = STATE_NORMAL

        elif state == STATE_LINE_COMMENT:
            if c in ("\n", "\r"):
                state = STATE_NORMAL
            else:
                chars[i] = " "

        elif state == STATE_BLOCK_COMMENT:
            if c == "*" and next_c == "/":
                chars[i] = " "
                chars[i + 1] = " "
                i += 1
                state = STATE_NORMAL
            elif c not in ("\n", "\r"):
                chars[i] = " "

        i += 1

    return "".join(chars)


def normalise(text: str, preserve_builtins: bool = True) -> str:
    """
    Normalise cleaned source code:
    - String literals replaced with <STR>
    - Hex and numeric literals replaced with <NUM>
    - Preserves protected security-critical tokens (e.g. msg.sender, tx.origin, delegatecall).
    """
    if not text:
        return ""

    # Replace string literals
    def _replace_string(match):
        return "<STR>"

    # Single and double quoted strings
    pattern_string = r'(\'[^\'\n\r\\]*(?:\\.[^\'\n\r\\]*)*\'|"[^"\n\r\\]*(?:\\.[^"\n\r\\]*)*")'
    text = re.sub(pattern_string, _replace_string, text)

    # Replace Hex literals
    text = re.sub(r'\b0x[0-9a-fA-F]+\b', '<NUM>', text)

    # Replace Decimal / Scientific numbers
    text = re.sub(r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b', '<NUM>', text)

    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize Solidity code into list of syntactic tokens.
    Recognizes composite security keywords (e.g., 'msg.sender', 'tx.origin', 'block.timestamp')
    as atomic tokens when encountered.
    """
    if not text:
        return []

    # First match composite security expressions
    token_pattern = re.compile(
        r'(?:msg\.sender|msg\.value|msg\.data|msg\.sig|'
        r'tx\.origin|tx\.gasprice|'
        r'block\.timestamp|block\.number|block\.coinbase|block\.difficulty|block\.gaslimit|block\.chainid|'
        r'<STR>|<NUM>|[a-zA-Z_$][a-zA-Z0-9_$]*|[0-9]+|'
        r'==|!=|<=|>=|=>|->|\+\+|--|&&|\|\||\*\*|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|'
        r'[{}()\[\],;.?:]|[-+*/%&|^!~<>=])'
    )
    tokens = token_pattern.findall(text)
    return tokens


def truncate_tokens(tokens: List[str], max_len: int = 4096) -> Tuple[List[str], bool]:
    """
    Apply long-file policy: truncate to max_len tokens and set a flag.
    """
    if len(tokens) > max_len:
        return tokens[:max_len], True
    return tokens, False
