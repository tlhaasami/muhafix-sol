"""
Muhafiz.sol - preprocessing module (blueprint P1.4, interface of Section 7.4).

Public API
----------
clean_source(text)            -> text with every comment blanked, same line count
normalise(text)               -> literals replaced by <STR>/<NUM>
tokenize(text)                -> list of tokens
normalise_identifiers(tokens) -> user-defined identifiers replaced by <ID>
tokenize_lines(text, ids)     -> per-line token lists of a *raw* source
prepare_tokens(text, ids)     -> full pipeline raw text -> tokens
truncate_tokens(tokens, n)    -> long-file policy (head + tail) and a flag

Design notes
------------
* ``clean_source`` is a small state machine (NORMAL, STRING, LINE_COMMENT,
  BLOCK_COMMENT) that reads the text once. Instead of stepping one character at
  a time in Python it jumps between "interesting" characters with pre-compiled
  regular expressions, which makes it ~50x faster on the 1M-character flattened
  contracts of Dataset A while keeping the same semantics.
* Comment characters become spaces; ``\\n`` and ``\\r`` are kept, so line *and*
  column numbers stay valid. This is what makes line highlighting possible and
  what removes the SmartBugs ``@vulnerable_at_lines`` / ``// <yes> <report>``
  annotations without disturbing the annotated line numbers.
* Security-relevant built-ins (``call``, ``send``, ``transfer``,
  ``delegatecall``, ``selfdestruct``, ``msg.sender``, ``tx.origin``,
  ``block.timestamp`` ...) are never normalised away - they carry the signal.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Protected vocabulary: Solidity keywords, types, global variables and members
# that must survive identifier normalisation.
# --------------------------------------------------------------------------- #
_KEYWORDS = """
pragma solidity import as from using for is contract interface library abstract
struct enum event error function modifier constructor fallback receive
public private internal external pure view payable constant immutable anonymous
indexed virtual override storage memory calldata returns return if else while do
break continue throw revert require assert emit new delete try catch assembly
unchecked type true false wei gwei szabo finney ether seconds minutes hours days
weeks years var let switch case default leave
""".split()

_TYPES = ["address", "bool", "string", "bytes", "byte", "int", "uint", "mapping",
          "fixed", "ufixed"]
_TYPES += [f"int{n}" for n in range(8, 257, 8)] + [f"uint{n}" for n in range(8, 257, 8)]
_TYPES += [f"bytes{n}" for n in range(1, 33)]

_BUILTINS = """
msg tx block abi now this super selfdestruct suicide gasleft blockhash
keccak256 sha256 sha3 ripemd160 ecrecover addmod mulmod
call delegatecall callcode staticcall send transfer value gas balance code
codehash length push pop sender origin timestamp number coinbase difficulty
prevrandao gaslimit chainid basefee gasprice data sig encode encodePacked
encodeWithSelector encodeWithSignature decode selector
""".split()

COMPOSITE_TOKENS = [
    "msg.sender", "msg.value", "msg.data", "msg.sig", "msg.gas",
    "tx.origin", "tx.gasprice",
    "block.timestamp", "block.number", "block.coinbase", "block.difficulty",
    "block.gaslimit", "block.chainid", "block.prevrandao", "block.basefee",
]

PROTECTED = frozenset(_KEYWORDS + _TYPES + _BUILTINS + COMPOSITE_TOKENS)
PLACEHOLDERS = ("<STR>", "<NUM>", "<ID>")

# --------------------------------------------------------------------------- #
# clean_source: line-preserving comment removal
# --------------------------------------------------------------------------- #
_NORMAL_NEXT = re.compile(r"//|/\*|\"|'")
_LINE_END = re.compile(r"[\r\n]")
_BLOCK_END = re.compile(r"\*/")
_NOT_NEWLINE = re.compile(r"[^\r\n]")
_STRING_NEXT = {'"': re.compile(r"[\\\"\r\n]"), "'": re.compile(r"[\\'\r\n]")}


def _blank(segment: str) -> str:
    """Replace every non-newline character by a space (keeps line/col layout)."""
    return _NOT_NEWLINE.sub(" ", segment)


def clean_source(text: str) -> str:
    """Blank out ``//`` and ``/* */`` comments while preserving line numbers.

    Comment markers that appear inside string literals are left untouched. An
    unterminated string ends at the end of its line (Solidity strings cannot
    span lines), and an unterminated block comment runs to the end of file.
    """
    if not text:
        return ""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _NORMAL_NEXT.search(text, i)            # state NORMAL
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        tok = m.group()
        if tok == "//":                             # state LINE_COMMENT
            e = _LINE_END.search(text, m.start())
            end = e.start() if e else n
            out.append(" " * (end - m.start()))
            i = end
        elif tok == "/*":                           # state BLOCK_COMMENT
            e = _BLOCK_END.search(text, m.start() + 2)
            end = e.end() if e else n
            out.append(_blank(text[m.start():end]))
            i = end
        else:                                       # state STRING
            pat, j = _STRING_NEXT[tok], m.end()
            while True:
                s = pat.search(text, j)
                if s is None:
                    j = n
                    break
                ch = s.group()
                if ch == "\\":
                    j = s.end() + 1                 # skip the escaped character
                    continue
                j = s.end() if ch == tok else s.start()  # newline: close string
                break
            j = min(j, n)
            out.append(text[m.start():j])
            i = j
    return "".join(out)


# --------------------------------------------------------------------------- #
# normalise / tokenize
# --------------------------------------------------------------------------- #
_STRING_LIT = re.compile(
    r"""(?:\b(?:hex|unicode))?(?:"(?:[^"\\\r\n]|\\.)*"|'(?:[^'\\\r\n]|\\.)*')"""
)
_HEX_LIT = re.compile(r"\b0[xX][0-9a-fA-F_]+\b")
_NUM_LIT = re.compile(r"(?<![\w$])(?:\d[\d_]*(?:\.\d[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?\b")

_COMPOSITE_RE = "|".join(re.escape(t) for t in sorted(COMPOSITE_TOKENS, key=len, reverse=True))
_TOKEN_RE = re.compile(
    rf"(?:{_COMPOSITE_RE})(?![\w$])"
    r"|<STR>|<NUM>|<ID>"
    r"|[A-Za-z_$][A-Za-z0-9_$]*"
    r"|\d+"
    r"|>>>=|<<=|>>=|>>>|\*\*|==|!=|<=|>=|=>|->|\+\+|--|&&|\|\||<<|>>"
    r"|\+=|-=|\*=|/=|%=|&=|\|=|\^=|:="
    r"|[{}()\[\],;.?:]|[-+*/%&|^!~<>=]"
)
_IDENT_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")


def normalise(text: str) -> str:
    """Replace string, hex and number literals with <STR> / <NUM>.

    Placeholders are padded with spaces so "a<1" never becomes "a<<NUM>"
    (which would tokenise as "<<"). Newlines are never touched, so the line
    structure survives.
    """
    if not text:
        return ""
    text = _STRING_LIT.sub(" <STR> ", text)
    text = _HEX_LIT.sub(" <NUM> ", text)
    return _NUM_LIT.sub(" <NUM> ", text)


def tokenize(text: str) -> List[str]:
    """Split (cleaned, normalised) Solidity text into tokens.

    Composite globals such as ``msg.sender`` are kept as one token; other
    member accesses (``x.call``) become ``x``, ``.``, ``call``. Characters that
    are not valid Solidity (stray unicode, ``@``, ``#``) are dropped.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def normalise_identifiers(tokens: List[str], prev: str = "") -> List[str]:
    """Replace user-defined identifiers with <ID> (optional step of P1.4).

    Kept unchanged: Solidity keywords, types, global variables and members
    (``PROTECTED``), placeholders, and any name directly after a ``.``. Member
    names (``transferFrom``, ``add``, ``call``) name external APIs, which carry
    security signal; only *declared* names such as ``balances`` or ``withdraw``
    are anonymised. ``prev`` is the token preceding ``tokens`` (used when
    normalising a file line by line).
    """
    out: List[str] = []
    for tok in tokens:
        if prev != "." and tok not in PROTECTED and _IDENT_TOKEN.match(tok):
            out.append("<ID>")
        else:
            out.append(tok)
        prev = tok
    return out


def split_lines(text: str) -> List[str]:
    """Split on CRLF, CR or LF - the same convention used for line numbers."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def count_lines(text: str) -> int:
    return len(split_lines(text))


def prepare_tokens(text: str, ids: bool = False) -> List[str]:
    """Raw Solidity -> tokens (clean, normalise literals, tokenize[, <ID>])."""
    toks = tokenize(normalise(clean_source(text)))
    return normalise_identifiers(toks) if ids else toks


def tokenize_lines(text: str, ids: bool = False) -> List[List[str]]:
    """Raw Solidity -> one token list per source line (line i is index i-1).

    Concatenating the lists gives exactly ``prepare_tokens(text, ids)``; this is
    what lets the explanation module blank single lines cheaply.
    """
    lines = [tokenize(line) for line in split_lines(normalise(clean_source(text)))]
    if not ids:
        return lines
    out, prev = [], ""
    for toks in lines:
        out.append(normalise_identifiers(toks, prev))
        if toks:
            prev = toks[-1]
    return out


def truncate_tokens(tokens: List[str], max_len: int = 20000,
                    head_fraction: float = 0.25) -> Tuple[List[str], bool]:
    """Long-file policy: keep the first ``head_fraction`` and the end of the file.

    Flattened Etherscan files put libraries first and the main contract last,
    so the tail carries most of the contract-specific signal. Returns the
    (possibly shortened) list and a flag telling whether truncation happened.
    """
    if max_len <= 0 or len(tokens) <= max_len:
        return tokens, False
    head = int(max_len * head_fraction)
    return tokens[:head] + tokens[len(tokens) - (max_len - head):], True
