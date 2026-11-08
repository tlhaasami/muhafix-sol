"""
Muhafiz.sol - line-level explanation by occlusion (blueprint 8.5, P3.2).

    explain(source, class_name, top_k=3, predictor=None) -> list of
        {"line": int, "score": float, "text": str, "function": str | None}

Method (two-stage occlusion):
 1. find function-like blocks (function / modifier / constructor / fallback /
    receive) by brace matching on the cleaned, literal-normalised code;
 2. blank each block in turn and record how much the class probability drops;
 3. inside the top blocks (plus top-level lines if there are few blocks) blank
    each line in turn; the line score is the probability drop;
 4. return the k lines with the largest drops.

The module is model-agnostic: it only needs ``removal_probabilities`` from the
predictor (probability of every class for the full token list and for copies
with token spans removed). Odd input never raises: it returns an empty list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import load_config
from src.preprocess import clean_source, normalise, split_lines

_BLOCK_START = re.compile(r"\b(function|modifier|constructor|fallback|receive)\b\s*([A-Za-z_$][\w$]*)?")


@dataclass
class Block:
    kind: str
    name: str
    start_line: int   # 1-based inclusive
    end_line: int     # 1-based inclusive


def find_blocks(source: str) -> List[Block]:
    """Function-like blocks with their line ranges (brace matching, comment/string safe)."""
    text = normalise(clean_source(source))
    # map character offset -> line number
    line_starts = [0]
    for m in re.finditer(r"\r\n|\r|\n", text):
        line_starts.append(m.end())
    starts = np.array(line_starts)

    def line_of(pos: int) -> int:
        return int(np.searchsorted(starts, pos, side="right"))

    blocks: List[Block] = []
    pos = 0
    while True:
        m = _BLOCK_START.search(text, pos)
        if not m:
            break
        brace = text.find("{", m.end())
        semi = text.find(";", m.end())
        if brace == -1 or (semi != -1 and semi < brace):   # declaration without body
            pos = m.end()
            continue
        depth, i = 0, brace
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = min(i, len(text) - 1)
        name = m.group(2) or m.group(1)
        if m.group(1) == "function" and m.group(2) is None:
            name = "fallback"
        blocks.append(Block(m.group(1), name, line_of(m.start()), line_of(end)))
        pos = end + 1
    return blocks


def _spans_for_lines(owner: Sequence[int], lines: Sequence[int]) -> List[Tuple[int, int]]:
    """Token span [a, b) covering the given (contiguous) set of lines in ``owner``."""
    lines = set(lines)
    idx = [i for i, ln in enumerate(owner) if ln in lines]
    if not idx:
        return []
    return [(idx[0], idx[-1] + 1)]


def explain_all(source: str, predictor, top_k: Optional[int] = None, model: Optional[str] = None
                ) -> Dict[str, object]:
    """Occlusion for every class at once (one batch of removals).

    Returns {"lines": {class: [items...]}, "functions": {class: [items...]}, "base": {class: p}}.
    """
    cfg = load_config()["explain"]
    top_k = top_k or cfg["top_k"]
    empty = {"lines": {c: [] for c in predictor.classes}, "functions": {c: [] for c in predictor.classes},
             "base": {}}
    try:
        raw_lines = split_lines(source)
        _, tokens, owner = predictor.line_tokens(source)
        if len(tokens) < 5:
            return empty
        blocks = [b for b in find_blocks(source)][: cfg["max_functions"]]
        block_spans = []
        for b in blocks:
            sp_ = _spans_for_lines(owner, range(b.start_line, b.end_line + 1))
            block_spans.append(sp_[0] if sp_ else None)
        valid = [(b, s) for b, s in zip(blocks, block_spans) if s is not None]

        base, removed = predictor.removal_probabilities(tokens, [s for _, s in valid], model)
        func_drop = base[None, :] - removed if len(valid) else np.zeros((0, len(predictor.classes)))

        # candidate lines: inside the top blocks per class (union), plus top-level code
        cand: set = set()
        n_top = cfg["top_functions"]
        for c in range(len(predictor.classes)):
            order = np.argsort(-func_drop[:, c])[:n_top] if len(valid) else []
            for j in order:
                b = valid[int(j)][0]
                cand.update(range(b.start_line, b.end_line + 1))
        in_block = set()
        for b, _ in valid:
            in_block.update(range(b.start_line, b.end_line + 1))
        token_lines = sorted(set(owner))
        outside = [ln for ln in token_lines if ln not in in_block]
        cand.update(outside)
        cand = [ln for ln in token_lines if ln in cand][: cfg["max_lines"]]
        line_spans = [_spans_for_lines(owner, [ln])[0] for ln in cand]
        _, removed_l = predictor.removal_probabilities(tokens, line_spans, model)
        line_drop = base[None, :] - removed_l if len(cand) else np.zeros((0, len(predictor.classes)))

        def owner_block(ln: int) -> Optional[str]:
            for b, _ in valid:
                if b.start_line <= ln <= b.end_line:
                    return b.name
            return None

        out_lines, out_funcs = {}, {}
        for c, name in enumerate(predictor.classes):
            order = np.argsort(-line_drop[:, c], kind="mergesort")[:top_k]
            out_lines[name] = [{"line": int(cand[j]), "score": float(line_drop[j, c]),
                                "text": raw_lines[cand[j] - 1].strip()[:200], "function": owner_block(cand[j])}
                               for j in order if line_drop[j, c] > 0]
            forder = np.argsort(-func_drop[:, c], kind="mergesort")[:5] if len(valid) else []
            out_funcs[name] = [{"function": valid[int(j)][0].name, "kind": valid[int(j)][0].kind,
                                "start_line": valid[int(j)][0].start_line, "end_line": valid[int(j)][0].end_line,
                                "score": float(func_drop[int(j), c])} for j in forder]
        return {"lines": out_lines, "functions": out_funcs,
                "base": {c: float(p) for c, p in zip(predictor.classes, base)}}
    except Exception:  # never let an explanation crash the app
        return empty


def explain(source: str, class_name: str, top_k: int = 3, predictor=None, model: Optional[str] = None
            ) -> List[Dict]:
    """Interface contract 7.4: top-k lines for one class."""
    if predictor is None:
        from src.predict import Predictor

        predictor = Predictor()
    if class_name not in predictor.classes:
        raise ValueError(f"unknown class {class_name!r}; choose from {predictor.classes}")
    return explain_all(source, predictor, top_k, model)["lines"][class_name]
