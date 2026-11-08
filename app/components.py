"""
Muhafiz.sol - rendering helpers for the Streamlit app (blueprint P3.3).

Everything here produces small, self-contained HTML snippets so the result page
stays readable for non-technical users: a risk badge with icon + words (never
colour alone), probability bars with the decision threshold marked, and a code
viewer with line numbers and highlighted lines.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Dict, List, Optional

# Plain-language meaning of every label family the model may use (blueprint 5).
CLASS_INFO: Dict[str, Dict[str, str]] = {
    "reentrancy": {
        "title": "Reentrancy",
        "plain": "Money is sent out before the contract updates its own records, so an attacker can ask "
                 "again and again before the balance changes - like a shopkeeper who pays a refund before "
                 "crossing it out of the ledger.",
        "fix": "Update balances first, then send money (checks-effects-interactions), or add a reentrancy lock.",
    },
    "access-control": {
        "title": "Access control",
        "plain": "A sensitive action (moving funds, destroying the contract, changing the owner) may be "
                 "usable by anyone, or the permission check relies on tx.origin.",
        "fix": "Restrict sensitive functions with an owner/role check based on msg.sender.",
    },
    "unchecked-calls": {
        "title": "Unchecked calls",
        "plain": "The contract sends money or calls another contract but never checks whether that worked, "
                 "so it may carry on as if a failed payment succeeded.",
        "fix": "Check the return value of call/send (or use transfer / require(success)).",
    },
    "arithmetic": {
        "title": "Arithmetic (divide before multiply)",
        "plain": "A calculation divides before it multiplies, which silently throws away precision and can "
                 "produce wrong amounts.",
        "fix": "Multiply first and divide last, or use a fixed-point maths library.",
    },
    "bad-randomness": {
        "title": "Bad randomness",
        "plain": "A 'random' number is derived from block data (time, block hash) that miners can see or "
                 "nudge, so outcomes such as lottery winners can be predicted or influenced.",
        "fix": "Use a verifiable randomness source (e.g. a VRF oracle) or a commit-reveal scheme.",
    },
    "locked-ether": {
        "title": "Locked ether",
        "plain": "The contract can receive ether but has no way to send it out again, so funds can get stuck.",
        "fix": "Add a guarded withdraw function, or reject payments the contract cannot handle.",
    },
    "double-spending": {
        "title": "msg.value / delegatecall in a loop",
        "plain": "The same payment value is reused inside a loop, so one payment can be counted several times.",
        "fix": "Track the amount spent inside the loop instead of reusing msg.value.",
    },
    "other": {
        "title": "Other weaknesses",
        "plain": "A mix of less common issues flagged by the training labels (uninitialised storage, strict "
                 "equality on balances, hidden backdoors and similar).",
        "fix": "Review the highlighted lines manually.",
    },
}

RISK_STYLE = {
    "low": {"icon": "✅", "word": "Low risk", "bg": "#e6f4ea", "fg": "#0d652d", "border": "#34a853",
            "text": "No weakness type reached its alert level. This is NOT a guarantee that the contract is safe."},
    "medium": {"icon": "⚠️", "word": "Medium risk", "bg": "#fef7e0", "fg": "#7a4f01", "border": "#f9ab00",
               "text": "At least one weakness type reached its alert level. Review the highlighted lines."},
    "high": {"icon": "⛔", "word": "High risk", "bg": "#fce8e6", "fg": "#8c1d18", "border": "#d93025",
             "text": "At least one weakness type is far above its alert level. Fix the highlighted code "
                     "and get a professional review before deploying."},
}

DISCLAIMER = ("Muhafiz.sol is a student research tool and a fast first-pass triage aid. It does not replace a "
              "professional security audit, never certifies that a contract is safe, and is not affiliated with "
              "or approved by PVARA or any regulator.")


def class_title(c: str) -> str:
    return CLASS_INFO.get(c, {"title": c})["title"]


def risk_badge(level: str) -> str:
    s = RISK_STYLE[level]
    return (f'<div style="background:{s["bg"]};border-left:6px solid {s["border"]};border-radius:10px;'
            f'padding:14px 18px;margin:6px 0 12px 0">'
            f'<div style="font-size:1.6rem;font-weight:700;color:{s["fg"]}">{s["icon"]} {s["word"]}</div>'
            f'<div style="color:{s["fg"]};margin-top:4px">{html.escape(s["text"])}</div></div>')


def probability_bars(result: Dict, order: Optional[List[str]] = None) -> str:
    """Horizontal bars, threshold marked with a dark tick; flagged classes labelled in words."""
    probs, thr, flags = result["probabilities"], result["thresholds"], result["flags"]
    order = order or sorted(probs, key=lambda c: -probs[c])
    rows = []
    for c in order:
        p, t, f = probs[c], thr[c], flags[c]
        colour = "#d93025" if f else "#2a78d6"
        status = "⚠ flagged" if f else "below alert level"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:7px 0" '
            f'title="{html.escape(class_title(c))}: {p:.0%} (alert level {t:.0%})">'
            f'<div style="width:190px;font-weight:600">{html.escape(class_title(c))}</div>'
            f'<div style="flex:1;position:relative;height:18px;background:#eef0f3;border-radius:9px">'
            f'<div style="width:{p * 100:.1f}%;height:100%;background:{colour};border-radius:9px"></div>'
            f'<div style="position:absolute;left:calc({t * 100:.1f}% - 1px);top:-4px;width:3px;height:26px;'
            f'background:#202124;border-radius:2px"></div></div>'
            f'<div style="width:150px;text-align:right;font-variant-numeric:tabular-nums">'
            f'<b>{p:.0%}</b> <span style="color:#5f6368;font-size:0.85em">{status}</span></div></div>')
    legend = ('<div style="color:#5f6368;font-size:0.85em;margin-top:6px">Bar = how likely the model thinks '
              'this weakness is present. Dark tick = alert level (decision threshold). Red bars crossed it.</div>')
    return "<div>" + "".join(rows) + legend + "</div>"


def code_viewer(source: str, highlights: Dict[int, float], max_lines: int = 2500) -> str:
    """Source with line numbers; highlighted lines get a warm background and their rank."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ranked = {ln: i + 1 for i, ln in enumerate(sorted(highlights, key=lambda k: -highlights[k]))}
    show = range(len(lines))
    note = ""
    if len(lines) > max_lines:
        keep = set()
        for ln in highlights or [1]:
            keep.update(range(max(0, ln - 40), min(len(lines), ln + 40)))
        show = sorted(keep)
        note = (f'<div style="color:#5f6368;padding:4px 8px">Long file ({len(lines):,} lines): showing the '
                f'code around the highlighted lines only.</div>')
    out, prev = [], -1
    for i in show:
        if prev != -1 and i != prev + 1:
            out.append('<div style="color:#9aa0a6;padding-left:8px">⋮</div>')
        prev = i
        ln = i + 1
        txt = html.escape(lines[i]) or "&nbsp;"
        if ln in ranked:
            style = "background:#fde7c8;border-left:4px solid #eb6834"
            tag = f'<span style="color:#b3541e;font-weight:700;padding-right:6px">#{ranked[ln]}</span>'
        else:
            style, tag = "border-left:4px solid transparent", ""
        out.append(f'<div style="{style};white-space:pre;display:flex">'
                   f'<span style="color:#9aa0a6;min-width:3.2em;text-align:right;padding-right:10px;'
                   f'user-select:none">{ln}</span><span>{tag}{txt}</span></div>')
    return (f'{note}<div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;'
            f'line-height:1.5;background:#fbfbfa;border:1px solid #e4e3df;border-radius:8px;padding:8px 0;'
            f'max-height:520px;overflow:auto">{"".join(out)}</div>')


def report_markdown(result: Dict, explanation: Dict, similar: List[Dict], source_name: str) -> str:
    """Downloadable scan report (F11)."""
    s = RISK_STYLE[result["risk_level"]]
    lines = [f"# Muhafiz.sol scan report", "",
             f"- File: {source_name}", f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
             f"- Model: {result['model']}", f"- Overall: **{s['word']}** - {s['text']}", "",
             "## Weakness types", "", "| Type | Probability | Alert level | Flagged |", "|---|---|---|---|"]
    for c in sorted(result["probabilities"], key=lambda k: -result["probabilities"][k]):
        lines.append(f"| {class_title(c)} | {result['probabilities'][c]:.1%} | {result['thresholds'][c]:.1%} | "
                     f"{'yes' if result['flags'][c] else 'no'} |")
    lines += ["", "## Most influential lines", ""]
    for c, items in explanation.get("lines", {}).items():
        if result["flags"].get(c) and items:
            lines.append(f"**{class_title(c)}**")
            lines += [f"- line {it['line']} (score {it['score']:.3f}): `{it['text']}`" for it in items]
            lines.append("")
    if similar:
        lines += ["## Similar known contracts (training data)", ""]
        lines += [f"- {s_['contract_name']} ({s_['address']}), similarity {s_['similarity']:.2f}, labels: "
                  f"{', '.join(s_['labels'])}" for s_ in similar]
    lines += ["", "---", DISCLAIMER]
    return "\n".join(lines)


def report_json(result: Dict, explanation: Dict, similar: List[Dict], source_name: str) -> str:
    return json.dumps({"file": source_name, "result": result, "explanation": explanation, "similar": similar,
                       "disclaimer": DISCLAIMER}, indent=2)
