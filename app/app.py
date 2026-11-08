"""
Muhafiz.sol - Streamlit application (blueprint 9, P3.1/P3.3/P3.4).

    streamlit run app/app.py

Flow: paste or upload a Solidity contract (or pick a sample) -> Scan ->
risk badge, per-type probabilities with alert levels, highlighted lines,
similar known contracts and a downloadable report. A second tab shows how good
the model is (numbers read from reports/tables, the same files used in the
report), a third tab scans several files at once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import components as ui  # noqa: E402
from src.predict import InputError, MockPredictor  # noqa: E402

SAMPLES = ROOT / "app" / "sample_contracts"
SAMPLE_LABELS = {
    "01_clean_token.sol": "A carefully written token (should look clean)",
    "02_reentrancy_bank.sol": "Bank with the refund-order mistake (reentrancy)",
    "03_unchecked_send.sol": "Refunds that ignore failed payments (unchecked calls)",
    "04_weak_randomness_lottery.sol": "Lottery using block data as randomness",
    "05_tricky_safe_with_keywords.sol": "Tricky: safe vault that uses 'call' correctly",
    "06_tricky_renamed_reentrancy.sol": "Tricky: reentrancy hidden behind odd names",
}
MAX_UPLOAD_MB = ui.MAX_UPLOAD_MB

st.set_page_config(page_title="Muhafiz.sol - smart contract checker", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the trained model (first time only)...")
def get_predictor():
    """Real predictor unless MUHAFIZ_MOCK=1 or the model files are missing."""
    if os.environ.get("MUHAFIZ_MOCK") == "1":
        return MockPredictor(), "mock (MUHAFIZ_MOCK=1)"
    try:
        from src.predict import Predictor

        return Predictor(), None
    except FileNotFoundError as exc:
        return MockPredictor(), f"model files not found ({exc.filename}); showing a mock"


def run_scan(source: str, name: str, model: str, scale: float):
    predictor, _ = get_predictor()
    result = predictor.predict(source, model=model, threshold_scale=scale)
    explanation, similar = {"lines": {}, "functions": {}}, []
    if not isinstance(predictor, MockPredictor):
        from src.explain import explain_all

        explanation = explain_all(source, predictor, model=model)
        similar = predictor.similar(source, k=3)
    st.session_state["scan"] = {"source": source, "name": name, "result": result,
                                "explanation": explanation, "similar": similar}


def read_upload(upload) -> str:
    try:
        return ui.read_upload(upload, MAX_UPLOAD_MB)
    except ui.UploadTooLarge as exc:
        raise InputError(str(exc)) from exc


def _use_text():
    st.session_state["input_source"] = "text"


def _use_upload():
    st.session_state["input_source"] = "upload"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
predictor, mock_reason = get_predictor()

# Deep links for the demo: ?sample=02_reentrancy_bank.sol&scan=1 loads (and scans) a sample.
qp = st.query_params
link_sample = qp.get("sample")
if link_sample and not link_sample.endswith(".sol"):
    link_sample += ".sol"  # ?sample=02_reentrancy_bank works as well as ?sample=02_reentrancy_bank.sol
if link_sample in SAMPLE_LABELS and st.session_state.get("deep_link_done") != link_sample:
    st.session_state["deep_link_done"] = link_sample
    st.session_state["code"] = (SAMPLES / link_sample).read_text(encoding="utf-8")
    st.session_state["source_name"] = link_sample
    st.session_state["loaded_sample"] = link_sample
    st.session_state["input_source"] = "text"
    if qp.get("scan") == "1":
        try:
            run_scan(st.session_state["code"], link_sample, predictor.default_model, 1.0)
        except InputError:
            pass

with st.sidebar:
    st.markdown("## 🛡️ Muhafiz.sol")
    st.caption("*Muhafiz* means guardian in Urdu. This tool reads a smart contract and points out "
               "weaknesses it has learned to recognise.")
    sample = st.selectbox("Try a sample contract", ["(none)"] + list(SAMPLE_LABELS),
                          format_func=lambda k: SAMPLE_LABELS.get(k, "Choose a sample..."))
    if sample != "(none)" and st.session_state.get("loaded_sample") != sample:
        st.session_state["code"] = (SAMPLES / sample).read_text(encoding="utf-8")
        st.session_state["loaded_sample"] = sample
        st.session_state["source_name"] = sample
        st.session_state["input_source"] = "text"
    model_names = {"mlp": "Neural network (MLP)", "logreg": "Logistic regression", "forest": "Random forest",
                   "knn": "Nearest neighbours", "tree": "Decision tree", "mock": "Mock"}
    model = st.radio("Model", predictor.available_models, index=predictor.available_models.index(
        predictor.default_model), format_func=lambda m: model_names.get(m, m) +
        (" - recommended" if m == predictor.default_model else ""))
    scale = st.slider("Sensitivity", 0.5, 1.5, 1.0, 0.05,
                      help="Moves every alert level. Left = flag more (fewer misses, more false alarms). "
                           "Right = flag less.", format="%.2fx")
    st.caption("◀ more alerts · stricter ▶ (1.00x = levels chosen on validation data)")
    with st.expander("How to read the result"):
        st.markdown("1. **Risk badge**: low / medium / high, using the rule shown under it.\n"
                    "2. **Bars**: one per weakness type. If a bar crosses the dark tick, it is flagged.\n"
                    "3. **Highlighted lines**: the lines whose removal lowers the score the most.\n"
                    "4. **Similar contracts**: real Ethereum contracts the model learned from.")
    with st.expander("Model card"):
        card = ROOT / "docs" / "model_card.md"
        st.markdown(card.read_text(encoding="utf-8") if card.exists() else "Model card not generated yet.")
    if mock_reason:
        st.warning(f"Demo mode: {mock_reason}")

st.markdown("# 🛡️ Muhafiz.sol — smart contract weakness checker")
st.markdown("Paste a Solidity smart contract, press **Scan**, and see which kinds of security weakness it may "
            "contain and *where*. Results appear in a few seconds.")

tab_scan, tab_metrics, tab_batch, tab_about = st.tabs(
    ["🔍 Scan a contract", "📊 How good is the model?", "📁 Scan many files", "ℹ️ About & limits"])

# --------------------------------------------------------------------------- #
# Scan tab
# --------------------------------------------------------------------------- #
with tab_scan:
    left, right = st.columns([3, 2])
    with left:
        st.text_area("Solidity source code", key="code", height=260, on_change=_use_text,
                     placeholder="pragma solidity ^0.8.0;\ncontract MyContract { ... }")
    with right:
        upload = st.file_uploader("...or upload a .sol file", type=["sol", "txt"], on_change=_use_upload,
                                  help=f"Up to {MAX_UPLOAD_MB} MB. A single flattened file works best.")
        use_upload = upload is not None and st.session_state.get("input_source", "upload") == "upload"
        if upload is not None:
            st.caption(f"Scan will use the uploaded file **{upload.name}**." if use_upload else
                       "Scan will use the code in the text box (chosen after the upload).")
        scan = st.button("🔍 Scan", type="primary", width="stretch")
    if scan:
        try:
            if use_upload:
                source, name = read_upload(upload), upload.name
            else:
                source = st.session_state.get("code", "")
                name = st.session_state.get("source_name", "pasted-contract.sol")
            with st.spinner("Scanning..."):
                run_scan(source, name, model, scale)
        except InputError as exc:
            st.session_state.pop("scan", None)
            st.error(f"🙁 {exc}")

    scan_state = st.session_state.get("scan")
    if scan_state:
        res = scan_state["result"]
        for w in res["warnings"]:
            st.warning(w)
        st.markdown(f"### Result for `{scan_state['name']}`")
        b1, b2 = st.columns([2, 3])
        with b1:
            st.html(ui.risk_badge(res["risk_level"]))
            st.caption("Rule: **low** if no type reaches its alert level; **medium** if at least one does; "
                       "**high** if one is far past it (alert level + 25% of the remaining distance to 100%).")
            st.caption(f"Scored in {res['latency_ms']:.0f} ms with the {model_names.get(res['model'], res['model'])}.")
        with b2:
            st.html(ui.probability_bars(res))

        flagged = [c for c, f in res["flags"].items() if f]
        order = flagged + [c for c in sorted(res["probabilities"], key=lambda k: -res["probabilities"][k])
                           if c not in flagged]
        st.markdown("### Where in the code?")
        chosen = st.selectbox("Explain this weakness type", order, format_func=lambda c: (
            "⚠ " if res["flags"][c] else "") + ui.class_title(c))
        info = ui.CLASS_INFO.get(chosen, {})
        if info:
            st.info(f"**{info['title']}** — {info['plain']}\n\n*Typical fix:* {info['fix']}")
        items = [it for it in scan_state["explanation"].get("lines", {}).get(chosen, [])
                 if it["score"] >= ui.MIN_SHOWN_DROP]
        if items:
            st.markdown("Highlighted = the lines that push this score up the most "
                        "(removing them lowers the probability by the amount shown).")
            st.dataframe(pd.DataFrame([{"rank": i + 1, "line": it["line"], "in function": it["function"] or "-",
                                        "score drop": round(it["score"], 3), "code": it["text"]}
                                       for i, it in enumerate(items)]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No single line stands out for this type (removing any one line barely changes the score).")
        st.html(ui.code_viewer(scan_state["source"], {it["line"]: it["score"] for it in items}))
        funcs = scan_state["explanation"].get("functions", {}).get(chosen, [])
        if funcs:
            with st.expander("Function-level view"):
                st.dataframe(pd.DataFrame([{"function": f["function"], "kind": f["kind"],
                                            "lines": f"{f['start_line']}-{f['end_line']}",
                                            "score drop": round(f["score"], 3)} for f in funcs]),
                             hide_index=True, width="stretch")

        if scan_state["similar"]:
            st.markdown("### Similar known contracts")
            st.caption("Real Ethereum contracts from the training data that look most like yours, with the "
                       "weakness labels the Slither analyser gave them.")
            cols = st.columns(len(scan_state["similar"]))
            for col, s in zip(cols, scan_state["similar"]):
                with col:
                    st.markdown(f"**{s['contract_name']}**  \nsimilarity {s['similarity']:.0%}  \n"
                                f"labels: {', '.join(ui.class_title(l) for l in s['labels'])}  \n"
                                f"[view on Etherscan]({s['etherscan_url']})")
                    st.code(s["snippet"][:400] or "(no preview)", language="solidity")

        d1, d2 = st.columns(2)
        args = (res, scan_state["explanation"], scan_state["similar"], scan_state["name"])
        d1.download_button("⬇ Download report (Markdown)", ui.report_markdown(*args),
                           file_name="muhafiz_scan_report.md", width="stretch")
        d2.download_button("⬇ Download report (JSON)", ui.report_json(*args),
                           file_name="muhafiz_scan_report.json", width="stretch")

# --------------------------------------------------------------------------- #
# Metrics tab (F7) - numbers come from the same CSV files as the report
# --------------------------------------------------------------------------- #
with tab_metrics:
    tables = ROOT / "reports" / "tables"
    st.markdown("### How good is the model?")
    st.markdown("The test-split numbers come from contracts kept out of training and tuning, with near-copies "
                "kept apart. They are the same numbers as in the project report.")
    e1 = tables / "e1_test_summary.csv"
    if e1.exists():
        df = pd.read_csv(e1)
        st.markdown("**Model comparison on the held-out test split** (higher is better; PR-AUC and F1 range 0-1)")
        st.dataframe(df, hide_index=True, width="stretch")
    e2 = tables / "e2_leakage.csv"
    if e2.exists():
        st.markdown("**Why careful splitting matters** - scores when near-copies of contracts are allowed in both "
                    "training and test (random split) versus kept apart (cluster split):")
        st.dataframe(pd.read_csv(e2), hide_index=True, width="stretch")
    e3 = tables / "e3_smartbugs_recall.csv"
    if e3.exists():
        st.markdown("**Independent check on SmartBugs Curated** (contracts annotated by humans; recall = share "
                    "of known-vulnerable contracts that were flagged, with 95% bootstrap interval). 42 of the 143 "
                    "SmartBugs contracts also occur in the training data, so recall is shown for all contracts and "
                    "for those with no overlap (the honest number):")
        st.dataframe(pd.read_csv(e3), hide_index=True, width="stretch")
    e3b = tables / "e3b_hit_rate.csv"
    if e3b.exists():
        st.markdown("**Do the highlights point at the right lines?** (share of SmartBugs contracts whose annotated "
                    "vulnerable line is among the top-1 / top-3 highlighted lines)")
        st.dataframe(pd.read_csv(e3b), hide_index=True, width="stretch")
    if not e1.exists():
        st.info("Evaluation tables have not been generated yet (run `python -m src.run_experiments`).")

# --------------------------------------------------------------------------- #
# Batch tab (F13)
# --------------------------------------------------------------------------- #
with tab_batch:
    st.markdown("### Scan several contracts and rank them by risk")
    files = st.file_uploader("Upload .sol files", type=["sol", "txt"], accept_multiple_files=True, key="batch")
    if files and st.button("Scan all", type="primary"):
        rows = []
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            try:
                r = predictor.predict(read_upload(f), model=model, threshold_scale=scale)
                top = max(r["probabilities"], key=r["probabilities"].get)
                rows.append({"file": f.name, "risk": r["risk_level"], "highest type": ui.class_title(top),
                             "probability": round(r["probabilities"][top], 3),
                             "flagged types": ", ".join(ui.class_title(c) for c, v in r["flags"].items() if v) or "-"})
            except InputError as exc:
                rows.append({"file": f.name, "risk": "error", "highest type": str(exc), "probability": None,
                             "flagged types": "-"})
            bar.progress((i + 1) / len(files))
        order = {"high": 0, "medium": 1, "low": 2, "error": 3}
        out = pd.DataFrame(rows).sort_values(["risk", "probability"], key=lambda s: s.map(order) if s.name == "risk"
                                             else -s.fillna(0))
        st.dataframe(out, hide_index=True, width="stretch")
        st.download_button("⬇ Download table (CSV)", out.to_csv(index=False), file_name="muhafiz_batch.csv")

# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("""
### What Muhafiz.sol does
It turns the contract's code into numbers (which words and short word-sequences appear), feeds them to a
model trained on **tens of thousands of real Ethereum contracts**, and returns a probability for each weakness
type. It then hides one function or line at a time to see which parts of the code the score depends on.

### What it does **not** do
- It does not replace a professional audit and never says a contract is "safe".
- It does not run the code, compile it, or connect to the blockchain.
- It does not create attacks; it only points at weaknesses.

### Known limits
- The training labels came from an automatic analyser (Slither), not from human experts, so the model
  mostly learns to imitate that tool - including its mistakes.
- Some weakness types (integer overflow, denial of service, front running, timestamp tricks) have no
  matching training label, so the model cannot flag them at all.
- Old-style (Solidity 0.4) contracts and very small contracts look different from most training data.
- A low score can still hide a real vulnerability (false negatives).
""")

st.divider()
st.caption("⚖️ " + ui.DISCLAIMER)
