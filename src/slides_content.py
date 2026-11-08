"""
Slide-by-slide content of the 20-slide deck (blueprint 19.2). Numbers come from Facts.
"""

from __future__ import annotations

import json

import pandas as pd
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from src.make_slides import AMBER, FIG, INK, LIGHT, MUTED, NAVY, RED, TEAL, WHITE, Deck
from src.config import path
from src.report_utils import MODEL_NAMES, Facts, pct

RGB_ICE = RGBColor(0xCA, 0xDC, 0xFC)



def _short(case_id: str) -> str:
    """File name of a case study, with long address-style names shortened to fit a card title."""
    name = case_id.split("/")[-1]
    return f"{name[:10]}...{name[-8:]}" if name.startswith("0x") and len(name) > 26 else name

def build(deck: Deck, f: Facts) -> None:
    I = Inches
    e1 = f.t["e1_test_summary"]
    e2 = f.t["e2_leakage"]
    e3 = f.t["e3_smartbugs_recall"]
    s3 = f.t["e3_summary"]
    hr = f.t["e3b_hit_rate"]
    prim = e1[e1.key == f.primary].iloc[0]
    pooled = s3[s3.key == f.primary]["smartbugs_pooled_recall_confirmed"].iloc[0]
    d = f.dedup

    # 1 Title -------------------------------------------------------------
    s = deck.slide(dark=True)
    deck.text(s, "MUHAFIZ.SOL", I(0.8), I(2.0), I(11.5), I(1.2), 60, WHITE, True, "Cambria")
    deck.text(s, "A machine-learning guardian for Ethereum smart contracts", I(0.8), I(3.2), I(11.5), I(0.7), 26,
              RGB_ICE)
    deck.text(s, ["AI2002 Artificial Intelligence - Semester Project, Fall 2026 - Path A (software)",
                  "Group 39: Sohaib (24I-3108) - Shaiman (24I-3074) - Talha Sami (24I-3118)"],
              I(0.8), I(4.6), I(11.5), I(1.0), 16, WHITE)
    deck.notes(s, "Muhafiz means guardian in Urdu; .sol is the Solidity file extension.")

    # 2 Problem -------------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Money lives in code that cannot be patched")
    deck.card(s, I(0.6), I(1.7), I(5.9), I(4.9), "The shopkeeper's mistake",
              "1. Check the ledger: customer is owed 100 rupees.\n2. Hand over the 100 rupees.\n3. Only then write "
              "'owed: 0'.\n\nA clever customer asks again between steps 2 and 3 - and is paid again and again. In a "
              "smart contract this is reentrancy, the pattern behind famous thefts.", "!")
    deck.stat(s, I(7.1), I(1.8), I(5.6), "US$3.4 bn", "stolen in crypto, Jan - early Dec 2025 (Chainalysis)")
    deck.stat(s, I(7.1), I(3.6), I(5.6), "~12 %", "of stolen value from contract-code exploits (TRM Labs via secondary source)",
              AMBER)
    deck.text(s, "Audits are slow and costly; students and small teams often deploy unreviewed.", I(7.1), I(5.6),
              I(5.6), I(0.9), 15, INK, italic=True)

    # 3 Motivation / Pakistan ---------------------------------------------
    s = deck.slide()
    deck.title(s, "Why now, and why here")
    deck.card(s, I(0.6), I(1.7), I(3.9), I(4.6), "Pakistan 2026",
              "Virtual Assets Act passed 6 March 2026; PVARA licenses the sector. Pakistan ranked 3rd in Chainalysis's "
              "2025 adoption index.", "1")
    deck.card(s, I(4.75), I(1.7), I(3.9), I(4.6), "Local demand",
              "New, small Web3 teams need cheap first-pass screening before paying for a full audit.", "2")
    deck.card(s, I(8.9), I(1.7), I(3.9), I(4.6), "Honest positioning",
              "A student triage aid. Not a certification, not a compliance product, not affiliated with PVARA.", "3")

    # 4 Solution -------------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Paste a contract, get an explained risk report in seconds")
    deck.image(s, FIG / "app_1_reentrancy_result.png", I(0.6), I(1.55), w=I(7.6))
    deck.bullets(s, [("Probabilities ", f"for {len(f.classes)} weakness types"), ("Risk level ", "low / medium / high with its rule"),
                     ("Highlighted lines ", "that drive the score"), ("Similar contracts ", "from 45K real deployments"),
                     ("Download ", "a Markdown / JSON report")], I(8.6), I(1.8), I(4.2), I(4.5), 17)

    # 5 Datasets -----------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Two datasets: one to learn, one to check honestly")
    deck.card(s, I(0.6), I(1.7), I(5.9), I(4.6), "A - Slither Audited Contracts",
              f"{f.funnel['rows_in_raw_shards']:,} verified Ethereum contracts labelled by the Slither analyser "
              f"(38 detectors -> {len(f.classes)} families). Hugging Face, MIT licence. Used for training, validation "
              "and the in-distribution test.", "A")
    deck.card(s, I(6.85), I(1.7), I(5.9), I(4.6), "B - SmartBugs Curated",
              "143 vulnerable contracts annotated line by line by people (207 vulnerabilities). GitHub, Apache-2.0. "
              "Test only - never used for training or tuning; all annotation comments stripped first.", "B")

    # 6 Pipeline -----------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Pipeline: offline training, online scanning")
    deck.image(s, FIG / "pipeline.png", I(0.6), I(1.45), w=I(12.1))

    # 7 Preprocessing / dedup ---------------------------------------------
    s = deck.slide()
    deck.title(s, "Clean data first: duplicates would have faked our scores")
    deck.stat(s, I(0.6), I(1.7), I(3.9), f"{d['exact_duplicate_pct']:.0f} %", "of contracts are exact or renamed copies (removed)")
    deck.stat(s, I(4.75), I(1.7), I(3.9), f"{pct(d['random_split_leakage_rate'], 0)}",
              "of test contracts would have a clone in training under a random split", RED)
    deck.stat(s, I(8.9), I(1.7), I(3.9), "0 %", "under our clone-group split", TEAL)
    deck.bullets(s, ["Comments blanked, line numbers kept; literals -> <STR>/<NUM>",
                     "MinHash + LSH on token 5-grams (Jaccard >= 0.8) -> clone groups; giant template families -> train",
                     "Label 4 resolved: 'safe' in big/small, divide-before-multiply in 'all'; 0 failed Slither runs",
                     f"Split {f.sizes['train']:,} / {f.sizes['val']:,} / {f.sizes['test']:,}; TF-IDF fitted on train only"],
                 I(0.6), I(4.2), I(12.1), I(2.6), 16)

    # 8 Technique -----------------------------------------------------------
    s = deck.slide()
    deck.title(s, "AI technique: multi-label classification on code tokens")
    deck.card(s, I(0.6), I(1.7), I(5.9), I(4.8), "Why machine learning",
              "Works on raw source with no compiler, scores in under a second, handles code that does not compile, "
              "and ranks every weakness type by probability. Limit: it learns Slither's labels, so we test it on "
              "human annotations.", "?")
    deck.card(s, I(6.85), I(1.7), I(5.9), I(4.8), "Input -> output",
              "Contract -> tokens -> TF-IDF of 1-3-grams (20,000 features) -> one probability per class -> "
              "per-class threshold (F2 on validation) -> flags and risk level -> occlusion explanation.", ">")

    # 9 Models compared ------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Six model families, one fair comparison")
    sel = f.t["model_selection_val"].copy()
    sel["model"] = sel["model"].map(MODEL_NAMES)
    sel["fit_seconds"] = sel["fit_seconds"].map(lambda v: f"{v / 60:.1f}")
    deck.table(s, sel[["model", "val_macro_pr_auc", "val_macro_f2", "fit_seconds"]].rename(columns={
        "val_macro_pr_auc": "val macro PR-AUC", "val_macro_f2": "val macro F2", "fit_seconds": "train time (min)"}),
               I(0.6), I(1.7), I(7.4), size=14)
    deck.text(s, [f"Primary model: {MODEL_NAMES[f.primary]}", "",
                  "Chosen by validation macro PR-AUC only - test and SmartBugs untouched until the git tag model-v1.",
                  "", "Train times are wall-clock on one laptop CPU, with several jobs sharing it."],
              I(8.4), I(1.9), I(4.4), I(3.5), 17, NAVY)

    # 10 MLP --------------------------------------------------------------
    s = deck.slide()
    deck.title(s, "The neural network: how it learns")
    deck.image(s, FIG / "mlp_learning_curves.png", I(0.6), I(1.6), w=I(7.6))
    deck.bullets(s, ["20,000 -> 512 -> 128 -> 7, ReLU, dropout 0.3", "Sigmoid per class (several weaknesses at once)",
                     "Weighted binary cross-entropy for rare classes", "Backpropagation + Adam; early stopping on val PR-AUC",
                     f"3 seeds: val PR-AUC {f.thresholds['mlp_seed_val_pr_auc']['mean']:.3f} +/- "
                     f"{f.thresholds['mlp_seed_val_pr_auc']['std']:.3f}"], I(8.5), I(1.8), I(4.3), I(4.8), 15)

    # 11 thresholds & risk ---------------------------------------------------
    s = deck.slide()
    deck.title(s, "Thresholds favour recall; the risk rule is explicit")
    thv = f.t["thresholds_val"]
    tp = thv[thv.model == f.primary][["class", "threshold", "precision", "recall"]]
    deck.table(s, tp, I(0.6), I(1.7), I(6.6), size=13)
    deck.card(s, I(7.6), I(1.7), I(5.2), I(4.6), "Risk level",
              "LOW - no class reaches its threshold\nMEDIUM - at least one does\nHIGH - one reaches t + 0.25 x (1 - t)\n\n"
              "Thresholds maximise F2 on validation (a miss costs more than a false alarm), with precision >= 0.2.", "R")

    # 12 E1 ---------------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Results: model comparison on unseen contract families")
    order = [k for k in ["prior", "logreg", "tree", "forest", "knn", "mlp"] if k in set(e1.key)]
    e1o = e1.set_index("key").loc[order]
    deck.bar_chart(s, [MODEL_NAMES[k].replace(" neural network", "").replace("Majority / prior baseline", "Prior baseline")
                       for k in order], {"macro PR-AUC": e1o["macro_pr_auc"], "macro F1": e1o["macro_f1"]},
                   I(0.6), I(1.5), I(8.2), I(5.2), "Test split (cluster-aware)", [NAVY, TEAL])
    deck.stat(s, I(9.2), I(1.8), I(3.6), f"{prim['macro_pr_auc']:.3f}", f"macro PR-AUC of the {MODEL_NAMES[f.primary]} "
              f"{prim['macro_pr_auc_ci']}")
    deck.text(s, f"Prior baseline: {e1[e1.key == 'prior'].iloc[0]['macro_pr_auc']:.3f} PR-AUC but "
                 f"{e1[e1.key == 'prior'].iloc[0]['label_accuracy']:.0%} label accuracy - why accuracy is not our metric.",
              I(9.2), I(4.0), I(3.6), I(2.0), 14, MUTED)

    # 13 E2 + E3 -----------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Leakage ablation and the independent SmartBugs check")
    e2o = e2.set_index("key").loc[[k for k in order if k in set(e2.key)]]
    deck.bar_chart(s, [MODEL_NAMES[k].split(" ")[0] for k in e2o.index],
                   {"cluster split": e2o["cluster_test_macro_pr_auc"], "random split": e2o["random_test_macro_pr_auc"]},
                   I(0.6), I(1.5), I(6.2), I(5.0), "E2: macro PR-AUC, cluster vs random split", [NAVY, AMBER])
    conf = e3[e3.status == "confirmed"]
    deck.bar_chart(s, [c.replace("_", " ").replace("unchecked low level calls", "unchecked calls") for c in conf["smartbugs_category"]],
                   {"recall": conf["recall"]}, I(7.0), I(1.5), I(5.8), I(3.7), "E3: SmartBugs recall (confirmed classes)",
                   [TEAL], fmt="0%")
    deck.text(s, f"Pooled recall {pct(pooled)} over {int(conf['contracts'].sum())} human-annotated contracts; "
                 "wide intervals, strong domain shift (tiny Solidity 0.4 examples).", I(7.0), I(5.35), I(5.8), I(1.2),
              13, MUTED)

    # 14 Explanation -----------------------------------------------------
    s = deck.slide()
    deck.title(s, "Explanation: hide a line, watch the score drop")
    allr = hr.iloc[0]
    deck.card(s, I(0.6), I(1.7), I(6.0), I(4.8), "Two-stage occlusion",
              "1. Blank each function, measure the probability drop.\n2. Inside the top functions, blank each line.\n"
              "3. Highlight the lines with the biggest drops.\n\nExact fast trick: only n-grams touching the removed "
              "span change, so counts(full) - counts(L+S+R) + counts(L+R).", "E")
    deck.stat(s, I(7.0), I(1.8), I(5.6), pct(allr["strict@3"], 0), f"of SmartBugs contracts: annotated line in the top 3 "
              f"(random lines: {pct(allr['random@3'], 0)})")
    deck.stat(s, I(7.0), I(3.9), I(5.6), pct(allr["relaxed@3"], 0), "top-3 line in the same function as the bug", AMBER)

    # 15 App -----------------------------------------------------------------
    s = deck.slide()
    deck.title(s, "The Streamlit app")
    deck.image(s, FIG / "app_2_highlighted_code.png", I(0.6), I(1.5), w=I(7.8))
    deck.bullets(s, ["One command: streamlit run app/app.py", "Paste, upload or pick one of 6 samples",
                     "Plain-language explanation + typical fix", "Sensitivity slider, model toggle, batch scan",
                     "Metrics tab reads the same tables as the report", "Disclaimer always visible"],
                 I(8.8), I(1.8), I(4.0), I(4.8), 15)

    # 16 Testing -----------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Testing: evidence, not claims")
    tj = path("docs_dir") / "test_evidence.json"
    tests = json.loads(tj.read_text()) if tj.exists() else {"total": 0, "passed": 0}
    deck.stat(s, I(0.6), I(1.7), I(3.9), f"{tests['passed']}/{tests['total']}", "automated tests passing")
    deck.stat(s, I(4.75), I(1.7), I(3.9), "A1-A6 B1-B6 C2-C5", "acceptance tests passed (C1 and C6 need the team's laptops)",
              NAVY, size=26)
    deck.stat(s, I(8.9), I(1.7), I(3.9), "10 + fuzz", "fixed inputs and random inputs: no crash", TEAL)
    deck.bullets(s, ["No annotation text and no clone group crosses splits (automatic tests)",
                     "Model frozen before test evaluation - the evaluation script checks the git tag",
                     "App output equals predict() exactly; sensitivity slider is monotone",
                     "Fast occlusion equals full re-vectorisation (exactness test)"], I(0.6), I(4.3), I(12.1), I(2.5), 16)

    # 17 Case studies -----------------------------------------------------
    s = deck.slide()
    deck.title(s, "Qualitative results: where it works and where it fails")
    succ = [c for c in f.cases if c["kind"] == "success"][:2]
    fail = [c for c in f.cases if c["kind"] == "failure"][:2]
    y = 1.7
    for c in succ:
        deck.card(s, I(0.6), I(y), I(5.9), I(2.25), f"Hit - {_short(c['id'])}",
                  f"{c['class']}, p = {c['probability']:.2f}. Top line: {c['top_line_text'][0][:70] if c['top_line_text'] else '-'}",
                  "+")
        y += 2.45
    y = 1.7
    for c in fail:
        deck.card(s, I(6.85), I(y), I(5.9), I(2.25), f"Miss - {_short(c['id'])}",
                  f"{c['class']}, p = {c['probability']:.2f}. Bug at: {c['vulnerable_text'][0][:60] if c['vulnerable_text'] else '-'}",
                  "-", head_color=RED)
        y += 2.45

    # 18 Limitations --------------------------------------------------------
    s = deck.slide()
    deck.title(s, "Limitations and ethics")
    deck.card(s, I(0.6), I(1.7), I(5.9), I(4.9), "Limits",
              "Learns Slither's labels (and its mistakes). Cannot flag overflow, DoS, front running or timestamp bugs - "
              "no training label. Weaker on tiny Solidity 0.4 code. Token patterns, not program semantics. A low score "
              "can still hide a real bug.", "!")
    deck.card(s, I(6.85), I(1.7), I(5.9), I(4.9), "Ethics",
              "Defensive only: detects and explains, never generates exploits. Never certifies safety; disclaimer in "
              "app, report and slides. Public contracts, no personal data, licences respected, datasets cited. No "
              "regulatory claims.", "E")

    # 19 Conclusion -------------------------------------------------------
    s = deck.slide(dark=True)
    deck.text(s, "Conclusion", I(0.8), I(0.7), I(11.5), I(0.9), 40, WHITE, True, "Cambria")
    deck.bullets(s, [f"Working ML triage tool: {MODEL_NAMES[f.primary]} with {prim['macro_pr_auc']:.3f} macro PR-AUC on unseen families",
                     f"Independent check: {pct(pooled)} recall on human-annotated SmartBugs (confirmed classes)",
                     "Honest evaluation: clone-aware split, frozen model, leakage measured",
                     "Next: code-model embeddings, human labels for missing classes, cloud deployment"],
                 I(0.8), I(2.0), I(11.5), I(4.2), 20, WHITE)

    # 20 Team & questions ---------------------------------------------------
    s = deck.slide()
    deck.title(s, "Team and questions")
    deck.card(s, I(0.6), I(1.7), I(3.9), I(3.6), "Talha Sami", "Phase 1 - data, labels, de-duplication, splits, features",
              "1")
    deck.card(s, I(4.75), I(1.7), I(3.9), I(3.6), "Sohaib", "Phase 2 - models, MLP, thresholds, experiments E1-E4", "2")
    deck.card(s, I(8.9), I(1.7), I(3.9), I(3.6), "Shaiman", "Phase 3 - explanation, Streamlit app, testing, delivery", "3")
    deck.text(s, "Questions?", I(0.6), I(5.6), I(12.1), I(0.9), 36, NAVY, True, "Cambria", align=PP_ALIGN.CENTER)

