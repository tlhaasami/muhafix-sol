"""
Report front matter and chapters 1-3 (Phase 1 content: problem, methodology, data).
All numbers come from ``Facts`` (reports/tables and build artefacts).
"""

from __future__ import annotations

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.config import load_config, path
from src.report_utils import MODEL_NAMES, Facts, model_lc, pct

FIG = path("figures_dir")


def front_matter(doc, f: Facts) -> None:
    from src.make_report import ACCENT, GREY

    doc.p("AI2002: ARTIFICIAL INTELLIGENCE", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY)
    doc.p("Semester Project, Fall 2026 - Path A (Software-Based)", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY)
    t = doc.p("MUHAFIZ.SOL", size=30, align=WD_ALIGN_PARAGRAPH.CENTER, color=ACCENT)
    t.runs[0].bold = True
    doc.p("Smart Contract Vulnerability Detector", size=16, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.p("Final Project Report", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY)
    doc.p("")
    doc.table(pd.DataFrame([
        ["Group", "39"],
        ["Members", "Sohaib (24I-3108), Shaiman (24I-3074), Talha Sami (24I-3118)"],
        ["Path", "A - software (hardware not applicable)"],
        ["Phase owners", "Phase 1 data: Talha Sami - Phase 2 models: Sohaib - Phase 3 app and delivery: Shaiman"],
        ["Datasets", "Slither Audited Smart Contracts (training); SmartBugs Curated (external test)"],
        ["Deliverables", "Source code, Streamlit app (streamlit run app/app.py), this report, slides"],
    ], columns=["Item", "Details"]), "Project summary", widths=[3.5, 13])
    doc.h("Abstract", 2)
    e1 = f.t["e1_test_summary"]
    e1p = f.e1(f.primary) if e1 is not None else None
    e3 = f.t["e3_summary"]
    pooled = e3[e3.key == f.primary]["smartbugs_pooled_recall_confirmed"].iloc[0] if e3 is not None else None
    text = (
        "Smart contracts hold real money and cannot be patched after deployment, yet professional audits are slow "
        "and expensive. Muhafiz.sol is a machine-learning triage tool that reads the Solidity source of an Ethereum "
        f"contract and returns, within seconds, a probability for each of {len(f.classes)} weakness families, an "
        "overall risk level, the lines that drive the prediction and the most similar known contracts. Models are "
        f"trained on {f.funnel['after_exact_deduplication']:,} unique contracts from the Slither Audited Smart "
        "Contracts dataset after exact and near-duplicate (MinHash) grouping and a clone-aware split, and are tested "
        "independently on the human-annotated SmartBugs Curated benchmark. Six model families are compared. ")
    if e1p is not None:
        text += (f"The selected {model_lc(MODEL_NAMES[f.primary])} reaches a macro PR-AUC of {e1p['macro_pr_auc']:.3f} "
                 f"(95 % CI {e1p['macro_pr_auc_ci']}) on the held-out test split")
        text += (f" and flags {pct(pooled)} of the SmartBugs contracts in the three categories with a confirmed "
                 "label mapping. " if pooled is not None else ". ")
    text += ("We show how much clone leakage would inflate the scores, measure whether the highlighted lines hit the "
             "annotated vulnerable lines, and state the limits plainly: the model learns tool-generated labels, "
             "cannot see weakness types that have no training label, and is not a substitute for an audit.")
    doc.p(text)
    doc.page_break()
    doc.h("Contents", 2)
    doc.toc()
    doc.page_break()


def chapter1(doc, f: Facts) -> None:
    doc.h("1. Introduction")
    doc.h("1.1 Problem statement", 2)
    doc.p("A smart contract is a small program stored on a blockchain such as Ethereum. It holds and moves digital "
          "money according to fixed rules and, once deployed, generally cannot be changed. A coding mistake therefore "
          "cannot be quietly fixed afterwards: if an attacker finds it, the funds the contract holds can be drained. "
          "Security audits exist to find such flaws before deployment, but they are slow and costly, so many students "
          "and small teams deploy with little or no review. They need a fast way to find the most common weakness "
          "patterns and to see where in the code they occur.")
    doc.p("A classic example is the refund mistake. A shopkeeper checks the ledger (the customer is owed 100 rupees), "
          "hands over the money, and only then writes zero in the ledger. A customer who repeats the request between "
          "the second and third steps is paid again and again. In smart contracts this ordering mistake is called "
          "reentrancy. Muhafiz.sol should flag such a contract and highlight the lines where money is sent before the "
          "balance is updated.")
    doc.h("1.2 Real-world justification", 2)
    doc.p("Chainalysis reports more than US$3.4 billion in cryptocurrency stolen between January and early December "
          "2025, and CertiK's Hack3d report for 2025 records US$3.35 billion lost to hacks, scams and exploits across "
          "310 incidents, about US$1.70 billion of it on Ethereum. An honest qualification matters: most of the 2025 "
          "dollar value came from a few infrastructure and key-compromise attacks, not from contract-code bugs. TRM "
          "Labs figures, as summarised by a secondary source, attribute about 12 % of stolen value (52 incidents) to "
          "code exploits. Contract-code flaws are therefore a smaller share of the money lost, but a recurring and "
          "preventable one - exactly the category a pre-deployment check addresses. (The secondary figures must be "
          "re-checked against the primary reports before submission; risk R11.)")
    doc.p("Pakistan's Parliament passed the Virtual Assets Act 2026 on 6 March 2026, establishing the Pakistan Virtual "
          "Assets Regulatory Authority (PVARA) to license virtual-asset service providers and give regulatory clarity "
          "for token launches and DeFi protocols; the same report places Pakistan third in Chainalysis's 2025 adoption "
          "index. A newly licensed sector creates demand from small local teams for low-cost screening. Muhafiz.sol is "
          "a student research tool: we do not claim that PVARA requires contract scanning, and it is not a compliance "
          "product.")
    doc.h("1.3 Why machine learning?", 2)
    doc.p("Rule-based analysers such as Slither need code that compiles with the right compiler version and apply "
          "hand-written detectors. A trained model reads raw source text, needs no compiler, returns a ranked "
          "probability for every class within a second and can score code that does not compile. Because our "
          "training labels were produced by Slither, the model largely learns to approximate Slither - which is why "
          "we position it as a triage aid and test it on an independently annotated benchmark.")
    doc.h("1.4 Selected path", 2)
    doc.p("The project follows Path A (software only). No hardware, sensors or actuators are used and the project is "
          "never presented as a hybrid. The core functionality depends entirely on trained models: without them the "
          "application produces no output, and no static-analysis rule runs at prediction time.")
    doc.h("1.5 Objectives and how they were met", 2)
    doc.table(pd.DataFrame([
        ["O1", "Clean, leak-free pipeline from two public datasets", "Automatic tests A1 (no annotation text) and A2 (no clone group in two splits) pass"],
        ["O2", "Train and fairly compare at least four model families", "Six families compared on the same split (chapter 5, E1)"],
        ["O3", "Evaluate on an independent annotated benchmark", "Per-class recall with bootstrap intervals on SmartBugs (E3)"],
        ["O4", "Show how much clone leakage inflates scores", "Random vs cluster-aware split for all models (E2)"],
        ["O5", "Highlight risky lines and measure correctness", "Line hit-rate@1/@3 against annotated lines (E3b)"],
        ["O6", "Working Streamlit app with the full flow", "streamlit run app/app.py; ten fixed inputs tested (chapter 7)"],
        ["O7", "Honest reporting including limitations", "Chapter 8, model card, data card and in-app disclaimer"],
    ], columns=["ID", "Objective", "Evidence"]), "Measurable objectives (blueprint 4.1) and their evidence",
              widths=[1.2, 6.5, 9])
    doc.h("1.6 Scope, users, inputs and outputs", 2)
    doc.bullets([
        ("In scope: ", "Solidity source pasted or uploaded as one file; multi-label weakness prediction; risk level; "
                       "line highlighting; similar contracts; batch ranking of several files."),
        ("Out of scope: ", "hardware; exploit generation or automatic fixing; scanning deployed addresses; bytecode; "
                           "non-Ethereum chains; running Slither or a compiler in the app; any claim of certification."),
        ("Users: ", "student and freelance Solidity developers, small Web3 / token-launch teams, auditors deciding what "
                    "to review first, launchpad review staff."),
        ("Input -> output: ", "one .sol file -> per-class probabilities, flags, low/medium/high risk, top-k risky lines, "
                              "three similar training contracts and a downloadable report."),
        ("Decision supported: ", "deploy, fix specific functions first, or send for a full manual audit."),
    ])


def chapter2(doc, f: Facts) -> None:
    doc.h("2. Methodology and System Architecture")
    doc.p("Figure 1 shows the complete workflow. The left block is the offline pipeline that builds the data and "
          "trains, freezes and evaluates the models; the right block is what runs when a user presses Scan. The app "
          "reuses exactly the same cleaning and feature code as training, so a contract is represented identically "
          "in both places.")
    doc.figure(FIG / "pipeline.png", "Muhafiz.sol pipeline (training, evaluation and application)", 16.5)
    doc.h("2.1 Components", 2)
    doc.table(pd.DataFrame([
        ["Data loader", "Dataset A column-range download from the Hub; Dataset B cloned at a pinned commit", "ingest_slither.py, build_gold.py"],
        ["Preprocessor", "Line-preserving comment removal, literal normalisation, tokenisation", "preprocess.py"],
        ["Labels", "Detector -> family mapping, rare-class rule, SmartBugs mapping", "labels.py"],
        ["Splitter", "SHA-256 exact groups, MinHash-LSH near-duplicate groups, stratified group split, random split", "splits.py"],
        ["Feature builder", "TF-IDF 1-3-grams fitted on train only; SVD-512", "features.py"],
        ["Models", "Prior, logistic regression, decision tree, random forest, KNN, MLP (PyTorch)", "models.py, train.py"],
        ["Evaluator", "Metrics, bootstrap intervals, thresholds, calibration, experiments E1-E5", "evaluate.py, run_experiments.py, eval_explain.py"],
        ["Predictor", "predict() and similar() used by the app", "predict.py"],
        ["Explainer", "Two-stage occlusion: functions, then lines", "explain.py"],
        ["Application", "Input, scan, result page, metrics tab, batch scan, disclaimer", "app/app.py, app/components.py"],
    ], columns=["Component", "Responsibility", "Code (src/ unless stated)"]), "System components",
              widths=[2.8, 8.7, 5.3])
    doc.h("2.2 Interface contract between the phases", 2)
    doc.p("The interfaces of blueprint section 7.4 were fixed first so the three phases could proceed in parallel; "
          "the app was developed against a MockPredictor with the same output schema.")
    doc.table(pd.DataFrame([
        ["clean_source(text)", "raw Solidity", "same text with comments blanked, identical line count"],
        ["tokenize(text)", "cleaned text", "list of tokens"],
        ["predict(source)", "raw Solidity", "probability and flag per class, risk_level, latency_ms (+ warnings, truncation flag)"],
        ["explain(source, class, top_k)", "raw Solidity, class, k", "list of {line, score, text, function}"],
        ["similar(source, k)", "raw Solidity, k", "training contracts with labels, similarity and Etherscan link"],
    ], columns=["Function", "Input", "Output"]), "Interface contract", widths=[4.5, 3.5, 8.8])
    doc.h("2.3 Experimental protocol", 2)
    doc.numbered([
        "Build the data pack once (python -m src.build_data); the TF-IDF vocabulary, IDF weights and SVD are fitted on the training split only.",
        "Tune every model family on the validation split only (python -m src.train).",
        "Choose per-class thresholds (F2 with a precision guard) and the calibration on validation data.",
        "Freeze the model, thresholds and calibration and mark the freeze with the git tag model-v1.",
        "Only then evaluate on the test split and on SmartBugs; python -m src.run_experiments refuses to run otherwise.",
    ])


def chapter3(doc, f: Facts) -> None:
    d = f.dedup
    cfg = load_config()
    doc.h("3. Dataset Description")
    doc.h("3.1 Dataset A: Slither Audited Smart Contracts (training)", 2)
    doc.table(pd.DataFrame([
        ["Source", "huggingface.co/datasets/mwritescode/slither-audited-smart-contracts (raw shards and split files)"],
        ["Creator", "Rossini, M. (2022), University of Bologna course project"],
        ["Construction", "Verified Ethereum contracts (Smart Contract Sanctuary / Etherscan), flattened and analysed with Slither (38 detectors)"],
        ["Licence", "MIT for the repository; contracts keep their original licences; academic, non-commercial use"],
        ["Features used", "source_code (input); raw Slither detector list (target). Bytecode is not downloaded."],
        ["Size", f"{f.funnel['rows_in_raw_shards']:,} contracts; {f.funnel['after_exact_deduplication']:,} unique after exact de-duplication"],
        ["Target", f"multi-label, {len(f.classes)} families: {', '.join(f.classes)}"],
    ], columns=["Item", "Details"]), "Dataset A", widths=[3, 13.8])
    doc.p("Acquisition. The Hub repository uses a loading script that recent versions of the datasets library no "
          "longer execute, and its automatic parquet export is marked partial. We therefore read the nine raw shards "
          "directly, fetching only the byte ranges of the address, source and Slither-result columns (the unused "
          "bytecode column is 44 % of every shard). Keeping the raw Slither JSON let us re-derive every label instead "
          "of trusting the pre-collapsed integer ids.")
    doc.h("3.2 Label semantics: what does label 4 mean?", 2)
    l4 = f.t["label4_audit"]
    if l4 is not None:
        doc.table(l4[["config", "contracts", "id4_name", "id4_share", "id4_with_other_ids_share", "max_id_seen",
                      "safe_id"]], "Upstream integer ids re-created from the raw Slither output (label-4 audit)")
    doc.p("The loading script uses a different list of class names for each configuration. In big and small, id 4 is "
          "safe and never appears together with another id. In all - the configuration the dataset viewer shows by "
          "default - id 4 is the divide-before-multiply detector and safe is id 38. The blueprint's two observations "
          "(\"4 means safe\" and \"4 appears with other ids up to 38\") are therefore both true, but about different "
          "configurations. Safe is not used as a class: a contract has no flagged weakness when every class is below "
          "its threshold. We also checked for failed Slither runs, which the loader would silently label safe: none "
          f"of the {f.funnel['rows_in_raw_shards']:,} runs failed.")
    doc.p("The audit also showed that every contract with a locked-ether, bad-randomness or double-spending finding is "
          "in the small configuration and none is in big. Training on big alone would have made the SmartBugs Bad "
          "Randomness category impossible to evaluate, so all contracts are used with our own split. One upstream "
          "mapping was corrected: unchecked-send (an ignored send() result, SWC-104) is moved from locked-ether to "
          "unchecked-calls. Full reasoning is in docs/label_decision.md.")
    fam = f.t["family_distribution"]
    if fam is not None:
        doc.table(fam[["family", "positives", "prevalence", "kept_as_class"]],
                  "Label families in the de-duplicated pool; rare-class rule: >= 1 % prevalence and >= 300 training positives",
                  nd=4)
    doc.h("3.3 Dataset B: SmartBugs Curated (external test only)", 2)
    sbc = f.t["smartbugs_counts"]
    tot = sbc[sbc.category == "TOTAL"].iloc[0]
    doc.p("SmartBugs Curated (github.com/smartbugs/smartbugs-curated, pinned at commit 230e649, Apache-2.0) groups "
          "vulnerable contracts by the DASP taxonomy and annotates each vulnerable line. Our own count is "
          f"{int(tot.contracts)} contracts, {int(tot.annotated_vulnerabilities)} annotated vulnerabilities and "
          f"{int(tot.annotated_lines)} distinct annotated lines. It is used only for testing. All comments - including "
          "the @vulnerable_at_lines header and the // <yes> <report> markers - are removed with the line-preserving "
          "cleaner before any model sees a contract, and the build fails if any marker survives.")
    doc.table(sbc, "SmartBugs Curated - our counts per category")
    lm = f.label_map["smartbugs_category_mapping"]
    doc.table(pd.DataFrame([[k, v["class"] or "-", v["status"], v["reason"]] for k, v in lm.items()],
                           columns=["SmartBugs category", "Our class", "Status", "Reason"]),
              "Class mapping between the datasets (only confirmed pairs are headline results)",
              widths=[3.6, 2.8, 2.3, 8.1], font=8)
    doc.h("3.4 Preprocessing", 2)
    doc.numbered([
        "Line-preserving comment removal: a state machine (normal, string, line comment, block comment) replaces comment characters by spaces and keeps every newline, so line numbers stay valid for highlighting and the SmartBugs annotations disappear without moving the annotated lines.",
        "Literal normalisation: strings become <STR>, numbers and hex literals <NUM>. User identifiers are kept for the model; replacing them by <ID> is tested as an ablation (E4) and used for clone detection.",
        "Tokenisation with a regular expression that keeps keywords, operators and identifiers and treats msg.sender, tx.origin, block.timestamp and similar globals as single tokens.",
        f"Long-file policy: at most {cfg['preprocessing']['max_tokens']:,} tokens; longer flattened files keep their first quarter and the end, where the main contract sits; the app warns when this happens.",
        "Exact and near-duplicate grouping and a group-aware split (section 3.5).",
        f"TF-IDF over token 1-3-grams with a vocabulary of {f.feature_stats.get('vocabulary_size', 0):,} n-grams chosen on a seeded subset of training contracts; IDF and SVD fitted on the training split only.",
    ])
    lt = f.t["length_stats"]
    if lt is not None:
        sl = lt[lt.dataset == "slither_dedup"].set_index("quantile")
        doc.p(f"{int(sl.loc[-1, 'tokens']):,} contracts ({pct(sl.loc[-1, 'lines'])}) exceed the token budget and are "
              "truncated.")
    doc.figure(FIG / "length_histogram.png", "Contract length in tokens: training contracts are far longer than SmartBugs", 15)
    doc.h("3.5 De-duplication and group-aware splits", 2)
    doc.p(f"Etherscan contains many copies of the same token and library code. Hashing the normalised token string "
          f"(identifiers anonymised) removed {d['exact_duplicates']:,} exact or renamed duplicates "
          f"({d['exact_duplicate_pct']:.1f} %). MinHash signatures of token 5-gram shingles with locality-sensitive "
          f"hashing found {d['near_duplicate_pairs']:,} near-duplicate pairs (estimated Jaccard similarity >= 0.8) among "
          f"the remaining contracts; their connected components form {d['groups_after_exact_dedup']:,} clone groups "
          f"(largest: {d['largest_group_after_exact_dedup']:,} contracts). Whole groups were assigned to train / "
          f"validation / test ({f.sizes['train']:,} / {f.sizes['val']:,} / {f.sizes['test']:,}) with greedy "
          "multi-label stratification so that every class has positives in every split. Under a random split of the "
          f"same sizes, {pct(d['random_split_leakage_rate'])} of validation and test contracts would have a clone in "
          f"the training data; under our split the figure is {pct(d['group_split_leakage_rate'])}.")
    doc.p(f"A by-product is a direct measurement of label noise: {d['exact_groups_with_label_conflicts']:,} groups of "
          "identical (normalised) code carry different Slither labels.")
    doc.figure(FIG / "duplicate_groups.png", "Contracts by clone-group size after exact de-duplication", 14)
    doc.figure(FIG / "label_distribution.png", "Class prevalence per split (cluster-aware split)", 15)
    doc.figure(FIG / "label_cooccurrence.png", "Label co-occurrence: probability of the column class given the row class", 11)
    doc.p("Weakness families often occur together: most reentrancy contracts also carry an unchecked-calls finding "
          "(Figure 5). This is why the task is multi-label, with one independent probability per class, rather than "
          "picking a single class per contract.")
    doc.figure(FIG / "solidity_versions.png", "Compiler generation: training pool vs SmartBugs (domain shift)", 14)
    doc.h("3.6 Data limitations", 2)
    doc.bullets([
        ("Weak labels: ", "targets are Slither findings, not expert judgements; in-distribution scores measure agreement with Slither."),
        ("Clone leakage: ", "handled by grouping and measured explicitly (E2)."),
        ("Domain shift: ", "SmartBugs contracts are small and almost all Solidity 0.4; training contracts are long, real deployments."),
        ("No clean SmartBugs contracts: ", "only recall can be measured there; false-positive rates come from the test split."),
        ("Unmapped categories: ", "Arithmetic (overflow), DoS, Front Running, Time Manipulation and Short Addresses have no training label."),
    ])
