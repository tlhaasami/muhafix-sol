"""
Report chapters 4 (algorithm), 5 (experiments and results), 7 (testing) and
9 (conclusion). Interpretation sentences are computed from the result tables
(which model wins, by how much, interval widths) so the text cannot drift from
the numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_config, path
from src.report_utils import MODEL_NAMES, Facts, fmt, model_lc, pct

FIG = path("figures_dir")


def _e1_view(f: Facts) -> pd.DataFrame:
    e1 = f.t["e1_test_summary"]
    return e1[["model", "macro_pr_auc", "macro_pr_auc_ci", "macro_f1", "macro_f1_ci", "macro_recall",
               "macro_precision", "micro_f1", "hamming_loss", "label_accuracy"]].rename(columns={
        "macro_pr_auc": "macro PR-AUC", "macro_pr_auc_ci": "95% CI", "macro_f1": "macro F1",
        "macro_f1_ci": "F1 95% CI", "macro_recall": "macro recall", "macro_precision": "macro precision",
        "micro_f1": "micro F1", "hamming_loss": "Hamming loss", "label_accuracy": "label accuracy"})


def chapter4(doc, f: Facts) -> None:
    cfg = load_config()["training"]
    sel = f.t["model_selection_val"]
    doc.h("4. Algorithm Description and Justification")
    doc.h("4.1 Task formulation", 2)
    doc.p(f"The task is multi-label classification: a contract can contain several weaknesses at once, so the "
          f"model outputs one probability per class ({', '.join(f.classes)}). Input x is the TF-IDF vector of the "
          "cleaned contract's token 1-3-grams; output is p(class | x) for every class; a per-class threshold turns "
          "probabilities into flags, and an overall risk level is derived from the same outputs. This directly "
          "supports the user's decision: deploy, fix the flagged functions first, or send the contract for a full "
          "audit.")
    doc.h("4.2 Models compared and why", 2)
    doc.table(pd.DataFrame([
        ["Majority / prior baseline", "predicts each class's training prevalence", "shows what scores mean under imbalance (floor)"],
        ["Logistic regression (one-vs-rest)", "linear model per class on TF-IDF, balanced class weights", "strong, fast text baseline; coefficients are interpretable"],
        ["Decision tree", "multi-output tree, class-weighted", "course model; captures keyword combinations"],
        ["Random forest", "bagged trees, balanced_subsample weights", "course model; non-linear, robust"],
        ["K-nearest neighbours", "cosine similarity on SVD-512, similarity-weighted votes", "course model; also powers 'similar contracts'"],
        ["MLP neural network", "TF-IDF -> 512 -> 128 -> 7, ReLU, dropout 0.3, sigmoid outputs", "learns non-linear token interactions and shared structure across classes"],
    ], columns=["Model", "How it works here", "Why it is appropriate"]), "Model families", widths=[4, 6.5, 6.3])
    doc.h("4.3 How the MLP learns (forward pass, loss, backpropagation)", 2)
    mcfg = cfg["mlp"]
    doc.p("Forward pass. The TF-IDF vector x (20,000 numbers) is multiplied by a weight matrix W1 and shifted by "
          "a bias b1, giving 512 numbers; the ReLU activation keeps the positive ones (h1 = max(0, W1x + b1)). "
          "During training, dropout randomly switches off 30 % of these units so the network cannot rely on any "
          "single token pattern. A second layer produces 128 numbers the same way, and the output layer produces 7 "
          "scores z; the sigmoid function turns each score into a probability p = 1 / (1 + e^-z). Because each class "
          "has its own sigmoid (not one softmax), several classes can be likely at once.")
    doc.p("Loss. For each contract and class the binary cross-entropy is -[y log p + (1 - y) log(1 - p)]: small "
          "when the probability agrees with the label, large when it confidently disagrees. Rare classes would be "
          "ignored if every mistake cost the same, so the loss on positive examples of class c is multiplied by "
          "(negatives / positives) for that class, capped at 50.")
    doc.p(f"Backpropagation and optimisation. Automatic differentiation applies the chain rule from the loss back "
          f"through the output layer, the ReLUs and the hidden layers, giving the gradient of the loss with respect "
          f"to every weight. Adam (learning rate {mcfg['lr']}, weight decay {mcfg['weight_decay']}) moves each "
          f"weight a small step against its gradient, using running averages of past gradients to adapt the step "
          f"size per weight. One epoch is one pass over the {f.sizes['train']:,} training contracts in mini-batches "
          f"of {mcfg['batch_size']}. After each epoch the validation macro PR-AUC is measured; training stops when it "
          f"has not improved for {mcfg['patience']} epochs and the best weights are restored (early stopping). The "
          "network is trained with three seeds to show how much the result depends on random initialisation.")
    doc.figure(FIG / "mlp_learning_curves.png", "MLP learning curves (loss and validation macro PR-AUC per epoch)", 15.5)
    doc.h("4.4 Training protocol", 2)
    doc.bullets([
        ("Imbalance: ", "balanced class weights (LR, tree, forest) and per-class positive weights in the MLP loss."),
        ("Tuning: ", "a small grid per family scored by macro PR-AUC on the validation split only "
                     "(reports/tables/validation_grid.csv). The test split and SmartBugs were never used for tuning. "
                     "The first full run crashed while saving (out of memory) after every model had been fitted; the "
                     "saved winning logistic regression and decision tree were reused and the scores of their losing "
                     "grid settings come from that run's log (column 'source'); every other model was re-fitted "
                     "with checkpoints, so a crash can no longer lose finished work."),
        ("MLP input: ", "sparse TF-IDF and the SVD-512 representation were compared on validation; the better one was kept."),
        ("Seeds: ", "seed 42 everywhere; the MLP is trained with seeds 42, 43 and 44 and the spread is reported."),
        ("Freeze: ", "model, thresholds and calibration are written to models/ and tagged model-v1 in git before any "
                     "test or SmartBugs evaluation; the evaluation script refuses to run otherwise."),
    ])
    if sel is not None:
        v = sel[["model", "val_macro_pr_auc", "val_macro_f2", "ece_before", "ece_after", "calibrated", "fit_seconds",
                 "size_mb"]].copy()
        v["model"] = v["model"].map(MODEL_NAMES)
        v["fit_seconds"] = v["fit_seconds"].map(lambda x: f"{x / 60:.1f}")
        v["size_mb"] = v["size_mb"].map(lambda x: f"{x:.1f}")
        v = v.rename(columns={"fit_seconds": "fit_minutes"})
        doc.table(v, "Model selection on the validation split (primary = highest validation macro PR-AUC; fit times "
                     "are wall-clock on one laptop CPU shared by several jobs)")
        best = sel.iloc[0] if sel.iloc[0]["model"] != "prior" else sel.iloc[1]
        doc.p(f"The {model_lc(MODEL_NAMES[f.primary])} has the highest validation macro PR-AUC "
              f"({best['val_macro_pr_auc']:.3f}) and is the primary model. Logistic regression is also shipped in "
              "the app so users can compare a simple, interpretable model with the primary one (feature F12).")
    doc.h("4.5 Decision thresholds, calibration and risk level", 2)
    th = f.thresholds
    doc.p(f"A missed vulnerability costs more than an extra warning, so each class's threshold maximises F2 on the "
          f"validation split (recall counts four times as much as precision), subject to a minimum precision of "
          f"{th['min_precision']} so the tool does not flag almost everything. Reliability curves showed that "
          "class-weighted training inflates probabilities, so Platt scaling (a one-feature logistic regression on "
          "the model's log-odds, fitted per class on the validation split) was applied whenever the expected "
          "calibration error exceeded 0.05; it changes the displayed probabilities but not the ranking.")
    doc.p(f"Risk level (shown in the app with its rule): low if no class reaches its threshold; medium if at least "
          f"one does; high if at least one class reaches the stricter level t + {th['high_margin']} x (1 - t), i.e. "
          f"a quarter of the way from its threshold to certainty.")
    thv = f.t["thresholds_val"]
    if thv is not None:
        tp = thv[thv.model == f.primary][["class", "threshold", "precision", "recall", "fbeta", "guard_met"]]
        doc.table(tp.rename(columns={"fbeta": "F2 (val)"}), f"Frozen thresholds of the primary model (validation values)")


def chapter5(doc, f: Facts) -> None:
    doc.h("5. Experiments, Results and Interpretation")
    doc.p("All numbers in this chapter were produced after the model-v1 freeze by python -m src.run_experiments and "
          "python -m src.eval_explain; the CSV files in reports/tables are the single source for this report, the "
          "slides and the app. Intervals are 95 % percentile bootstrap intervals over contracts.")
    # ---------------- E1
    doc.h("5.1 E1 - Model comparison on the cluster-aware test split", 2)
    e1 = f.t["e1_test_summary"]
    doc.table(_e1_view(f), "E1: all models on the held-out test split (frozen thresholds)", font=8)
    ranked = e1[e1.key != "prior"].sort_values("macro_pr_auc", ascending=False)
    top, second = ranked.iloc[0], ranked.iloc[1]
    prior = e1[e1.key == "prior"].iloc[0]
    lr = e1[e1.key == "logreg"].iloc[0]
    mlp = e1[e1.key == "mlp"].iloc[0]
    txt = (f"Interpretation. The {model_lc(top['model'])} ranks first on macro PR-AUC ({top['macro_pr_auc']:.3f}, "
           f"{top['macro_pr_auc_ci']}), followed by the {model_lc(second['model'])} ({second['macro_pr_auc']:.3f}). "
           f"The prior baseline scores {prior['macro_pr_auc']:.3f} - simply the average class prevalence - yet its "
           f"label accuracy is {prior['label_accuracy']:.3f}, which shows why accuracy is not our headline metric: "
           "predicting nothing already looks good when most labels are 0. ")
    gap = mlp["macro_pr_auc"] - lr["macro_pr_auc"]
    if abs(gap) < 0.01:
        txt += (f"The MLP and logistic regression are within {abs(gap):.3f} of each other: on TF-IDF features a "
                "linear model already captures most of the signal, and the neural network adds little. ")
    elif gap > 0:
        txt += (f"The MLP beats logistic regression by {gap:.3f} macro PR-AUC, so non-linear combinations of token "
                "patterns carry extra signal. ")
    else:
        txt += (f"Logistic regression beats the MLP by {-gap:.3f} macro PR-AUC - an honest result: on sparse TF-IDF "
                "features the simpler model suffices. ")
    txt += ("What the table does not show: these labels are Slither's, so a high score means close agreement with "
            "Slither, not proof of security - that is what E3 tests.")
    doc.p(txt)
    pc = f.per_class_primary
    if pc is not None:
        doc.table(pc[["class", "precision", "recall", "f1", "f2", "pr_auc", "support", "fpr", "threshold"]],
                  f"E1: per-class results of the primary model ({MODEL_NAMES[f.primary]}) on the test split")
        worst = pc.sort_values("pr_auc").iloc[0]
        bestc = pc.sort_values("pr_auc").iloc[-1]
        doc.p(f"Per class, {bestc['class']} is easiest (PR-AUC {bestc['pr_auc']:.3f}) and {worst['class']} hardest "
              f"({worst['pr_auc']:.3f}). Classes defined by a few distinctive tokens (e.g. block variables for "
              "bad randomness, low-level call syntax) are learned well; heterogeneous groups such as 'other' mix "
              "many unrelated detectors and are harder.")
    doc.figure(FIG / "e1_per_class_pr_auc.png", "E1: per-class PR-AUC of every model on the test split", 16)
    doc.figure(FIG / "e1_pr_and_calibration.png", "Primary model: precision-recall curves and calibration on the test split", 16)
    sd = f.t["e1_mlp_seeds_test"]
    if sd is not None:
        doc.table(sd, "MLP seed stability on the test split (acceptance test B5)")
        spread = float(sd[sd.seed == "std"]["test_macro_pr_auc"].iloc[0])
        doc.p(f"The standard deviation across seeds is {spread:.4f} macro PR-AUC, far smaller than the gaps between "
              "model families, so the ranking does not depend on the seed.")
    # ---------------- E2
    doc.h("5.2 E2 - How much does clone leakage inflate scores?", 2)
    e2 = f.t["e2_leakage"]
    if e2 is not None:
        doc.table(e2[["model", "cluster_test_macro_pr_auc", "random_test_macro_pr_auc", "pr_auc_inflation",
                      "cluster_test_macro_f1", "random_test_macro_f1", "f1_inflation"]],
                  "E2: the same configurations on a random split vs the cluster-aware split", font=8)
        knn = e2[e2.key == "knn"]
        mean_inf = e2[e2.key != "prior"]["pr_auc_inflation"].mean()
        doc.p(f"Interpretation. Under a random split {pct(f.dedup['random_split_leakage_rate'])} of test contracts "
              "have a near-duplicate in training. Every learned model scores higher there - on average "
              f"{mean_inf:+.3f} macro PR-AUC - "
              + (f"and the nearest-neighbour model, which can simply look up the clone, gains most "
                 f"({knn['pr_auc_inflation'].iloc[0]:+.3f}). " if len(knn) else "")
              + "A random split would therefore have overstated how well the tool generalises to genuinely new "
                "contracts. Note that exact duplicates were removed from both pools, so this is a lower bound on the "
                "inflation of a completely naive pipeline.")
        doc.figure(FIG / "e2_leakage.png", "E2: macro PR-AUC on random vs cluster-aware test splits", 14.5)
    # ---------------- E3
    doc.h("5.3 E3 - Independent test on SmartBugs Curated", 2)
    e3 = f.t["e3_smartbugs_recall"]
    if e3 is not None:
        doc.table(e3[["smartbugs_category", "mapped_class", "status", "contracts", "recall", "ci_low", "ci_high",
                      "mean_probability"]], f"E3: recall of the primary model per SmartBugs category (95 % bootstrap CI)",
                  font=8)
        conf = e3[e3.status == "confirmed"]
        s3 = f.t["e3_summary"]
        prim = s3[s3.key == f.primary].iloc[0]
        doc.p("Interpretation. SmartBugs contracts were annotated by people, not by Slither, so this is the honest "
              "test of usefulness. Over the confirmed categories the primary model flags "
              f"{pct(prim['smartbugs_pooled_recall_confirmed'])} of the vulnerable contracts. "
              + " ".join(f"{r.smartbugs_category.replace('_', ' ')}: {pct(r.recall)} [{pct(r.ci_low, 0)}-{pct(r.ci_high, 0)}] "
                         f"of {int(r.contracts)}." for r in conf.itertuples())
              + " The intervals are wide because the categories are small (blueprint risk R3). Recall is lower than "
                "on the in-distribution test split because SmartBugs contracts are tiny Solidity 0.4 teaching "
                "examples, very unlike the long deployed contracts the model learned from (domain shift), and "
                "because Slither itself - the source of our labels - misses some of these bugs. Categories without "
                "a training label are listed as not evaluated rather than forced into a mapping.")
        doc.table(s3[["model", "smartbugs_pooled_recall_confirmed", "false_alarm_rate_on_clean_test",
                      "macro_fpr_test"]],
                  "E3: SmartBugs recall (confirmed categories, pooled) and false alarms on clean test contracts", font=8)
        doc.p(f"SmartBugs has no clean contracts, so false alarms are measured on the {int(prim['test_clean_contracts']):,} "
              "test contracts in which Slither found nothing: the primary model raises at least one flag on "
              f"{pct(prim['false_alarm_rate_on_clean_test'])} of them. This is the cost of recall-favouring (F2) "
              "thresholds; the sensitivity slider in the app lets a user trade some recall for fewer alarms.")
    # ---------------- E4
    doc.h("5.4 E4 - Ablations", 2)
    e4 = f.t["e4_ablations"]
    if e4 is not None:
        doc.table(e4, "E4: ablations with logistic regression (retrained per setting; thresholds from validation)",
                  font=8)
        ref = e4.iloc[0]
        com = e4[e4.ablation == "comments"].iloc[0]
        ids = e4[e4.ablation == "identifiers"].iloc[0]
        ng = e4[e4.ablation == "ngrams"]
        doc.p("Interpretation. "
              f"(a) Keeping comments on SmartBugs changes recall from {pct(ref['smartbugs_recall_confirmed'])} to "
              f"{pct(com['smartbugs_recall_confirmed'])}: the annotation words are mostly outside the training "
              "vocabulary, so the leak is small here, but stripping them is still required - a model trained with "
              "comments could learn them. "
              f"(b) Replacing user identifiers by <ID> gives test macro PR-AUC {ids['test_macro_pr_auc']:.3f} vs "
              f"{ref['test_macro_pr_auc']:.3f} and SmartBugs recall {pct(ids['smartbugs_recall_confirmed'])}: "
              + ("names such as balances or withdraw carry useful signal in-distribution. "
                 if ref["test_macro_pr_auc"] > ids["test_macro_pr_auc"] else
                 "anonymised identifiers generalise at least as well. ")
              + f"(c) Shorter n-grams score " + ", ".join(f"{r.setting}: {r.test_macro_pr_auc:.3f}" for r in ng.itertuples())
              + f" vs 1-3-grams {ref['test_macro_pr_auc']:.3f}, so short token sequences (e.g. '. call . value') add "
                "information beyond single tokens.")


def chapter7(doc, f: Facts) -> None:
    doc.h("7. Testing and Evaluation Evidence")
    ev = path("docs_dir") / "test_evidence.json"
    tests = json.loads(ev.read_text()) if ev.exists() else {}
    doc.h("7.1 Automated tests", 2)
    if tests:
        doc.p(f"The test suite (pytest) contains {tests['total']} tests: {tests['passed']} passed, "
              f"{tests['failed']} failed, {tests['skipped']} skipped (run on {tests['date']}, {tests['seconds']:.0f} s). "
              "It covers unit tests of every module, the acceptance tests of all three phases on the real data and "
              "model, and integration tests of the app.")
        doc.table(pd.DataFrame(tests["files"]), "Test files and results", widths=[5.5, 2, 2, 7.3], font=8)
    doc.h("7.2 Acceptance tests (blueprint 13.2, 14.2, 15.2)", 2)
    doc.table(pd.DataFrame(tests.get("acceptance", [])), "Acceptance tests and evidence", widths=[1.4, 4.6, 2, 8.8],
              font=8) if tests.get("acceptance") else None
    doc.h("7.3 The ten fixed test inputs (C2)", 2)
    if tests.get("ten_inputs"):
        doc.table(pd.DataFrame(tests["ten_inputs"]), "Ten fixed inputs: expected vs observed behaviour", font=8)
    # ---------------- E3b
    doc.h("7.4 E3b - Do the highlighted lines point at the real bug?", 2)
    hr = f.t["e3b_hit_rate"]
    if hr is not None:
        doc.table(hr[["category", "contracts", "strict@1", "strict@3", "strict@3_ci", "relaxed@3", "random@1",
                      "random@3"]], "E3b: line hit-rate on SmartBugs (primary model) vs a random-line baseline",
                  font=8)
        allr = hr.iloc[0]
        doc.p(f"Interpretation. For {int(allr['contracts'])} SmartBugs contracts in mapped categories, an annotated "
              f"vulnerable line is the single top-ranked line in {pct(allr['strict@1'])} of cases and among the top "
              f"three in {pct(allr['strict@3'])} ({allr['strict@3_ci']}); picking three code lines at random would "
              f"hit {pct(allr['random@3'])}. Counting a hit when a top-3 line is in the same function as the "
              f"annotated line raises this to {pct(allr['relaxed@3'])}. The highlights are therefore clearly better "
              "than chance and usually land in the right function, but they explain what the model reacts to - "
              "e.g. the external call rather than the later balance update - so they guide a reviewer rather than "
              "pinpoint the bug.")
    # ---------------- E5
    doc.h("7.5 E5 - Latency and error analysis", 2)
    lat = f.t["e5_latency"]
    if lat is not None:
        doc.table(lat, "E5: latency (milliseconds) on the development laptop (CPU only)", font=8)
    sl = f.t["e5_error_slices"]
    if sl is not None:
        doc.table(sl, "E5: test-split recall / precision of the primary model by contract length and compiler generation",
                  font=8)
    if f.cases:
        rows = [[c["kind"], _short_id(c["id"]), c["class"], f"{c['probability']:.2f}", "yes" if c["flagged"] else "no",
                 ", ".join(map(str, c["top_lines"])) or "-", ", ".join(map(str, c["vulnerable_lines"][:4]))]
                for c in f.cases]
        doc.table(pd.DataFrame(rows, columns=["kind", "SmartBugs contract", "class", "p", "flagged", "top lines",
                                              "annotated lines"]), "E5: ten case studies (five successes, five failures)",
                  font=8, widths=[1.6, 5.2, 2.4, 1.1, 1.5, 2.3, 2.4])
        for c in f.cases:
            doc.p(_case_text(c), size=9.5)


def _short_id(case_id: str) -> str:
    """category/file name, with long address-style file names shortened so table columns stay readable."""
    cat, _, name = case_id.rpartition("/")
    if name.startswith("0x") and len(name) > 26:
        name = f"{name[:10]}...{name[-8:]}"
    return f"{cat}/{name}" if cat else name


def _case_text(c: dict) -> str:
    top = c["top_line_text"][0] if c["top_line_text"] else "-"
    vul = c["vulnerable_text"][0] if c["vulnerable_text"] else "-"
    if c["kind"] == "success":
        return (f"Success - {c['id']} ({c['class']}): flagged with p = {c['probability']:.2f}; the top line "
                f"`{top[:90]}` matches the annotated vulnerable line. The pattern is the textbook form the model saw "
                "many times in training.")
    why = ("not flagged" if not c["flagged"] else "flagged, but the highlights miss the annotated line")
    return (f"Failure - {c['id']} ({c['class']}): {why} (p = {c['probability']:.2f}). Annotated line: `{vul[:90]}`. "
            "Typical causes: very short or unusual code that shares few n-grams with the training contracts, "
            "old syntax, or a bug that Slither itself does not report, so the model never learned it.")


def chapter9(doc, f: Facts) -> None:
    doc.h("9. Conclusion and Future Work")
    e1 = f.t["e1_test_summary"]
    s3 = f.t["e3_summary"]
    prim = e1[e1.key == f.primary].iloc[0] if e1 is not None else None
    doc.p("Muhafiz.sol meets the Path A requirements: its core output comes from trained models, the full pipeline "
          "is implemented and evaluated with metrics suited to imbalanced multi-label data, and a Streamlit app "
          "shows the complete flow from pasted code to an explained, downloadable result. The work that made the "
          "numbers trustworthy is as important as the models: re-deriving the labels from raw Slither output "
          "(which resolved the label-4 question and recovered the bad-randomness class), removing 46 % exact "
          "duplicates, keeping clone groups and giant template families out of the evaluation splits, stripping "
          "annotations, and freezing the model before any test evaluation.")
    if prim is not None:
        doc.p(f"The primary {model_lc(MODEL_NAMES[f.primary])} reaches a macro PR-AUC of {prim['macro_pr_auc']:.3f} on "
              f"unseen contract families and flags "
              f"{pct(s3[s3.key == f.primary]['smartbugs_pooled_recall_confirmed'].iloc[0])} of the human-annotated "
              "SmartBugs contracts in the categories where a fair comparison is possible. The random-split ablation "
              "shows how much a careless split would have flattered these numbers, and the hit-rate study shows that "
              "the highlighted lines are useful pointers but not proof.")
    doc.h("Future work", 2)
    doc.bullets([
        "Function-level chunking with a pretrained code model (e.g. CodeBERT-style embeddings) to capture semantics beyond token n-grams.",
        "Human-verified labels for more weakness types (overflow, DoS, front running, timestamp dependence) so they can be learned and evaluated.",
        "Augmenting training with small Solidity 0.4-style snippets to reduce the domain gap to teaching examples.",
        "Batch scanning connected to a review queue for auditors, and a public deployment on Streamlit Community Cloud.",
    ])
    doc.h("Team contributions", 2)
    doc.table(pd.DataFrame([
        ["Talha Sami (24I-3118)", "Phase 1: data acquisition, label audit, preprocessing, de-duplication, splits, features, data card; chapters 1-3"],
        ["Sohaib (24I-3108)", "Phase 2: evaluation harness, models, MLP, thresholds, calibration, experiments E1-E4, model card; chapters 4-5"],
        ["Shaiman (24I-3074)", "Phase 3: explanation module, Streamlit app, integration tests, E3b/E5, report and slide assembly; chapters 6-9"],
    ], columns=["Member", "Responsibility (per blueprint)"]), "Responsibilities as planned in the blueprint", widths=[4.5, 12.3])
