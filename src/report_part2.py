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
        ["K-nearest neighbours", "cosine similarity on SVD-512, similarity-weighted votes", "course model; the app's 'similar contracts' uses a separate 256-dimension index with the same idea"],
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
        ("MLP input: ", "sparse TF-IDF and the SVD-512 representation were compared on validation with seed 42 "
                        f"(macro PR-AUC {_grid(f, 'mlp', 'svd')} for SVD vs {_grid(f, 'mlp', 'tfidf')} for TF-IDF); "
                        "TF-IDF was kept."),
        ("Seeds: ", "seed 42 everywhere; the MLP is trained with seeds 42, 43 and 44 and the spread is reported. The "
                    "primary model is the seed with the best validation score, and an MLP's validation score is the "
                    "best epoch (early stopping), so its validation number is slightly optimistic compared with the "
                    "single fits of the other families; the three-seed means are given next to it."),
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
        seeds = f.thresholds.get("mlp_seed_val_pr_auc", {})
        doc.p(f"The {model_lc(MODEL_NAMES[f.primary])} has the highest validation macro PR-AUC "
              f"({best['val_macro_pr_auc']:.3f}; mean over the three seeds "
              f"{seeds.get('mean', float('nan')):.3f} +/- {seeds.get('std', float('nan')):.3f}) and is the primary "
              "model; it stays first on the three-seed mean, so the choice does not rest on a lucky seed. Logistic "
              "regression is also shipped in the app so users can compare a simple, interpretable model with the "
              "primary one (feature F12). The MLP weights are stored as 16-bit floats after the thresholds were "
              "chosen; this changes probabilities by at most 0.0003 and flips 4 of about 68,000 validation "
              "decisions.")
    doc.h("4.5 Decision thresholds, calibration and risk level", 2)
    th = f.thresholds
    doc.p(f"A missed vulnerability costs more than an extra warning, so each class's threshold maximises F2 on the "
          f"validation split (recall counts four times as much as precision), subject to a minimum precision of "
          f"{th['min_precision']} so the tool does not flag almost everything. Reliability curves showed that "
          "class-weighted training inflates probabilities, so Platt scaling (a one-feature logistic regression on "
          "the model's log-odds, fitted per class on the validation split) was applied whenever a model's expected "
          "calibration error, averaged over the classes, exceeded 0.05; it changes the displayed probabilities but "
          "not the ranking. The averaged ECE of the MLP was 0.047, just below the gate, so its probabilities are not "
          "rescaled. That was a weak rule: per class, several classes are above 0.05, and on the test split the "
          + _calibration_note(f) + " A per-class gate (or always calibrating) would have been better; it cannot be "
          "changed after the freeze, so the app's percentages for the MLP should be read as scores, not exact "
          "probabilities.")
    doc.p(f"Risk level (shown in the app with its rule): low if no class reaches its threshold; medium if at least "
          f"one does; high if at least one class reaches the stricter level t + {th['high_margin']} x (1 - t), i.e. "
          f"a quarter of the way from its threshold to certainty.")
    thv = f.t["thresholds_val"]
    if thv is not None:
        tp = thv[thv.model == f.primary][["class", "threshold", "precision", "recall", "fbeta", "guard_met"]]
        doc.table(tp.rename(columns={"fbeta": "F2 (val)"}), f"Frozen thresholds of the primary model (validation values)")


def _grid(f: Facts, family: str, mlp_input: str, seed: int = 42) -> str:
    g = f.t["validation_grid"] if "validation_grid" in f.t and f.t["validation_grid"] is not None else None
    if g is None:
        return "-"
    rows = g[(g.family == family) & g.params.str.contains(f'"input": "{mlp_input}"') &
             g.params.str.contains(f'"seed": {seed}')]
    return f"{rows['val_macro_pr_auc'].iloc[0]:.3f}" if len(rows) else "-"


def _calibration_note(f: Facts) -> str:
    cal = f.t["e1_calibration_primary"]
    if cal is None:
        return "primary model is somewhat over-confident above p = 0.1."
    rows = []
    for cls, g in cal.groupby("class"):
        hi = g[g["mean_predicted"] >= 0.3]
        if len(hi) and hi["count"].sum() > 0:
            gap = float(((hi["mean_predicted"] - hi["observed"]) * hi["count"]).sum() / hi["count"].sum())
            rows.append((gap, cls))
    rows.sort(reverse=True)
    worst = ", ".join(f"{c} by {g:.2f}" for g, c in rows[:2])
    return (f"primary model is over-confident where it predicts 0.3 or more (largest gaps: {worst}; "
            "reports/tables/e1_calibration_primary.csv).")


def chapter5(doc, f: Facts) -> None:
    doc.h("5. Experiments, Results and Interpretation")
    doc.p("All numbers in this chapter were produced after the model-v1 freeze by python -m src.run_experiments and "
          "python -m src.eval_explain; the CSV files in reports/tables are the single source for this report, the "
          "slides and the app. Test-split intervals are 95 % percentile cluster-bootstrap intervals: 1,000 "
          "resamples of whole clone groups, so near-duplicate contracts are not counted as independent evidence. "
          "SmartBugs intervals resample contracts.")
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
           f"The prior baseline flags every class on every contract, so its macro PR-AUC ({prior['macro_pr_auc']:.3f}) "
           "is simply the average class prevalence and its label accuracy is equally low. The opposite trivial rule, "
           f"never flagging anything, would reach a label accuracy of {1 - prior['label_accuracy']:.3f} while finding "
           "no weakness at all (recall 0). That is why accuracy is not our headline metric: on imbalanced labels "
           "it rewards silence. ")
    gap = mlp["macro_pr_auc"] - lr["macro_pr_auc"]
    if abs(gap) < 0.01:
        txt += (f"The MLP and logistic regression are within {abs(gap):.3f} of each other: on TF-IDF features a "
                "linear model already captures most of the signal, and the neural network adds little. ")
    elif gap > 0:
        txt += (f"The MLP beats logistic regression by {gap:.3f} macro PR-AUC, so non-linear combinations of token "
                "patterns carry extra signal. ")
        lo_m = float(str(mlp["macro_pr_auc_ci"]).strip("[]").split(",")[0])
        hi_l = float(str(lr["macro_pr_auc_ci"]).strip("[]").split(",")[1])
        if lo_m <= hi_l:
            txt += ("Their cluster-bootstrap intervals overlap, so this lead is consistent with the validation "
                    "ranking but not proven by the intervals alone. ")
    else:
        txt += (f"Logistic regression beats the MLP by {-gap:.3f} macro PR-AUC - an honest result: on sparse TF-IDF "
                "features the simpler model suffices. ")
    sd_ = f.t["e1_mlp_seeds_test"]
    if sd_ is not None and (sd_["seed"].astype(str) == "mean").any():
        txt += (f"The MLP row is the seed chosen on validation; the mean over the three seeds on test is "
                f"{float(sd_[sd_.seed.astype(str) == 'mean']['test_macro_pr_auc'].iloc[0]):.3f} (Table 15). ")
    txt += ("What the table does not show: these labels are Slither's, so a high score means close agreement with "
            "Slither, not proof of security - that is what E3 tests.")
    doc.p(txt)
    pc = f.per_class_primary
    if pc is not None:
        pc = pc.assign(threshold=pc["class"].map(f.frozen_thresholds()))
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
        learned = e2[e2.key != "prior"].sort_values("pr_auc_inflation", ascending=False)
        mean_inf = learned["pr_auc_inflation"].mean()
        first, second = learned.iloc[0], learned.iloc[1]
        doc.p(f"Interpretation. Under a random split {pct(f.dedup['random_split_leakage_rate'])} of validation and "
              "test contracts have a near-duplicate in training. Every learned model scores higher there, on average "
              f"{mean_inf:+.3f} macro PR-AUC. The {model_lc(first['model'])} gains most "
              f"({first['pr_auc_inflation']:+.3f}), followed by the {model_lc(second['model'])} "
              f"({second['pr_auc_inflation']:+.3f}): tree models that memorise specific token combinations profit "
              "most from having seen copies. A random split would therefore have overstated how well the tool "
              "generalises to genuinely new contracts. The comparison is not perfectly like-for-like, so the gap shows "
              "the size of the effect rather than an exact measurement: the random-split models reuse the TF-IDF "
              "vocabulary and IDF weights fitted on the cluster-aware training split, which contains many random-split "
              "test contracts, and the random test split also contains the six giant template families that the "
              "cluster-aware split keeps in training.")
        doc.figure(FIG / "e2_leakage.png", "E2: macro PR-AUC on random vs cluster-aware test splits", 14.5)
    # ---------------- E3
    doc.h("5.3 E3 - Independent test on SmartBugs Curated", 2)
    e3 = f.t["e3_smartbugs_recall"]
    if e3 is not None:
        ov = f.t["gold_overlap"]
        if ov is not None:
            n_ov, n_ex = int(ov["overlap"].sum()), int(ov["exact_match"].sum())
            by = ov[ov["overlap"]]["matched_split"].value_counts()
            doc.p(f"Overlap with the training data. Many SmartBugs files are real Etherscan contracts. A check added "
                  f"after the freeze (python -m src.gold_overlap, reports/tables/gold_overlap.csv) found that "
                  f"{n_ov} of the {len(ov)} SmartBugs contracts also occur in Dataset A, {n_ex} of them as exact "
                  "copies of the normalised code and the rest as near copies (Jaccard >= 0.8) or the same address; the "
                  "matching Dataset A contract is in "
                  + ", ".join(f"{k} for {v}" for k, v in by.items())
                  + ". These contracts are not independent of training, so the table reports recall on all contracts "
                    "and on the contracts with no overlap. The check changes no model or threshold; it only changes "
                    "how the numbers are read.")
        e3 = e3.assign(contracts_no_overlap=e3["contracts_no_overlap"].map(
            lambda v: "-" if pd.isna(v) else f"{int(v)}"))
        doc.table(e3[["smartbugs_category", "mapped_class", "status", "contracts", "recall", "ci_low", "ci_high",
                      "contracts_no_overlap", "recall_no_overlap"]].rename(columns={
                          "contracts_no_overlap": "contracts (no overlap)", "recall_no_overlap": "recall (no overlap)"}),
                  "E3: recall of the primary model per SmartBugs category (95 % bootstrap CI), all contracts and "
                  "contracts with no overlap with the training data", font=8)
        conf = e3[e3.status == "confirmed"]
        s3 = f.t["e3_summary"]
        prim = s3[s3.key == f.primary].iloc[0]
        clean_n = int(prim.get("smartbugs_confirmed_contracts_no_overlap", 0))
        doc.p("Interpretation. SmartBugs contracts were annotated by people, not by Slither, so this is the honest "
              f"test of usefulness. On the {clean_n} confirmed-category contracts with no copy in the training data "
              f"the primary model flags {pct(prim['smartbugs_pooled_recall_confirmed_no_overlap'])}; this is the "
              "independent headline number. Including the overlapping contracts it flags "
              f"{pct(prim['smartbugs_pooled_recall_confirmed'])} of all {int(conf['contracts'].sum())}. Per category "
              "(all contracts): "
              + " ".join(f"{r.smartbugs_category.replace('_', ' ')}: {pct(r.recall)} [{pct(r.ci_low, 0)}-{pct(r.ci_high, 0)}] "
                         f"of {int(r.contracts)}." for r in conf.itertuples())
              + " The intervals are wide because the categories are small (blueprint risk R3). Recall is lower than "
                "on the in-distribution test split. Two explanations are plausible but were not tested: domain shift "
                "(SmartBugs contracts are short, mostly Solidity 0.4 examples, unlike the long deployed contracts in "
                "training; old syntax alone does not explain it, since test-split Solidity 0.4 contracts still reach "
                "0.84 micro recall in E5), and label shift (Slither, the source of the training labels, may not report "
                "every SmartBugs bug; we did not run Slither on SmartBugs to check). Categories without a training "
                "label are listed as not evaluated rather than forced into a mapping.")
        doc.p("Comparison with published work. Studies on Slither-labelled data report much higher scores, for example "
              "93.53 % F1 for the CodeBERT model in Lightning Cat [11]; LLM-based auditors such as FTSmartAudit [12] "
              "go further. Those figures are measured against tool labels on their own splits and are not comparable "
              "with recall on a human-annotated benchmark after clone-aware splitting, which is what this chapter "
              "reports.")
        doc.table(s3[["model", "smartbugs_pooled_recall_confirmed", "smartbugs_pooled_recall_confirmed_no_overlap",
                      "false_alarm_rate_on_clean_test", "macro_fpr_test"]].rename(columns={
                          "smartbugs_pooled_recall_confirmed": "smartbugs recall (all)",
                          "smartbugs_pooled_recall_confirmed_no_overlap": "smartbugs recall (no overlap)"}),
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
              f"(b) Replacing user identifiers by <ID> raises test macro PR-AUC to {ids['test_macro_pr_auc']:.3f} "
              f"(reference {ref['test_macro_pr_auc']:.3f}) but lowers SmartBugs recall to "
              f"{pct(ids['smartbugs_recall_confirmed'])} (reference {pct(ref['smartbugs_recall_confirmed'])}), so it "
              "helps in-distribution and does not help on the human-annotated set. "
              "(c) The n-gram range is the clearest result, and it goes against our choice: "
              + "; ".join(f"{r.setting} gives test macro PR-AUC {r.test_macro_pr_auc:.3f} and SmartBugs recall "
                          f"{pct(r.smartbugs_recall_confirmed)}" for r in ng.itertuples())
              + f", against {ref['test_macro_pr_auc']:.3f} and {pct(ref['smartbugs_recall_confirmed'])} for the "
                "1-3-grams we froze. Shorter n-grams generalise better to the small SmartBugs contracts. These "
                "comparisons use the test set, so the frozen setting is not re-selected on them; a validation-based "
                "choice of the n-gram range is the first item of future work.")


    section_model_v2(doc, f)

def section_model_v2(doc, f: Facts) -> None:
    """5.5: model-v2, retrained without the SmartBugs overlap (only if its results exist)."""
    from src.config import ROOT, load_config

    vcfg = load_config().get("variants", {}).get("v2", {}).get("paths", {})
    t2 = ROOT / vcfg.get("tables_dir", "reports/tables/v2")
    b2 = ROOT / vcfg.get("processed_dir", "data/processed_v2") / "build_summary.json"
    need = ["e1_test_summary.csv", "e3_summary.csv", "e3b_hit_rate.csv", "model_selection_val.csv"]
    if not all((t2 / n).exists() for n in need) or not b2.exists():
        return
    rd = lambda n: pd.read_csv(t2 / n)
    e1a, e1b = f.t["e1_test_summary"].set_index("key"), rd("e1_test_summary.csv").set_index("key")
    s3a, s3b = f.t["e3_summary"].set_index("key"), rd("e3_summary.csv").set_index("key")
    hra, hrb = f.t["e3b_hit_rate"].set_index("category"), rd("e3b_hit_rate.csv").set_index("category")
    sva, svb = f.t["model_selection_val"].set_index("model"), rd("model_selection_val.csv").set_index("model")
    ex = json.loads(b2.read_text())["smartbugs_exclusion"]
    k = f.primary
    v2_primary = svb[svb["primary"].astype(str) == "True"].index[0] if "primary" in svb else k
    doc.h("5.5 model-v2 - retrained without the SmartBugs overlap", 2)
    doc.p(f"After the review found the overlap described in 5.3, a second model was trained to make the external "
          f"test fully independent. Every Dataset A clone group containing a copy or near copy of a SmartBugs "
          f"contract was removed ({ex['clone_groups_removed']} groups, {ex['contracts_removed']} contracts: "
          f"{ex['removed_by_split']['train']} train, {ex['removed_by_split']['val']} validation, "
          f"{ex['removed_by_split']['test']} test); all other contracts kept their split, the TF-IDF pipeline was "
          "refitted, and every model was retrained with exactly the hyper-parameters model-v1 had chosen (MLP on "
          "TF-IDF, seed 44), so nothing was tuned on test results. Thresholds were again chosen on validation and "
          "the model was frozen under git tag model-v2 before its own test and SmartBugs evaluation "
          "(python -m src.build_v2, then the usual scripts with MUHAFIZ_VARIANT=v2). model-v1 stays the "
          "reported model; model-v2 checks how much the overlap mattered.")

    def g(df, idx, col):
        try:
            return float(df.loc[idx, col])
        except (KeyError, ValueError):
            return float("nan")
    conf_a = "confirmed classes"
    rows = [
        ["Primary model", MODEL_NAMES[k], MODEL_NAMES.get(v2_primary, v2_primary)],
        ["Validation macro PR-AUC (primary)", f"{g(sva, k, 'val_macro_pr_auc'):.3f}", f"{g(svb, v2_primary, 'val_macro_pr_auc'):.3f}"],
        ["Test macro PR-AUC (MLP)", f"{g(e1a, 'mlp', 'macro_pr_auc'):.3f} {e1a.loc['mlp', 'macro_pr_auc_ci']}",
         f"{g(e1b, 'mlp', 'macro_pr_auc'):.3f} {e1b.loc['mlp', 'macro_pr_auc_ci']}"],
        ["Test macro F1 (MLP)", f"{g(e1a, 'mlp', 'macro_f1'):.3f}", f"{g(e1b, 'mlp', 'macro_f1'):.3f}"],
        ["Test macro PR-AUC (logistic regression)", f"{g(e1a, 'logreg', 'macro_pr_auc'):.3f}", f"{g(e1b, 'logreg', 'macro_pr_auc'):.3f}"],
        ["SmartBugs recall, confirmed classes, all contracts (MLP)",
         f"{pct(g(s3a, 'mlp', 'smartbugs_pooled_recall_confirmed'))} of 91",
         f"{pct(g(s3b, 'mlp', 'smartbugs_pooled_recall_confirmed'))} of 91"],
        ["SmartBugs recall, contracts with no copy in training (MLP)",
         f"{pct(g(s3a, 'mlp', 'smartbugs_pooled_recall_confirmed_no_overlap'))} of "
         f"{int(g(s3a, 'mlp', 'smartbugs_confirmed_contracts_no_overlap'))}",
         f"{pct(g(s3b, 'mlp', 'smartbugs_pooled_recall_confirmed_no_overlap'))} of "
         f"{int(g(s3b, 'mlp', 'smartbugs_confirmed_contracts_no_overlap'))}"],
        ["False alarms on clean test contracts (MLP)", pct(g(s3a, 'mlp', 'false_alarm_rate_on_clean_test')),
         pct(g(s3b, 'mlp', 'false_alarm_rate_on_clean_test'))],
        ["E3b: annotated line in top 3 (confirmed classes)", pct(g(hra, conf_a, 'strict@3')), pct(g(hrb, conf_a, 'strict@3'))],
    ]
    doc.table(pd.DataFrame(rows, columns=["", "model-v1 (reported)", "model-v2 (no overlap)"]),
              "model-v1 vs model-v2 (test split and SmartBugs)", widths=[7.5, 4.6, 4.7], font=8.5)
    d_test = g(e1b, "mlp", "macro_pr_auc") - g(e1a, "mlp", "macro_pr_auc")
    r_all_a, r_all_b = g(s3a, "mlp", "smartbugs_pooled_recall_confirmed"), g(s3b, "mlp", "smartbugs_pooled_recall_confirmed")
    r_clean_a = g(s3a, "mlp", "smartbugs_pooled_recall_confirmed_no_overlap")
    doc.p(f"Interpretation. Removing {ex['removed_by_split']['train']} of about 45,500 training contracts leaves the "
          f"in-distribution result essentially unchanged (test macro PR-AUC {d_test:+.3f}). On SmartBugs, where all "
          f"91 confirmed-category contracts are now independent of training, model-v2 flags {pct(r_all_b)}, against "
          f"{pct(r_all_a)} for model-v1 on the same contracts and {pct(r_clean_a)} for model-v1 on its 51 clean "
          "contracts. "
          + ("The drop from model-v1's all-contract figure confirms that part of it came from recognising contracts "
             "seen in training; the independent figure is the one to quote. " if r_all_b < r_all_a - 0.01 else
             "The recall barely moves, so the overlap did not inflate model-v1's SmartBugs result much. ")
          + "The intervals on 91 contracts are narrower than on 51, which is the main gain of model-v2.")


def chapter7(doc, f: Facts) -> None:
    doc.h("7. Testing and Evaluation Evidence")
    ev = path("docs_dir") / "test_evidence.json"
    tests = json.loads(ev.read_text()) if ev.exists() else {}
    doc.h("7.1 Automated tests", 2)
    if tests:
        slow = ("with RUN_SLOW=1, which also runs the slow re-training and re-clustering tests; a plain `pytest` "
                "skips those" if tests.get("run_slow") else "a default run; the slow RUN_SLOW tests were skipped")
        doc.p(f"The test suite (pytest) contains {tests['total']} tests: {tests['passed']} passed, "
              f"{tests['failed']} failed, {tests['skipped']} skipped (run on {tests['date']}, {tests['seconds']:.0f} s, "
              f"{slow}). It covers unit tests of every module, the acceptance tests of all three phases on the real "
              "data and model, and integration tests of the app. Acceptance tests marked 'see evidence' are shown by "
              "result tables rather than a single automated test; 'partly (team)' means a check on another member's "
              "laptop is still to be done.")
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
        doc.table(hr[["category", "contracts", "strict@1", "strict@3", "strict@3_ci", "random@3", "relaxed@3",
                      "random_relaxed@3"]], "E3b: line hit-rate on SmartBugs (primary model) vs random-line baselines",
                  font=8)

        def row(name):
            r = hr[hr["category"] == name]
            return r.iloc[0] if len(r) else hr.iloc[0]
        conf, clean, flg = row("confirmed classes"), row("confirmed, no overlap with training data"), \
            row("confirmed, flagged by the model")
        doc.p(f"Interpretation. For the {int(conf['contracts'])} SmartBugs contracts in confirmed categories, an "
              f"annotated vulnerable line is the single top-ranked line in {pct(conf['strict@1'])} of cases and among "
              f"the top three in {pct(conf['strict@3'])} ({conf['strict@3_ci']}); three random code lines would hit "
              f"{pct(conf['random@3'])}. On the {int(clean['contracts'])} confirmed contracts with no copy in the "
              f"training data the top-3 rate is {pct(clean['strict@3'])}, and on the {int(flg['contracts'])} the model "
              f"actually flagged it is {pct(flg['strict@3'])}. The hit rate is computed for every contract, flagged or "
              "not: the explanation ranks lines for the class even when its probability is low. Counting a hit when a "
              f"top-3 line is in the same function as an annotated line gives {pct(conf['relaxed@3'])}, against "
              f"{pct(conf['random_relaxed@3'])} for three random lines under the same rule. The highlights are "
              "therefore clearly better than chance, but they explain what the model reacts to (for example the "
              "external call rather than the later balance update), so they guide a reviewer rather than pinpoint "
              "the bug. The first row also includes the 18 access-control contracts, whose mapping is only partial.")
    # ---------------- E5
    doc.h("7.5 E5 - Latency and error analysis", 2)
    lat = f.t["e5_latency"]
    if lat is not None:
        doc.table(lat, "E5: latency (milliseconds) on the development laptop (CPU only)", font=8)
    sl = f.t["e5_error_slices"]
    if lat is not None:
        warm = lat[lat["input"].str.endswith(".sol") & ~lat["input"].str.startswith("smartbugs")]
        cold = lat[lat["input"].str.startswith("cold")]
        doc.p(f"A warm prediction takes about {warm['warm_predict_ms_median'].median():.0f} ms for the sample "
              f"contracts and the explanation another {warm['explain_ms'].median():.0f} ms; loading all model files "
              f"once takes {float(cold['first_predict_ms'].iloc[0]) / 1000:.1f} s. A ten-times-longer synthetic file "
              "stays under a second for prediction, so latency is not a practical limit (acceptance test C5).")
    if sl is not None:
        doc.table(sl, "E5: test-split recall / precision of the primary model by contract length and compiler generation",
                  font=8)
        lb = sl[sl["slice"] == "len_bucket"]
        if len(lb) and "positives_per_contract" in lb:
            short, long_ = lb.iloc[0], lb.iloc[-1]
            doc.p(f"Interpretation. Recall rises from {pct(short['micro_recall'])} on the shortest quarter of test "
                  f"contracts to {pct(long_['micro_recall'])} on the longest, but this is correlational: long "
                  f"contracts also carry far more labels ({long_['positives_per_contract']:.2f} vs "
                  f"{short['positives_per_contract']:.2f} weaknesses per contract), and many of those are easy "
                  "families such as unchecked calls. By compiler generation the differences are smaller. The model is "
                  "weakest where a weakness is expressed by little code, which matches the SmartBugs failures below.")
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


# Case analyses written by the team after reading each contract and its explanation (E5).
CASE_NOTES = {
    "bad_randomness/random_number_generator.sol":
        "The randomness is built from block.timestamp, block.number and blockhash, exactly the block variables the "
        "bad-randomness positives in training are made of; the top line is the modulo on the blockhash-derived value.",
    "unchecked_low_level_calls/lotto.sol":
        "Two bare send() calls whose return values are ignored; this is the token pattern behind Slither's "
        "unchecked-send label, and both annotated lines are the top two highlights.",
    "unchecked_low_level_calls/0x663e4229142a27f00bafb5d087e1e730648314c3.sol":
        "A 2,470-line flattened contract. The explanation finds the unchecked msg.sender.send(autoBirthFee) on line "
        "1152; before the review fix that spends the line budget on the most important functions first, this line "
        "was never scored because the candidates were cut in file order.",
    "reentrancy/etherstore.sol":
        "The classic withdraw function: the call.value transfer happens before the balance is reduced, and that "
        "line is ranked first.",
    "reentrancy/0x7a8721a9d64c74da899424c1b52acbf58ddc9782.sol":
        "A small bank contract (a known honeypot template). The external call inside the if condition is ranked "
        "first; the tx.origin check is also highlighted because the two patterns often occur together in training.",
    "bad_randomness/old_blockhash.sol":
        "The bug is blockhash() of a block stored earlier, which returns zero after 256 blocks. There is no "
        "block.timestamp or modulo; the explanation still points at the right line, but the model does not consider "
        "this form risky because it differs from the patterns that dominate the training positives.",
    "unchecked_low_level_calls/0x0cbe050f75bc8f8c2d6c0d249fea125fd6e1acc9.sol":
        "Six lines of code with a bare a.call(). There is almost no context, and the TF-IDF vector of such a tiny "
        "file looks like nothing in the long training contracts, so every probability stays near zero.",
    "bad_randomness/lottery.sol":
        "Randomness from block.number % 2 in old-style Solidity (throw). Parity of the block number is a rare form "
        "of the weakness; the model reacts weakly (p = 0.001) although the annotated line is highlighted first.",
    "reentrancy/modifier_reentrancy.sol":
        "The re-entry happens through a modifier that calls back into an attacker-controlled contract, with no "
        "call.value at all. Token n-grams cannot see this control flow, which is the limit of a model without "
        "program semantics.",
    "reentrancy/reentrancy_bonus.sol":
        "Flagged (p = 0.57) and the external call is ranked first, but the annotated line is the call to "
        "withdrawReward() made before the bonus is marked as claimed. The highlight is one step away from the "
        "line a human marked.",
}


def _case_text(c: dict) -> str:
    top = c["top_line_text"][0] if c["top_line_text"] else "-"
    vul = c["vulnerable_text"][0] if c["vulnerable_text"] else "-"
    note = CASE_NOTES.get(c["id"])
    if c["kind"] == "success":
        head = (f"Success - {c['id']} ({c['class']}): flagged with p = {c['probability']:.2f}; top line "
                f"`{top[:90]}`.")
    else:
        why = "not flagged" if not c["flagged"] else "flagged, but the highlights miss the annotated line"
        head = f"Failure - {c['id']} ({c['class']}): {why} (p = {c['probability']:.2f}); annotated line `{vul[:90]}`."
    return head + " " + (note or "No analysis written for this case.")


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
              f"{pct(s3[s3.key == f.primary]['smartbugs_pooled_recall_confirmed_no_overlap'].iloc[0])} of the "
              "human-annotated SmartBugs contracts that have no copy in the training data, in the categories where a "
              "fair comparison is possible. The random-split ablation "
              "shows how much a careless split would have flattered these numbers, and the hit-rate study shows that "
              "the highlighted lines are useful pointers but not proof.")
    doc.h("Future work", 2)
    doc.bullets([
        "Choose the n-gram range on validation: the E4 ablation suggests 1-2-grams generalise better to SmartBugs.",
        "Remove every SmartBugs contract and its clone group from Dataset A before training, so the external test is fully independent; calibrate per class.",
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
