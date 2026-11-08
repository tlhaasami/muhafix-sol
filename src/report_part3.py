"""
Report chapters 6 (implementation) and 8 (limitations and ethics) plus the reference list.
"""

from __future__ import annotations

import pandas as pd

from src.config import load_config, path
from src.report_utils import Facts

FIG = path("figures_dir")

REFERENCES = [
    "Rossini, M. (2022). Slither Audited Smart Contracts Dataset. Hugging Face: huggingface.co/datasets/mwritescode/slither-audited-smart-contracts; GitHub: github.com/mwritescode/slither-audited-smart-contracts.",
    "SmartBugs Curated dataset. GitHub: github.com/smartbugs/smartbugs-curated (README, vulnerabilities.json, ICSE2020_curated_69.txt), commit 230e649.",
    "Durieux, T., Ferreira, J. F., Abreu, R., Cruz, P. (2020). Empirical review of automated analysis tools on 47,587 Ethereum smart contracts. ICSE 2020.",
    "Ferreira, J. F., Cruz, P., Durieux, T., Abreu, R. (2020). SmartBugs: A Framework to Analyze Solidity Smart Contracts. arXiv:2007.04771.",
    "NCC Group. Decentralized Application Security Project (DASP) Top 10 (dasp.co). Smart Contract Weakness Classification (SWC) Registry (swcregistry.io).",
    "Feist, J., Grieco, G., Groce, A. (2019). Slither: a static analysis framework for smart contracts. github.com/crytic/slither.",
    "Chainalysis (2026). Crypto hacking and stolen funds, 2025. chainalysis.com/blog/crypto-hacking-stolen-funds-2026/.",
    "CertiK. Hack3d: The Web3 Security Report 2025. certik.com/blog/hack3d-the-web3-security-report-2025.",
    "Smart Contract Security Statistics 2026 (secondary summary citing TRM Labs). spacedev.io/blog/smart-contract-security-statistics-2026.",
    "CoinEdition via CryptoRank (7 March 2026). Pakistan Passes Virtual Assets Act 2026 to Catalyze Crypto Adoption.",
    "Tang, X. et al. (2023). Lightning Cat: Deep learning-based solution for smart contract vulnerabilities detection. Scientific Reports 13, 20106. nature.com/articles/s41598-023-47219-0.",
    "Wei, Z. et al. (2024). FTSmartAudit: A Knowledge Distillation-Enhanced Framework for Automated Smart Contract Auditing Using Fine-Tuned LLMs. arXiv:2410.13918.",
    "Broder, A. Z. (1997). On the resemblance and containment of documents. Compression and Complexity of Sequences (MinHash).",
    "Sechidis, K., Tsoumakas, G., Vlahavas, I. (2011). On the stratification of multi-label data. ECML PKDD (iterative stratification).",
    "Zeiler, M. D., Fergus, R. (2014). Visualizing and understanding convolutional networks. ECCV (occlusion sensitivity).",
    "Platt, J. (1999). Probabilistic outputs for support vector machines (Platt scaling).",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12; scikit-learn.org. PyTorch (pytorch.org); Streamlit (docs.streamlit.io); datasketch (ekzhu.com/datasketch).",
    "National University of Computer and Emerging Sciences, Islamabad. AI2002 Artificial Intelligence, Semester Project Statement, Fall 2026.",
]


def chapter6(doc, f: Facts) -> None:
    cfg = load_config()["explain"]
    doc.h("6. Implementation Details")
    doc.h("6.1 Code structure", 2)
    doc.p("The repository follows blueprint section 7.3. Every script reads its paths, seeds and hyper-parameters "
          "from config.yaml, so nothing is hard-coded. Large data files are rebuilt by one command and are not "
          "committed; the frozen model files are committed so the app runs from a fresh clone.")
    doc.table(pd.DataFrame([
        ["python -m src.build_data", "Phase 1: download, gold set, tokens, de-duplication, splits, TF-IDF, EDA"],
        ["python -m src.train", "Phase 2: validation grid, thresholds, calibration, freeze to models/"],
        ["git tag model-v1", "freeze marker checked by the evaluation scripts"],
        ["python -m src.run_experiments", "E1-E4 tables and figures"],
        ["python -m src.eval_explain", "E3b and E5"],
        ["python -m src.make_report", "this report, the slides and the model / data cards"],
        ["streamlit run app/app.py", "the application"],
        ["pytest", "unit, acceptance and integration tests"],
    ], columns=["Command", "What it does"]), "One-command entry points", widths=[5.5, 11.3])
    doc.h("6.2 Preprocessing implementation", 2)
    doc.p("clean_source is a state machine with four states (normal, string, line comment, block comment). Instead of "
          "stepping through one character at a time in Python it jumps between the characters that can change state "
          "using pre-compiled regular expressions, which made it about fifty times faster on the million-character "
          "flattened contracts of Dataset A while keeping exactly the same semantics. Comment characters become "
          "spaces and every newline is kept, so both line and column positions survive. Edge cases covered by unit "
          "tests include comment markers inside strings, escaped quotes, unterminated strings and comments, Windows "
          "and old-Mac line endings, unicode and binary junk.")
    doc.h("6.3 Explanation module (occlusion)", 2)
    doc.p("The explanation answers \"which lines does this score depend on?\". It is computed in two stages so it stays "
          f"fast: (1) function-like blocks (function, modifier, constructor, fallback, receive) are found by brace "
          f"matching on the cleaned code, and each block (up to {cfg['max_functions']}) is removed in turn; (2) the "
          f"single lines inside the {cfg['top_functions']} blocks with the largest probability drop - plus top-level "
          f"lines - are removed one at a time (up to {cfg['max_lines']} lines). The score of a line is the drop in "
          "the class probability when that line is blanked; the top-k lines are highlighted.")
    doc.p("Re-vectorising the whole contract for every removed line would be slow for long files. Because TF-IDF "
          "only counts n-grams of at most three tokens, removing a span S changes only the n-grams that touch S. With "
          "L and R the two tokens on either side of S, the counts without S are exactly counts(full) - counts(L+S+R) "
          "+ counts(L+R). Only these tiny windows are vectorised, which is mathematically identical to "
          "re-vectorising (verified by an integration test) and makes line-level occlusion take well under a second. "
          "The module only needs the predictor's probability function, so it works with any of the models.")
    doc.h("6.4 Streamlit application", 2)
    doc.p("The app (app/app.py) implements the layout of blueprint section 9.1 with four tabs. The sidebar holds the "
          "sample-contract selector, the model selector (the recommended model and logistic regression for "
          "comparison), the sensitivity slider and the model card. The scan tab takes pasted code or a .sol upload, "
          "validates it (empty, an upload larger than 2 MB or text longer than 1.5 million characters, too short, or "
          "not Solidity), and shows: a risk badge with an icon, "
          "a word and the rule in plain language (never colour alone); a probability bar per weakness type with its "
          "alert level marked; a plain-language explanation and typical fix for the selected type; the source code "
          "with line numbers and the top-k lines highlighted; a function-level view; three similar training "
          "contracts with Etherscan links; and Markdown / JSON report downloads. Further tabs show the evaluation "
          "tables (read from the same CSV files as this report), a batch scan that ranks several files by risk, and "
          "the limitations. The model loads once (st.cache_resource) and results survive interaction "
          "(st.session_state). The footer always shows the audit disclaimer.")
    shots = sorted(FIG.glob("app_*.png"))
    for s in shots:
        doc.figure(s, s.stem.replace("app_", "App screenshot: ").replace("_", " "), 15.5)
    doc.table(pd.DataFrame([
        ["F1", "Paste or upload", "Must", "done"], ["F2", "Six sample contracts", "Must", "done"],
        ["F3", "Input validation", "Must", "done (fuzz-tested)"], ["F4", "Class probabilities with threshold", "Must", "done"],
        ["F5", "Risk level with rule", "Must", "done"], ["F6", "Highlighted lines", "Must", "done"],
        ["F7", "Metrics and limitations panel", "Must", "done (reads report tables)"],
        ["F8", "One-command run", "Must", "done on the development laptop; other two laptops pending"],
        ["F9", "Similar known contracts", "Should", "done"], ["F10", "Sensitivity slider", "Should", "done (monotone, tested)"],
        ["F11", "Download scan report", "Should", "done (Markdown and JSON)"], ["F12", "Model comparison toggle", "Should", "done"],
        ["F13", "Batch scan", "Could", "done"], ["F14", "Function-level view", "Could", "done"],
        ["F15", "Cloud deployment", "Could", "not done (needs the team's Streamlit Cloud account)"],
        ["F16", "Exploits, auto-fix, address scanning, bytecode, Slither at runtime", "Won't", "excluded"],
    ], columns=["ID", "Feature", "Priority", "Status"]), "MoSCoW feature status", widths=[1.2, 8, 1.8, 5.8])


def chapter8(doc, f: Facts) -> None:
    doc.h("8. Limitations and Ethics")
    doc.h("8.1 Technical limitations", 2)
    doc.bullets([
        ("Weak labels. ", "Training targets are Slither findings. The model learns to imitate Slither, including its false "
                          "positives and blind spots; in-distribution scores measure agreement with a tool, not with experts."),
        ("Missing weakness types. ", "Integer overflow, denial of service, front running, timestamp dependence and short "
                                     "addresses have no training label, so the model cannot flag them at all, however obvious."),
        ("Domain shift. ", "The training data is long, real, mostly modern Solidity; SmartBugs and many teaching examples are "
                           "short Solidity 0.4 snippets. Recall on such code is lower (E3, E5)."),
        ("No semantic understanding. ", "TF-IDF over token n-grams sees which tokens appear near each other, not control or "
                                        "data flow. Code that uses suspicious keywords correctly can be flagged, and unusual "
                                        "vulnerable code can be missed."),
        ("Small gold set, partly seen in training. ", "SmartBugs has 143 contracts; per-class intervals are wide and only "
                                     "recall can be measured. 42 of them also occur (exactly or nearly) in the training "
                                     "pool; we found this after the freeze and report SmartBugs results with and "
                                     "without them (E3, E3b)."),
        ("Calibration. ", "The calibration gate used the class-averaged error, so the primary MLP was not rescaled although "
                          "several classes are over-confident; its percentages are scores, not exact probabilities."),
        ("Explanations explain the model. ", "Occlusion shows which lines move the score, which can be correlated code rather "
                                             "than the bug itself (hit-rates in E3b quantify this)."),
        ("Long files. ", "Very long flattened files are truncated to a token budget (head and tail kept); lines in the middle "
                         "of such files are not analysed and the app says so."),
    ])
    doc.h("8.2 Ethics and safety", 2)
    doc.bullets([
        "The project is defensive only: it detects and explains weaknesses and never generates exploits or attack code.",
        "False negatives are the main risk. The interface and this report say clearly that the tool does not replace a "
        "professional audit and never certifies safety; the risk badge for 'low' says so explicitly.",
        "Labels come from a tool, so its errors propagate; this is disclosed and tested against an independent benchmark.",
        "Contracts are public and contain no personal data; each source keeps its original licence and both datasets are cited.",
        "The tool is not presented as meeting any regulatory requirement and is not affiliated with PVARA.",
        "Misuse: attackers can already run public analysers; a triage score does not give them new capability.",
    ])


def references(doc) -> None:
    doc.h("References")
    doc.numbered(REFERENCES)
