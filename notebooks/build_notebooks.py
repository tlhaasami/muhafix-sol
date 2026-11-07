"""
Generate and execute the four project notebooks (blueprint 7.3):

  01_data_audit.ipynb   label semantics (label 4), Slither failures, detector frequencies, 10 sample rows
  02_eda.ipynb          class balance per split, lengths, duplicates, compiler versions
  03_models.ipynb       validation grid, model selection, thresholds, MLP learning curves
  04_results.ipynb      E1-E5 tables and figures

    python notebooks/build_notebooks.py          (requires requirements-train.txt)

The notebooks only *read* artefacts produced by src.build_data, src.train,
src.run_experiments and src.eval_explain, so they stay consistent with the report.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option('display.max_colwidth', 120)
T = ROOT / 'reports' / 'tables'
F = ROOT / 'reports' / 'figures'"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


NOTEBOOKS = {
    "01_data_audit.ipynb": [
        md("# 01 - Data audit (Dataset A: Slither Audited Smart Contracts)\n"
           "Blueprint P1.2. Answers: how many contracts, did Slither succeed, what does label 4 mean, "
           "which detectors occur, and does a random sample of contracts look right?"),
        code(SETUP),
        md("## Filtering funnel"),
        code("pd.read_csv(T / 'dataset_funnel.csv')"),
        md("## The label-4 question\nWe re-create the integer ids the upstream loading script emits for each "
           "configuration from the raw Slither output. In the *big* and *small* configurations id 4 is `safe` and "
           "never co-occurs with another id. In the *all* configuration id 4 is `divide-before-multiply`, it does "
           "co-occur with other ids, and ids go up to 38 (`safe`). The dataset viewer shows the *all* "
           "configuration, which explains the confusing rows noted in the blueprint."),
        code("pd.read_csv(T / 'label4_audit.csv')"),
        md("## Slither detectors and label families (de-duplicated pool)"),
        code("pd.read_csv(T / 'detector_frequency.csv').head(40)"),
        code("pd.read_csv(T / 'family_distribution.csv')"),
        md("## Ten random contracts with their labels (sanity check)"),
        code("df = pd.read_parquet(ROOT / 'data/processed/dataset.parquet', columns=['address','contract_name','labels','detectors','n_lines','snippet'])\n"
             "sample = df.sample(10, random_state=42)\n"
             "for r in sample.itertuples():\n"
             "    print(f'{r.address}  {r.contract_name}  lines={r.n_lines}')\n"
             "    print('  detectors:', list(r.detectors)); print('  classes  :', list(r.labels))\n"
             "    print('  ' + '\\n  '.join(r.snippet.splitlines()[:4])); print()"),
    ],
    "02_eda.ipynb": [
        md("# 02 - Exploratory data analysis\nBlueprint P1.8: class balance per split, contract lengths, "
           "duplicate statistics and compiler versions (domain shift to SmartBugs)."),
        code(SETUP),
        code("display(Image(F / 'label_distribution.png'))\npd.read_csv(T / 'label_distribution.csv').pivot_table(index='class', columns=['split_type','split'], values='positives')"),
        code("display(Image(F / 'label_cooccurrence.png'))"),
        code("display(Image(F / 'length_histogram.png'))\npd.read_csv(T / 'length_stats.csv')"),
        code("display(Image(F / 'duplicate_groups.png'))\njson.load(open(ROOT / 'data/processed/build_summary.json'))['dedup']"),
        code("display(Image(F / 'solidity_versions.png'))\npd.read_csv(T / 'solidity_versions.csv')"),
        code("pd.read_csv(T / 'smartbugs_counts.csv')"),
    ],
    "03_models.ipynb": [
        md("# 03 - Models, tuning and selection (validation data only)\nBlueprint P2.2-P2.4. Every number in this "
           "notebook comes from the validation split; the test split and SmartBugs were not used."),
        code(SETUP),
        code("pd.read_csv(T / 'validation_grid.csv')"),
        code("pd.read_csv(T / 'model_selection_val.csv')"),
        code("pd.read_csv(T / 'thresholds_val.csv')"),
        code("display(Image(F / 'mlp_learning_curves.png'))\npd.read_csv(T / 'mlp_seeds.csv')"),
        code("json.load(open(ROOT / 'models/thresholds.json'))['primary']"),
    ],
    "04_results.ipynb": [
        md("# 04 - Results (frozen model-v1)\nExperiments E1-E5 from blueprint 8.7."),
        code(SETUP),
        md("## E1 - model comparison on the cluster-aware test split"),
        code("display(pd.read_csv(T / 'e1_test_summary.csv'))\ndisplay(Image(F / 'e1_per_class_pr_auc.png'))\n"
             "display(Image(F / 'e1_pr_and_calibration.png'))\npd.read_csv(T / 'e1_mlp_seeds_test.csv')"),
        md("## E2 - clone leakage (random vs cluster-aware split)"),
        code("display(pd.read_csv(T / 'e2_leakage.csv'))\ndisplay(Image(F / 'e2_leakage.png'))"),
        md("## E3 - SmartBugs Curated (independent, human-annotated)"),
        code("display(pd.read_csv(T / 'e3_smartbugs_recall.csv'))\npd.read_csv(T / 'e3_summary.csv')"),
        md("## E3b - do the highlighted lines hit the annotated lines?"),
        code("pd.read_csv(T / 'e3b_hit_rate.csv')"),
        md("## E4 - ablations"),
        code("pd.read_csv(T / 'e4_ablations.csv')"),
        md("## E5 - latency, error slices and case studies"),
        code("display(pd.read_csv(T / 'e5_latency.csv'))\ndisplay(pd.read_csv(T / 'e5_error_slices.csv'))\n"
             "pd.DataFrame(json.load(open(T / 'e5_case_studies.json')))"),
    ],
}


def build(execute: bool = True, only=None) -> None:
    for name, cells in NOTEBOOKS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbf.write(nb, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    import sys

    build(only=sys.argv[1:] or None)   # e.g. python notebooks/build_notebooks.py 01 02
