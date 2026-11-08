"""
Muhafiz.sol - training, model selection, thresholds and freezing (P2.2-P2.4).

    python -m src.train                                  # everything, sequential, resumable
    python -m src.train --quick                          # small grid on a subsample (development)

Long runs can be split into independent processes that share checkpoints
(``models/experiments/candidates``); every finished model is written to disk
immediately, so a crash never loses completed work and a rerun skips it:

    python -m src.train --stage classical --families logreg,forest,knn
    python -m src.train --stage classical --families tree
    python -m src.train --stage mlp --input tfidf --seed 42 --threads 3
    python -m src.train --stage mlp --input svd   --seed 42 --threads 3
    python -m src.train --stage mlp --input tfidf --seed 43 --threads 3
    python -m src.train --stage freeze

``--prefix random_ --fixed-from models/thresholds.json`` trains the same
configurations on the random split (experiment E2).

Discipline (acceptance test B2): this script reads ONLY the train and
validation matrices. The test split and SmartBugs are evaluated by
``src.run_experiments`` after the frozen model has been tagged ``model-v1``.

Steps
 1. small hyper-parameter grid per family, scored by validation macro PR-AUC
 2. MLP: TF-IDF input vs SVD input (seed 42), then 3 seeds on the better input
 3. per-class thresholds maximising F2 on validation with a precision guard
 4. calibration check (ECE); Platt scaling fitted on validation if needed
 5. primary model = best validation macro PR-AUC; freeze artefacts to models/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import load_config, path, read_json, seed, set_global_seed, write_json
from src.evaluate import (choose_thresholds, expected_calibration_error, pr_auc, summary_metrics)
from src.models import ForestModel, KNNModel, LogRegModel, MLPModel, PriorModel, TreeModel

CLASSICAL = ("prior", "logreg", "tree", "forest", "knn")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_xy(split: str, kind: str = "tfidf", prefix: str = "") -> Tuple[object, np.ndarray]:
    """Load features and labels for 'train'/'val' (or 'random_train'... with prefix)."""
    feat = path("features_dir")
    name = f"{prefix}{split}"
    if name.endswith("test") or name == "gold":
        raise PermissionError("train.py must not read test or gold data (acceptance test B2)")
    Y = np.load(feat / f"Y_{name}.npy")
    X = sp.load_npz(feat / f"X_{name}.npz").tocsr() if kind == "tfidf" else np.load(feat / f"Xsvd_{name}.npy")
    return X, Y


def file_size_mb(p: Path) -> float:
    return Path(p).stat().st_size / 1e6


# --------------------------------------------------------------------------- #
# Platt calibration (per class, fitted on validation)
# --------------------------------------------------------------------------- #
def fit_platt(y_val: np.ndarray, p_val: np.ndarray) -> List[Tuple[float, float]]:
    from sklearn.linear_model import LogisticRegression

    params = []
    for c in range(y_val.shape[1]):
        z = np.log(np.clip(p_val[:, c], 1e-6, 1 - 1e-6) / np.clip(1 - p_val[:, c], 1e-6, 1)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_val[:, c])
        params.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
    return params


def apply_platt(p: np.ndarray, params) -> np.ndarray:
    if not params:
        return p
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    a = np.array([q[0] for q in params])
    b = np.array([q[1] for q in params])
    return 1.0 / (1.0 + np.exp(-(z * a + b)))


# --------------------------------------------------------------------------- #
def grid(quick: bool) -> Dict[str, List[Dict]]:
    if quick:
        return {"prior": [{}], "logreg": [{"C": 1.0}], "tree": [{"max_depth": 16, "min_samples_leaf": 5}],
                "forest": [{"n_estimators": 60, "max_depth": None}], "knn": [{"k": 15}]}
    return {
        "prior": [{}],
        "logreg": [{"C": c} for c in (0.25, 1.0, 4.0)],
        "tree": [{"max_depth": d, "min_samples_leaf": m} for d in (16, 32) for m in (5, 20)],
        "forest": [{"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
                   {"n_estimators": 200, "max_depth": 40, "min_samples_leaf": 2}],
        "knn": [{"k": k} for k in (5, 15, 31)],
    }


def build(family: str, params: Dict, s: int, n_jobs: Optional[int] = None):
    """``n_jobs`` only changes parallelism (not results) so jobs can share the CPU."""
    jobs = {"n_jobs": n_jobs} if n_jobs else {}
    if family == "prior":
        return PriorModel()
    if family == "logreg":
        return LogRegModel(**params, **jobs)
    if family == "tree":
        return TreeModel(seed=s, **params)
    if family == "forest":
        return ForestModel(seed=s, **params, **jobs)
    if family == "knn":
        return KNNModel(**params)
    raise ValueError(family)


# --------------------------------------------------------------------------- #
# Data and checkpoints
# --------------------------------------------------------------------------- #
def load_data(prefix: str = "", quick: bool = False, svd: bool = True) -> Dict[str, object]:
    Xtr, Ytr = load_xy("train", "tfidf", prefix)
    Xva, Yva = load_xy("val", "tfidf", prefix)
    Str = Sva = None
    if svd:
        if prefix == "":
            Str, _ = load_xy("train", "svd", prefix)
            Sva, _ = load_xy("val", "svd", prefix)
        else:  # SVD of the random split's own vectors (the SVD was fitted on the cluster train split)
            vec = joblib.load(path("models_dir") / "vectorizer.joblib")
            Str, Sva = vec.transform_svd(Xtr), vec.transform_svd(Xva)
    if quick:
        rng = np.random.default_rng(seed())
        sub = np.sort(rng.choice(Xtr.shape[0], min(8000, Xtr.shape[0]), replace=False))
        Xtr, Ytr = Xtr[sub], Ytr[sub]
        Str = Str[sub] if Str is not None else None
    return {"Xtr": Xtr, "Ytr": Ytr, "Xva": Xva, "Yva": Yva, "Str": Str, "Sva": Sva}


def candidate_dir(out_dir: Optional[Path] = None, quick: bool = False) -> Path:
    d = (out_dir or (path("models_dir") / "experiments")) / ("candidates_quick" if quick else "candidates")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(obj, p: Path) -> None:
    """Write to a temporary file first so an interrupted save never leaves a corrupt checkpoint."""
    tmp = p.with_suffix(".tmp")
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, p)


def _grid_row(family: str, params: Dict, Yva, pv, fit_s: float, source: str = "fitted") -> Dict:
    aps = pr_auc(Yva, pv)
    return {"family": family, "params": json.dumps(params), "val_macro_pr_auc": float(np.nanmean(aps)),
            "fit_seconds": round(fit_s, 1), "source": source, **{f"val_pr_auc_{i}": a for i, a in enumerate(aps)}}


def fit_family(family: str, data: Dict, cdir: Path, quick: bool = False, prefix: str = "",
               fixed: Optional[Dict[str, Dict]] = None, n_jobs: Optional[int] = None) -> Dict:
    """Fit the grid of one family, keep the best configuration by validation macro PR-AUC."""
    out = cdir / f"{family}.joblib"
    if out.exists():
        _log(f"{prefix or 'cluster'} {family}: checkpoint exists, skipped")
        return joblib.load(out)
    s = seed()
    set_global_seed(s)
    configs = [fixed[family]] if fixed and family in fixed else grid(quick)[family]
    best, rows = None, []
    for params in configs:
        t0 = time.time()
        model = build(family, params, s, n_jobs)
        Xa, Xb = (data["Str"], data["Sva"]) if family == "knn" else (data["Xtr"], data["Xva"])
        model.fit(Xa, data["Ytr"])
        fit_s = time.time() - t0
        pv = model.predict_proba(Xb)
        row = _grid_row(family, params, data["Yva"], pv, fit_s)
        rows.append(row)
        _log(f"{prefix or 'cluster'} {family} {params}: val macro PR-AUC {row['val_macro_pr_auc']:.4f} ({fit_s:.0f}s)")
        if best is None or row["val_macro_pr_auc"] > best["val_ap"]:
            best = {"family": family, "model": model, "params": params, "val_proba": pv,
                    "val_ap": row["val_macro_pr_auc"], "fit_seconds": fit_s}
    best["grid_rows"] = rows
    _save(best, out)
    return best


def mlp_common(quick: bool) -> Dict:
    mcfg = load_config()["training"]["mlp"]
    return dict(hidden=mcfg["hidden"], dropout=mcfg["dropout"], lr=mcfg["lr"],
                weight_decay=mcfg["weight_decay"], batch_size=mcfg["batch_size"],
                max_epochs=6 if quick else mcfg["max_epochs"], patience=mcfg["patience"])


def fit_mlp(input_kind: str, sd: int, data: Dict, cdir: Path, quick: bool = False, prefix: str = "") -> Dict:
    out = cdir / f"mlp_{input_kind}_s{sd}.joblib"
    if out.exists():
        _log(f"{prefix or 'cluster'} mlp input={input_kind} seed={sd}: checkpoint exists, skipped")
        return joblib.load(out)
    Xa, Xb = (data["Xtr"], data["Xva"]) if input_kind == "tfidf" else (data["Str"], data["Sva"])
    t0 = time.time()
    m = MLPModel(seed=sd, **mlp_common(quick)).fit(Xa, data["Ytr"], Xb, data["Yva"])
    pv = m.predict_proba(Xb)
    ap = float(np.nanmean(pr_auc(data["Yva"], pv)))
    run = {"input": input_kind, "seed": sd, "model": m, "val_proba": pv, "val_ap": ap,
           "fit_seconds": time.time() - t0}
    _save(run, out)
    _log(f"{prefix or 'cluster'} mlp input={input_kind} seed={sd}: val macro PR-AUC {ap:.4f}")
    return run


def assemble(cdir: Path, Yva: np.ndarray, fixed: Optional[Dict[str, Dict]] = None, quick: bool = False) -> Dict:
    """Collect the checkpoints into the results dict used by freeze() and experiment E2."""
    results: Dict[str, Dict] = {}
    rows: List[Dict] = []
    for fam in CLASSICAL:
        f = cdir / f"{fam}.joblib"
        if not f.exists():
            raise FileNotFoundError(f"missing checkpoint {f}; run --stage classical --families {fam}")
        results[fam] = joblib.load(f)
        rows += results[fam]["grid_rows"]
    seeds = load_config()["training"]["mlp"]["seeds"]
    runs = {(r["input"], r["seed"]): r for r in (joblib.load(f) for f in sorted(cdir.glob("mlp_*_s*.joblib")))}
    if fixed and "mlp" in fixed:
        input_kind, chosen_seeds = fixed["mlp"]["input"], [fixed["mlp"]["seed"]]
    else:
        first = {k: runs[(k, seeds[0])]["val_ap"] for k in ("tfidf", "svd") if (k, seeds[0]) in runs}
        if len(first) < 2:
            raise FileNotFoundError("need the seed-%d MLP runs for both inputs (tfidf, svd)" % seeds[0])
        input_kind, chosen_seeds = max(first, key=first.get), seeds
    missing = [s for s in chosen_seeds if (input_kind, s) not in runs]
    if missing:
        raise FileNotFoundError(f"missing MLP checkpoints input={input_kind} seeds={missing}")
    chosen = [runs[(input_kind, s)] for s in chosen_seeds]
    for r in sorted(runs.values(), key=lambda r: (r["seed"], r["input"])):
        if fixed and "mlp" in fixed and (r["input"], r["seed"]) != (input_kind, chosen_seeds[0]):
            continue
        rows.append(_grid_row("mlp", {"input": r["input"], "seed": r["seed"]}, Yva, r["val_proba"],
                              r["fit_seconds"]))
    best_mlp = max(chosen, key=lambda r: r["val_ap"])
    results["mlp"] = {"model": best_mlp["model"], "params": {"input": input_kind, "seed": best_mlp["seed"]},
                      "val_proba": best_mlp["val_proba"], "val_ap": best_mlp["val_ap"],
                      "fit_seconds": best_mlp["fit_seconds"], "seed_aps": [r["val_ap"] for r in chosen],
                      "runs": chosen}
    results["_val_table"] = pd.DataFrame(rows)
    results["_Yva"] = Yva
    return results


def train_all(quick: bool = False, prefix: str = "", out_dir: Path | None = None,
              fixed: Dict[str, Dict] | None = None) -> Dict[str, Dict]:
    """Fit every family (resuming from checkpoints); return {family: {model, params, val_proba, val_ap}}.

    ``fixed`` re-uses hyper-parameters chosen earlier (used by E2 on the random
    split so both splits use the same configurations).
    """
    cdir = candidate_dir(out_dir, quick)
    data = load_data(prefix, quick)
    for fam in CLASSICAL:
        fit_family(fam, data, cdir, quick, prefix, fixed)
    seeds = load_config()["training"]["mlp"]["seeds"]
    if fixed and "mlp" in fixed:
        fit_mlp(fixed["mlp"]["input"], fixed["mlp"]["seed"], data, cdir, quick, prefix)
    else:
        for kind in ("tfidf", "svd"):
            fit_mlp(kind, seeds[0], data, cdir, quick, prefix)
        first = {k: joblib.load(cdir / f"mlp_{k}_s{seeds[0]}.joblib")["val_ap"] for k in ("tfidf", "svd")}
        kind = max(first, key=first.get)
        for sd in seeds[1:]:
            fit_mlp(kind, sd, data, cdir, quick, prefix)
    return assemble(cdir, data["Yva"], fixed, quick)


# --------------------------------------------------------------------------- #
def freeze(results: Dict, quick: bool = False) -> Dict:
    """Thresholds, calibration, primary-model choice and saving of artefacts."""
    cfg = load_config()
    classes = read_json(path("models_dir") / "label_map.json")["target_classes"]
    tcfg = cfg["training"]
    Yva = results["_Yva"]
    models_dir = path("models_dir")
    exp_dir = models_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    tables = path("tables_dir")

    families = [f for f in results if not f.startswith("_")]
    selection, thr_rows = [], []
    frozen: Dict[str, Dict] = {}
    for fam in families:
        r = results[fam]
        pv = r["val_proba"]
        ece = float(np.mean([expected_calibration_error(Yva[:, c], pv[:, c]) for c in range(len(classes))]))
        platt = None
        if fam not in ("prior",) and ece > 0.05:
            platt = fit_platt(Yva, pv)
            pv = apply_platt(pv, platt)
        ece_after = float(np.mean([expected_calibration_error(Yva[:, c], pv[:, c]) for c in range(len(classes))]))
        thr, table = choose_thresholds(Yva, pv, classes, tcfg["threshold"]["beta"], tcfg["threshold"]["min_precision"])
        table.insert(0, "model", fam)
        thr_rows.append(table)
        sm = summary_metrics(Yva, pv, thr)
        frozen[fam] = {"thresholds": thr.tolist(), "platt": platt, "params": r["params"]}
        model = r["model"].compact() if fam == "mlp" else r["model"]
        _save(model, exp_dir / f"{fam}.joblib")
        selection.append({"model": fam, "params": json.dumps(r["params"]), "val_macro_pr_auc": r["val_ap"],
                          "val_macro_f2": sm["macro_f2"], "val_macro_f1": sm["macro_f1"],
                          "ece_before": ece, "ece_after": ece_after, "calibrated": platt is not None,
                          "fit_seconds": round(r["fit_seconds"], 1),
                          "size_mb": round(file_size_mb(exp_dir / f"{fam}.joblib"), 2)})

    sel = pd.DataFrame(selection).sort_values("val_macro_pr_auc", ascending=False)
    candidates = sel[sel["model"] != "prior"]
    primary = str(candidates.iloc[0]["model"])
    sel["primary"] = sel["model"] == primary
    tables.mkdir(parents=True, exist_ok=True)
    sel.to_csv(tables / "model_selection_val.csv", index=False, lineterminator="\n", float_format="%.5f")
    pd.concat(thr_rows).to_csv(tables / "thresholds_val.csv", index=False, lineterminator="\n", float_format="%.5f")
    results["_val_table"].to_csv(tables / "validation_grid.csv", index=False, lineterminator="\n", float_format="%.5f")

    # every MLP seed is kept so E1 can report the test spread across seeds (B5)
    for r in results["mlp"]["runs"]:
        pv = r["val_proba"]
        platt_s = fit_platt(Yva, pv) if frozen["mlp"]["platt"] else None
        thr_s, _ = choose_thresholds(Yva, apply_platt(pv, platt_s), classes,
                                     tcfg["threshold"]["beta"], tcfg["threshold"]["min_precision"])
        _save({"model": r["model"].compact(), "thresholds": thr_s.tolist(), "seed": r["seed"],
               "input": r["input"], "platt": platt_s}, exp_dir / f"mlp_seed{r['seed']}.joblib")
    mlp_seeds = results["mlp"]["seed_aps"]
    pd.DataFrame({"seed": [r["seed"] for r in results["mlp"]["runs"]], "val_macro_pr_auc": mlp_seeds,
                  "best_epoch": [r["model"].history.best_epoch + 1 for r in results["mlp"]["runs"]],
                  "epochs_run": [len(r["model"].history.train_loss) for r in results["mlp"]["runs"]]}
                 ).to_csv(tables / "mlp_seeds.csv", index=False, lineterminator="\n", float_format="%.5f")
    _plot_learning_curves(results["mlp"]["runs"])

    # app models: the primary plus logistic regression (F12 comparison toggle)
    app_models = sorted({primary, "logreg"} | ({"mlp"} if "mlp" in families else set()))
    for fam in app_models:
        src = exp_dir / f"{fam}.joblib"
        dst = models_dir / f"model_{fam}.joblib"
        dst.write_bytes(src.read_bytes())
    thresholds = {
        "primary": primary,
        "beta": tcfg["threshold"]["beta"],
        "min_precision": tcfg["threshold"]["min_precision"],
        "high_margin": tcfg["risk"]["high_margin"],
        "classes": classes,
        "models": {fam: frozen[fam] for fam in app_models},
        "all_models": frozen,
        "mlp_seed_val_pr_auc": {"mean": float(np.mean(mlp_seeds)), "std": float(np.std(mlp_seeds)),
                                "values": mlp_seeds},
        "quick": quick,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(thresholds, models_dir / "thresholds.json")
    _build_knn_index()
    _log(f"primary model: {primary}; frozen artefacts in {models_dir}")
    return thresholds


def _build_knn_index() -> None:
    """Index of training contracts for similar(): the first SIMILAR_DIMS SVD components
    (most of the variance), L2-normalised, float16 - keeps the committed file small."""
    from src.models import l2_normalise
    from src.predict import SIMILAR_DIMS

    S = np.load(path("features_dir") / "Xsvd_train.npy")[:, :SIMILAR_DIMS]
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet",
                         columns=["address", "split", "labels", "contract_name", "snippet", "n_lines", "group"])
    df = df[df["split"] == "train"].reset_index(drop=True)
    np.save(path("models_dir") / "similar_index.npy", l2_normalise(S).astype(np.float16))
    df[["address", "contract_name", "labels", "snippet", "n_lines", "group"]].to_parquet(
        path("models_dir") / "similar_meta.parquet", index=False, compression="zstd")


def _plot_learning_curves(runs) -> None:
    from src import plotstyle
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, r in enumerate(runs):
        h = r["model"].history
        ep = np.arange(1, len(h.train_loss) + 1)
        col = plotstyle.SERIES[k]
        axes[0].plot(ep, h.train_loss, color=col, label=f"seed {r['seed']} train")
        axes[0].plot(ep[:len(h.val_loss)], h.val_loss, color=col, linestyle="--", label=f"seed {r['seed']} val")
        axes[1].plot(ep[:len(h.val_pr_auc)], h.val_pr_auc, color=col, marker="o", markersize=4,
                     label=f"seed {r['seed']}")
        axes[1].axvline(h.best_epoch + 1, color=col, alpha=0.35, linewidth=1)
    axes[0].set_title("MLP loss (weighted BCE)")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Validation macro PR-AUC (early stopping)")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    plotstyle.save(fig, path("figures_dir") / "mlp_learning_curves.png")


def _fixed_from(p: Optional[str]) -> Optional[Dict[str, Dict]]:
    if not p:
        return None
    meta = read_json(Path(p))
    return {f: meta["all_models"][f]["params"] for f in meta["all_models"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["all", "classical", "mlp", "freeze"], default="all")
    ap.add_argument("--families", default=",".join(CLASSICAL), help="classical stage: comma-separated families")
    ap.add_argument("--input", choices=["tfidf", "svd"], default="tfidf", help="mlp stage: input features")
    ap.add_argument("--seed", type=int, default=None, help="mlp stage: seed")
    ap.add_argument("--threads", type=int, default=None, help="mlp stage: torch threads")
    ap.add_argument("--n-jobs", type=int, default=None, help="classical stage: cores for logreg/forest")
    ap.add_argument("--prefix", default="", help="'random_' trains on the random split (E2)")
    ap.add_argument("--fixed-from", default=None, help="thresholds.json whose chosen params are reused")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    fixed = _fixed_from(args.fixed_from)
    if args.stage == "all":
        res = train_all(quick=args.quick, prefix=args.prefix, out_dir=out_dir, fixed=fixed)
        if not args.prefix:
            freeze(res, quick=args.quick)
    elif args.stage == "classical":
        fams = [f for f in args.families.split(",") if f]
        data = load_data(args.prefix, args.quick, svd="knn" in fams)
        for fam in fams:
            fit_family(fam, data, candidate_dir(out_dir, args.quick), args.quick, args.prefix, fixed, args.n_jobs)
    elif args.stage == "mlp":
        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        data = load_data(args.prefix, args.quick, svd=args.input == "svd")
        sd = args.seed if args.seed is not None else load_config()["training"]["mlp"]["seeds"][0]
        fit_mlp(args.input, sd, data, candidate_dir(out_dir, args.quick), args.quick, args.prefix)
    else:
        if args.prefix:
            raise SystemExit("freeze is only defined for the cluster-aware split")
        _, Yva = load_xy("val")
        freeze(assemble(candidate_dir(out_dir, args.quick), Yva, quick=args.quick), quick=args.quick)
