"""
Adaptive, Hyperparameter-Optimised Hybrid Ensemble Framework
for Explainable Credit Card Fraud Detection
------------------------------------------------------------
MSc Data Science thesis, IU International University of Applied Sciences.

Pipeline (leakage-conscious):
  1. Load + validate data, remove exact duplicates
  2. Stratified 80:20 development / untouched hold-out split
  3. 60k stratified candidate-development subset (5-fold CV)
  4. Baseline and SMOTE-ENN comparison for 8 classifiers
  5. Optuna/TPE optimisation (objective: mean CV average precision)
  6. Four-condition ablation, error-complementarity Level-0 selection
  7. Out-of-fold stacking with Logistic Regression meta-learner
  8. F1-optimal threshold chosen on development OOF predictions only
  9. Single evaluation on the hold-out set
 10. SHAP explanations (feature level and meta level)

Usage:
    python fraud_detection_pipeline.py --data data/creditcard.csv --out outputs
    python fraud_detection_pipeline.py --data data/creditcard.csv --fast   # quick smoke run
"""
import argparse
import json
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import shap  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from imblearn.combine import SMOTEENN  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from imblearn.under_sampling import EditedNearestNeighbours  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, auc, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
SMOTE_RATIO = 0.10
PERFORMANCE_FLOOR_RATIO = 0.90
N_JOBS = -1

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN",
               "SVM", "XGBoost", "LightGBM", "CatBoost"]
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM"}
TREE_MODELS = {"Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"}


# ----------------------------------------------------------------------------
# Models and pipelines
# ----------------------------------------------------------------------------
def make_estimator(name, params=None, svm_probability=False):
    p = {} if params is None else dict(params)
    if name == "Logistic Regression":
        base = dict(max_iter=2000, random_state=RANDOM_STATE, solver="liblinear")
        base.update(p)
        return LogisticRegression(**base)
    if name == "Decision Tree":
        base = dict(random_state=RANDOM_STATE)
        base.update(p)
        return DecisionTreeClassifier(**base)
    if name == "Random Forest":
        base = dict(random_state=RANDOM_STATE, n_jobs=N_JOBS)
        base.update(p)
        return RandomForestClassifier(**base)
    if name == "KNN":
        base = dict(n_jobs=N_JOBS)
        base.update(p)
        return KNeighborsClassifier(**base)
    if name == "SVM":
        base = dict(random_state=RANDOM_STATE, probability=svm_probability, cache_size=4000)
        base.update(p)
        return SVC(**base)
    if name == "XGBoost":
        base = dict(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=N_JOBS,
                    tree_method="hist")
        base.update(p)
        return XGBClassifier(**base)
    if name == "LightGBM":
        base = dict(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)
        base.update(p)
        return LGBMClassifier(**base)
    if name == "CatBoost":
        base = dict(random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
                    thread_count=-1)
        base.update(p)
        return CatBoostClassifier(**base)
    raise ValueError(name)


def make_smoteenn():
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE, k_neighbors=5)
    enn = EditedNearestNeighbours(n_neighbors=3)
    return SMOTEENN(smote=smote, enn=enn, random_state=RANDOM_STATE)


def make_pipeline(name, params=None, use_smoteenn=False, svm_probability=False):
    """Scaler (if needed) -> SMOTE-ENN (training folds only) -> classifier."""
    steps = []
    if name in SCALED_MODELS:
        steps.append(("scaler", StandardScaler()))
    if use_smoteenn:
        steps.append(("smoteenn", make_smoteenn()))
    steps.append(("model", make_estimator(name, params, svm_probability)))
    return ImbPipeline(steps)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def exact_pr_auc(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return auc(recall, precision)


def metric_row(y_true, scores, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "PR_AUC": exact_pr_auc(y_true, scores),
        "Average_Precision": average_precision_score(y_true, scores),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Threshold": float(threshold),
    }


def optimise_threshold_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    i = int(np.nanargmax(f1_values))
    return {"threshold": float(thresholds[i]), "precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1_values[i])}


# ----------------------------------------------------------------------------
# Optuna search spaces
# ----------------------------------------------------------------------------
def suggest_params(trial, name):
    if name == "Logistic Regression":
        return {"C": trial.suggest_float("C", 1e-2, 20, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "Decision Tree":
        return {"max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "Random Forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "KNN":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 21, step=2),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_categorical("p", [1, 2])}
    if name == "SVM":
        return {"C": trial.suggest_float("C", 0.1, 20, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "kernel": "rbf",
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 8)}
    if name == "LightGBM":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", -1, 14),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0)}
    if name == "CatBoost":
        return {"iterations": trial.suggest_int("iterations", 100, 350, step=50),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)}
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to creditcard.csv")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--fast", action="store_true", help="Small smoke run (2 trials, 20k subset)")
    args = ap.parse_args()

    n_trials = 2 if args.fast else 5
    model_dev_size = 20000 if args.fast else 60000
    shap_sample_size = 250 if args.fast else 1500

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # 1. Load, validate, deduplicate ----------------------------------------
    df_raw = pd.read_csv(args.data)
    expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    assert list(df_raw.columns) == expected or set(expected) <= set(df_raw.columns), "Unexpected columns"
    assert set(df_raw["Class"].unique()) <= {0, 1}, "Class must be binary"
    print(f"Raw: {df_raw.shape}, fraud={int(df_raw.Class.sum())}, "
          f"missing={int(df_raw.isna().sum().sum())}, duplicates={int(df_raw.duplicated().sum())}")
    df = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"After de-duplication: {df.shape}, fraud={int(df.Class.sum())}")

    # 2. Development / hold-out split ---------------------------------------
    X = df.drop(columns="Class")
    y = df["Class"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # 3. Candidate-development subset (drawn from development only) ----------
    n_work = min(model_dev_size, len(y_dev))
    if n_work < len(y_dev):
        idx, _ = train_test_split(np.arange(len(y_dev)), train_size=n_work,
                                  stratify=y_dev, random_state=RANDOM_STATE)
        X_work, y_work = X_dev.iloc[idx].copy(), y_dev.iloc[idx].copy()
    else:
        X_work, y_work = X_dev.copy(), y_dev.copy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Dev={len(y_dev)}  candidate subset={len(y_work)}  hold-out={len(y_test)}")

    def evaluate_cv(name, params=None, use_smoteenn=False):
        pipe = make_pipeline(name, params, use_smoteenn)
        res = cross_validate(
            pipe, X_work, y_work, cv=cv, n_jobs=1, error_score="raise",
            scoring={"ap": "average_precision", "roc_auc": "roc_auc", "precision": "precision",
                     "recall": "recall", "f1": "f1"})
        return {"Model": name, "SMOTE_ENN": use_smoteenn,
                "CV_AP_Mean": np.mean(res["test_ap"]), "CV_AP_SD": np.std(res["test_ap"], ddof=1),
                "CV_ROC_AUC_Mean": np.mean(res["test_roc_auc"]),
                "CV_Precision_Mean": np.mean(res["test_precision"]),
                "CV_Recall_Mean": np.mean(res["test_recall"]),
                "CV_F1_Mean": np.mean(res["test_f1"]),
                "Fit_Time_Mean_s": np.mean(res["fit_time"])}

    # 4. Baselines (default, default + SMOTE-ENN) ---------------------------
    baseline_rows = []
    for name in MODEL_NAMES:
        for use in (False, True):
            print("Baseline:", name, "| SMOTE-ENN" if use else "")
            row = evaluate_cv(name, None, use)
            row["Condition"] = "Default + SMOTE-ENN" if use else "Default"
            baseline_rows.append(row)
    baseline_results = pd.DataFrame(baseline_rows)
    baseline_results.to_csv(out / "baseline_cv_results.csv", index=False)

    # 5. Optuna / TPE (objective = mean CV average precision) ---------------
    best_params = {}

    def objective_factory(name, use):
        def objective(trial):
            pipe = make_pipeline(name, suggest_params(trial, name), use)
            r = cross_validate(pipe, X_work, y_work, cv=cv, scoring="average_precision",
                               n_jobs=1, error_score="raise")
            return float(np.mean(r["test_score"]))
        return objective

    for name in MODEL_NAMES:
        for use in (False, True):
            key = f"{name}__{'SMOTEENN' if use else 'NoResampling'}"
            print(f"Optimising {key} | {n_trials} trials")
            safe = re.sub(r"[^A-Za-z0-9_]+", "_", key)
            study = optuna.create_study(
                direction="maximize", sampler=TPESampler(seed=RANDOM_STATE),
                study_name=safe, storage=f"sqlite:///{out / ('optuna_' + safe + '.db')}",
                load_if_exists=True)
            remaining = max(0, n_trials - len(study.trials))
            if remaining:
                study.optimize(objective_factory(name, use), n_trials=remaining, gc_after_trial=True)
            best_params[key] = study.best_params
    (out / "optuna_best_parameters.json").write_text(json.dumps(best_params, indent=2))

    optimised_rows = []
    for name in MODEL_NAMES:
        for use in (False, True):
            key = f"{name}__{'SMOTEENN' if use else 'NoResampling'}"
            row = evaluate_cv(name, best_params[key], use)
            row["Condition"] = "Optimised + SMOTE-ENN" if use else "Optimised"
            row["Params"] = best_params[key]
            optimised_rows.append(row)
    optimised_results = pd.DataFrame(optimised_rows)

    # 6. Ablation table + best branch per classifier ------------------------
    ablation = pd.concat([baseline_results, optimised_results.drop(columns="Params")],
                         ignore_index=True, sort=False)
    ablation.to_csv(out / "four_condition_ablation.csv", index=False)

    best_specs, best_rows = {}, []
    for name in MODEL_NAMES:
        sub = optimised_results[optimised_results.Model == name]
        top = sub.loc[sub.CV_AP_Mean.idxmax()]
        best_specs[name] = {"params": top["Params"], "use_smoteenn": bool(top["SMOTE_ENN"]),
                            "cv_ap": float(top["CV_AP_Mean"])}
        best_rows.append(top.drop(labels="Params"))
    best_branch_table = pd.DataFrame(best_rows).reset_index(drop=True)
    best_branch_table.to_csv(out / "best_branch_per_classifier.csv", index=False)

    # Out-of-fold binary-error correlation (complementarity) ----------------
    oof_errors = {}
    for name in MODEL_NAMES:
        spec = best_specs[name]
        pipe = make_pipeline(name, spec["params"], spec["use_smoteenn"])
        method = "decision_function" if name == "SVM" else "predict_proba"
        preds = cross_val_predict(pipe, X_work, y_work, cv=cv, method=method, n_jobs=1)
        scores = preds[:, 1] if method == "predict_proba" else preds
        cut = 0.5 if method == "predict_proba" else 0.0
        oof_errors[name] = ((scores >= cut).astype(int) != y_work.to_numpy()).astype(int)
    error_corr = pd.DataFrame(oof_errors).corr()
    error_corr.to_csv(out / "oof_error_correlation.csv")

    # Level-0 selection: best AP first, then least error-correlated ---------
    ranked = best_branch_table.sort_values("CV_AP_Mean", ascending=False).reset_index(drop=True)
    best_ap = ranked.loc[0, "CV_AP_Mean"]
    eligible = ranked[ranked.CV_AP_Mean >= PERFORMANCE_FLOOR_RATIO * best_ap]["Model"].tolist()
    selected = [ranked.loc[0, "Model"]]
    while len(selected) < 3:
        remaining = [m for m in eligible if m not in selected] or \
                    [m for m in ranked["Model"] if m not in selected]
        scored = sorted(
            (np.mean([abs(error_corr.loc[c, s]) for s in selected]), -best_specs[c]["cv_ap"], c)
            for c in remaining)
        selected.append(scored[0][2])
    print("Selected Level-0 models:", selected)

    # 7. Out-of-fold stacking ------------------------------------------------
    stack_oof = pd.DataFrame(index=X_work.index)
    for name in selected:
        spec = best_specs[name]
        pipe = make_pipeline(name, spec["params"], spec["use_smoteenn"],
                             svm_probability=(name == "SVM"))
        stack_oof[name] = cross_val_predict(pipe, X_work, y_work, cv=cv,
                                            method="predict_proba", n_jobs=1)[:, 1]
    meta_model = LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)
    meta_oof = cross_val_predict(meta_model, stack_oof, y_work, cv=cv,
                                 method="predict_proba", n_jobs=1)[:, 1]
    meta_model.fit(stack_oof, y_work)
    pd.DataFrame({"Base_Model": selected, "Meta_Coefficient": meta_model.coef_[0]}) \
        .to_csv(out / "meta_learner_coefficients.csv", index=False)

    # 8. Threshold chosen on development OOF predictions only ---------------
    stack_thr = optimise_threshold_f1(y_work.to_numpy(), meta_oof)
    print("Frozen stack threshold:", stack_thr)

    # 9. Final hold-out evaluation (one shot) -------------------------------
    final_models, holdout_meta, rows = {}, pd.DataFrame(index=X_test.index), []
    for name in selected:
        spec = best_specs[name]
        pipe = make_pipeline(name, spec["params"], spec["use_smoteenn"],
                             svm_probability=(name == "SVM"))
        pipe.fit(X_dev, y_dev)  # full development partition
        final_models[name] = pipe
        prob = pipe.predict_proba(X_test)[:, 1]
        holdout_meta[name] = prob
        thr = optimise_threshold_f1(y_work.to_numpy(), stack_oof[name].to_numpy())["threshold"]
        for label, t in (("0.50", 0.5), ("Development-optimised", thr)):
            r = metric_row(y_test.to_numpy(), prob, t)
            r.update({"Model": name, "Operating_Point": label})
            rows.append(r)

    stack_prob = meta_model.predict_proba(holdout_meta[selected])[:, 1]
    for label, t in (("0.50", 0.5), ("Development-optimised", stack_thr["threshold"])):
        r = metric_row(y_test.to_numpy(), stack_prob, t)
        r.update({"Model": "Stacked Ensemble", "Operating_Point": label})
        rows.append(r)
    final_holdout = pd.DataFrame(rows)[[
        "Model", "Operating_Point", "Threshold", "PR_AUC", "Average_Precision", "ROC_AUC",
        "Precision", "Recall", "F1", "Accuracy", "TN", "FP", "FN", "TP"]]
    final_holdout.to_csv(out / "final_holdout_comparison.csv", index=False)
    print(final_holdout.to_string(index=False))

    # 10. SHAP --------------------------------------------------------------
    tree_selected = [m for m in selected if m in TREE_MODELS]
    shap_name = max(tree_selected, key=lambda m: best_specs[m]["cv_ap"]) if tree_selected else selected[0]
    shap_pipe = final_models[shap_name]
    X_shap = X_test.sample(n=min(shap_sample_size, len(X_test)), random_state=RANDOM_STATE)

    if shap_name in TREE_MODELS:
        explainer = shap.Explainer(shap_pipe.named_steps["model"], X_shap)
        exp = explainer(X_shap)
        if exp.values.ndim == 3:  # positive class
            exp = shap.Explanation(values=exp.values[:, :, 1], base_values=np.asarray(exp.base_values)[:, 1]
                                   if np.asarray(exp.base_values).ndim == 2 else exp.base_values,
                                   data=exp.data, feature_names=exp.feature_names)
    else:
        background = shap.sample(X_work, min(100, len(X_work)), random_state=RANDOM_STATE)
        X_shap = X_shap.iloc[:min(400, len(X_shap))]
        fn = lambda d: shap_pipe.predict_proba(pd.DataFrame(d, columns=X_work.columns))[:, 1]  # noqa: E731
        exp = shap.Explainer(fn, background)(X_shap)

    importance = pd.DataFrame({"Feature": X_shap.columns,
                               "Mean_Absolute_SHAP": np.abs(exp.values).mean(axis=0)}) \
        .sort_values("Mean_Absolute_SHAP", ascending=False)
    importance.to_csv(out / "shap_global_importance.csv", index=False)
    print(importance.head(10).to_string(index=False))

    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out / "shap_beeswarm.png", dpi=200)
    plt.close()
    plt.figure()
    shap.plots.bar(exp, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out / "shap_global_bar.png", dpi=200)
    plt.close()

    # Meta-level explainability (base-model probabilities -> stack) ---------
    meta_exp = shap.LinearExplainer(meta_model, stack_oof)(
        stack_oof.sample(n=min(2000, len(stack_oof)), random_state=RANDOM_STATE))
    plt.figure()
    shap.plots.bar(meta_exp, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_meta_level.png", dpi=200)
    plt.close()

    # Reproducibility log ---------------------------------------------------
    import sklearn, imblearn, xgboost, lightgbm, catboost  # noqa: E401
    (out / "reproducibility_log.json").write_text(json.dumps({
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
        "optuna": optuna.__version__, "shap": shap.__version__, "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
        "random_state": RANDOM_STATE, "cv_folds": N_SPLITS, "optuna_trials": n_trials,
        "candidate_subset_size": int(len(y_work)), "full_development_size": int(len(y_dev)),
        "smote_ratio": SMOTE_RATIO}, indent=2))
    print("Done. Outputs in", out.resolve())


if __name__ == "__main__":
    main()
