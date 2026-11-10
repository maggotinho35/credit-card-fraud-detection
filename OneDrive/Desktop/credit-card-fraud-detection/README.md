# Explainable Credit Card Fraud Detection: Adaptive, Hyperparameter-Optimised Hybrid Ensemble

MSc Data Science thesis project, IU International University of Applied Sciences (Berlin), 2026.
Grade for the thesis: 1.7.

An end-to-end, leakage-conscious machine-learning pipeline for detecting fraud in an extremely
imbalanced dataset (0.17% fraud). It compares eight classifiers under four conditions, builds a
stacked ensemble from *complementary* models, tunes the decision threshold, and explains the
final model with SHAP. The central finding is a deliberately honest one: **the stacked ensemble
did not beat its best single model on the untouched hold-out set, so the simpler CatBoost model
was selected.**

## Highlights

- **Leakage control:** duplicates removed before splitting; 80:20 stratified split with an untouched hold-out set; scaling and SMOTE-ENN fitted on training folds only; thresholds chosen on development out-of-fold predictions only.
- **Eight classifiers:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, LightGBM, CatBoost.
- **Four-condition ablation:** default, default + SMOTE-ENN, Optuna-optimised, optimised + SMOTE-ENN.
- **Bayesian optimisation:** Optuna with the TPE sampler, objective = mean 5-fold CV average precision (not accuracy).
- **Complementarity-based stacking:** Level-0 models (CatBoost, SVM, Logistic Regression) chosen by out-of-fold error correlation, not just by rank; Logistic Regression meta-learner trained on out-of-fold probabilities.
- **Explainability:** SHAP at feature level (CatBoost) and at meta level (base-model probabilities).

## Data

[Credit Card Fraud Detection dataset (ULB / Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 transactions over two days, 492 frauds. Features `V1`-`V28` are anonymised PCA components,
plus `Time`, `Amount` and `Class`. After removing 1,081 exact duplicates: 283,726 transactions
(473 fraud). Place `creditcard.csv` in a `data/` folder (it is git-ignored).

## Results (untouched hold-out: 56,746 transactions, 95 frauds)

| Model | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| **CatBoost** | dev-optimised | **0.8148** | **0.8916** | 0.7789 | **0.8315** | 9 | 21 |
| Stacked ensemble | dev-optimised | 0.8030 | 0.8605 | 0.7789 | 0.8177 | 12 | 21 |
| SVM | dev-optimised | 0.7647 | 0.9420 | 0.6842 | 0.7927 | 4 | 30 |
| Logistic Regression | dev-optimised | 0.6942 | 0.8434 | 0.7368 | 0.7865 | 13 | 25 |

Other findings:

- SMOTE-ENN raised recall for all eight optimised classifiers but lowered precision for all of them; it improved average precision only for CatBoost, LightGBM, Random Forest and XGBoost.
- Best development result: CatBoost + SMOTE-ENN, CV average precision 0.8319.
- SHAP: `V14`, `V4`, `V12`, `V1` and `V11` were the most influential features. They are PCA components, so this is model attribution, not a causal or business explanation.

## Reproduce

```bash
git clone https://github.com/maggotinho35/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# quick smoke test (2 Optuna trials, 20k subset)
python fraud_detection_pipeline.py --data data/creditcard.csv --out outputs --fast

# full thesis configuration (5 trials per classifier/branch, 60k subset)
python fraud_detection_pipeline.py --data data/creditcard.csv --out outputs
```

Outputs (CSV tables, Optuna studies, SHAP plots, version log) are written to `outputs/`.
The random seed is fixed at 42. Exact numbers can vary slightly with library versions; the log in
`outputs/reproducibility_log.json` records the versions used.

## Limitations

One public benchmark covering two days; anonymised features; small Optuna budget (5 trials) and a
60k development subset for computational reasons; threshold chosen by F1 rather than business cost.
Future work: cost-sensitive thresholds, probability calibration, temporal validation, concept-drift
monitoring, and more recent datasets.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Optuna, XGBoost, LightGBM, CatBoost, SHAP, Matplotlib.

## Author

Ebenezer Arthur, MSc Data Science. [LinkedIn](https://linkedin.com/in/ebenezer-arthur)
