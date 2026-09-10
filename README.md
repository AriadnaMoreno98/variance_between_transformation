# Between-Class Variance Projection: A New Geometric Transformation for Minimalist Machine Learning

This repository contains the source code and experimental pipeline associated with the paper:

> **Between-Class Variance Projection: A New Geometric Transformation for Minimalist Machine Learning**
> A. Moreno-Román, M. Moreno-Espino, C. Yáñez-Márquez

## Overview

This codebase implements and evaluates `variance_between`
(`VarianceBetweenProjection`), a new two-dimensional transformation for
the Minimalist Machine Learning (MML) architecture, applied to binary
text classification of cybersecurity data. The pipeline reduces a TF-IDF
representation to an interpretable 2D projection `(z1, z2)` and
classifies with a horizontal decision boundary optimized in that space.
`variance_between` is compared against five existing MML transformations
and seven classical baselines, across two families of binary
classification tasks: CTI-to-MITRE (does a CTI report mention a given
ATT&CK tactic?) and ATT&CK Enterprise (does a technique belong to a given
tactic/platform, or is it a sub-technique?).

## Repository structure

- `data/` — the 25 task CSVs (`text`, `label` columns): 15 CTI-to-MITRE
  tasks (`cti_is_*.csv`) and 10 ATT&CK Enterprise tasks (`attck_*.csv`)
- `src/representations/lexical.py` — TF-IDF text representation
  (`TFIDFRepresentation`)
- `src/selection/selectors.py` — dMeans feature selection
  (`DMeansSelector`), scoring features by the difference of per-class
  means (the only selection method reported in the paper)
- `src/transformations/transformations.py` — the six 2D transformations
  compared in the paper: `variance_between` (proposed), `discriminative`,
  `polarity`, `weighted`, `split_halves`, `centroid_distance`
- `src/search/search.py` — horizontal decision boundary
  (`HorizontalSearch`) via exhaustive threshold search maximizing
  Balanced Accuracy on the training fold
- `src/mml/pipeline_v2.py` — `MMLPipeline`, chaining representation →
  selection → transformation → search
- `src/baselines/baseline_pipeline.py` — baseline classifiers:
  Logistic Regression, SVM (`LinearSVC`), KNN, Naive Bayes, and MLP, each
  also run with `class_weight="balanced"` as a separate baseline for
  Logistic Regression and SVM
- `src/evaluation/metrics.py` — evaluation metrics: Accuracy, Precision,
  Recall, F1, Balanced Accuracy, ROC AUC, G-mean, MCC, and PR-AUC
- `scripts/run_experiment.py` — runs the full sweep (6 transformations ×
  k ∈ {50, 100, 200} × 25 tasks, 5-fold stratified cross-validation),
  producing one CSV per combination

## Reproducing the experiments

1. Install dependencies:
```bash
   pip install -r requirements.txt
```

2. The datasets are already included in `data/` — no external download is
   required.

3. Run the experiment sweep:
```bash
   python scripts/run_experiment.py
```
   This produces one CSV per `(task, transformation, k)` combination in
   `results/runs/`. It is resumable: a combination already computed is
   skipped on a re-run.

The pipeline reproduces the paper's reported MML results: feature
selection via **dMeans** and the **six 2D transformations** listed above,
swept across `k` ∈ {50, 100, 200} and evaluated with 5-fold stratified
cross-validation, and compared against seven classical baselines
(Logistic Regression, SVM, KNN, Naive Bayes, MLP, and the balanced
variants of Logistic Regression and SVM). These are the only
feature-selection and transformation methods implemented in
`src/selection/selectors.py` and `src/transformations/transformations.py`
— other MML transformation variants explored during development (e.g.
`skewness`, `entropy_local`, `cosine_prototype`, `weighted_contrast`,
`soft_ratio`, `peak_activation`, `quantile`) were removed, not just
disabled, since they are not part of the paper's reported experiments.

## Random seeds

All stochastic components use `random_state=42`: the `StratifiedKFold` /
`RepeatedStratifiedKFold` cross-validation splits, and the baseline
classifiers with a random component (Logistic Regression, SVM, MLP).
TF-IDF, dMeans feature selection, and all six MML transformations are
deterministic and need no seed.

## Requirements

See `requirements.txt` for the required dependencies.
