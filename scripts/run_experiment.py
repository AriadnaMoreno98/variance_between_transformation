import sys
import time
import argparse
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from src.representations.lexical import TFIDFRepresentation
from src.selection.selectors import DMeansSelector
from src.search.search import HorizontalSearch
from src.mml.pipeline_v2 import MMLPipeline
from src.baselines.baseline_pipeline import BaselinePipeline
from src.evaluation.metrics import compute_classification_metrics
from src.transformations.transformations import (
    VarianceBetweenProjection,
    VarianceBetweenBalancedProjection,
    DiscriminativeProjection,
    PolarityProjection,
    WeightedProjection,
    SplitHalvesProjection,
    CentroidDistanceProjection,
)

DATA_PATH = PROJECT_ROOT / "data"

# Las 6 transformaciones estudiadas en el paper (variance_between es la
# propuesta nueva; el resto son las transformaciones MML con las que se compara),
# mas variance_between_balanced: variante exploratoria de variance_between con
# normalizacion por dispersion intra-clase estilo Cohen's d (ver Seccion 11 de
# notebooks/analysis.ipynb).
TRANSFORMATIONS = {
    "variance_between": VarianceBetweenProjection,
    "variance_between_balanced": VarianceBetweenBalancedProjection,
    "discriminative": DiscriminativeProjection,
    "polarity": PolarityProjection,
    "weighted": WeightedProjection,
    "split_halves": SplitHalvesProjection,
    "centroid_distance": CentroidDistanceProjection,
}

BASELINE_MODELS = ["logreg", "svm", "knn", "nb", "lr_balanced", "svm_balanced", "mlp"]

K_VALUES = [50, 100, 200]
MAX_WORKERS = 4


def build_validator(method, n_splits, n_repeats, random_state):
    if method == "kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if method == "repeated_kfold":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    raise ValueError(f"Método de validación no soportado: {method}")


def run_experiment(
    task_name,
    transformation_name,
    k,
    results_dir,
    validator_method="kfold",
    n_splits=5,
    n_repeats=3,
    random_state=42,
    run_baselines=False,
):
    """Corre (task, transformation, k) con TFIDF + DMeans + HorizontalSearch.
    Tolerante a interrupciones: si el CSV ya existe, se salta sin recalcular."""
    results_dir.mkdir(parents=True, exist_ok=True)

    base_filename = f"{task_name}_{transformation_name}_k{k}_{validator_method}"
    mml_path = results_dir / f"{base_filename}_mml.csv"
    baseline_path = results_dir / f"{base_filename}_baselines.csv"

    already_done = mml_path.exists() and (not run_baselines or baseline_path.exists())
    if already_done:
        return task_name, transformation_name, k, pd.read_csv(mml_path)["balanced_accuracy"].mean()

    task_path = DATA_PATH / f"{task_name}.csv"
    df = pd.read_csv(task_path)

    X = df["text"].values
    y = df["label"].values

    validator = build_validator(validator_method, n_splits, n_repeats, random_state)

    mml_rows = []
    baseline_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(validator.split(X, y)):
        X_train_text, X_test_text = X[train_idx].tolist(), X[test_idx].tolist()
        y_train, y_test = y[train_idx], y[test_idx]

        representation = TFIDFRepresentation()
        X_train_vec = representation.fit_transform(X_train_text)
        X_test_vec = representation.transform(X_test_text)

        mml_pipeline = MMLPipeline(
            selector=DMeansSelector(),
            transformation=TRANSFORMATIONS[transformation_name](),
            search=HorizontalSearch(),
            k=k,
        )

        start = time.time()
        y_pred = mml_pipeline.fit_predict(X_train_vec, y_train, X_test_vec)
        elapsed = time.time() - start

        metrics = compute_classification_metrics(y_test, y_pred)
        boundary = mml_pipeline.boundary_params()

        mml_rows.append({
            "task": task_name,
            "representation": "tfidf",
            "selector": "dmeans",
            "transformation": transformation_name,
            "search": "horizontal",
            "k": k,
            "validator": validator_method,
            "fold": fold_id,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "threshold": boundary.get("threshold"),
            "theta": boundary.get("theta"),
            "orientation": boundary.get("orientation"),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "runtime_seconds": elapsed,
        })

        if run_baselines:
            for model_name in BASELINE_MODELS:
                baseline_pipeline = BaselinePipeline(model_name=model_name)

                start = time.time()
                y_pred_base = baseline_pipeline.fit_predict(X_train_vec, y_train, X_test_vec)
                elapsed_base = time.time() - start

                metrics_base = compute_classification_metrics(y_test, y_pred_base)

                baseline_rows.append({
                    "task": task_name,
                    "representation": "tfidf",
                    "model": model_name,
                    "validator": validator_method,
                    "fold": fold_id,
                    "balanced_accuracy": metrics_base["balanced_accuracy"],
                    "f1": metrics_base["f1"],
                    "precision": metrics_base["precision"],
                    "recall": metrics_base["recall"],
                    "roc_auc": metrics_base["roc_auc"],
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "runtime_seconds": elapsed_base,
                })

    mml_results = pd.DataFrame(mml_rows)
    mml_results.to_csv(mml_path, index=False)

    if run_baselines:
        pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)

    return task_name, transformation_name, k, mml_results["balanced_accuracy"].mean()


def run_combination(task_name, transformation_name, k, results_dir, validator_method, n_splits, n_repeats, random_state, run_baselines):
    """Wrapper de modulo (no anidado) para que ProcessPoolExecutor pueda
    serializarlo y mandarlo a los procesos worker."""
    return run_experiment(
        task_name=task_name,
        transformation_name=transformation_name,
        k=k,
        results_dir=results_dir,
        validator_method=validator_method,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        run_baselines=run_baselines,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experimento del paper: corre las 6 transformaciones "
            "(variance_between + 5 de comparacion) x k en K_VALUES, "
            "sobre las tareas CTI-to-MITRE y ATT&CK Enterprise, en paralelo."
        )
    )
    parser.add_argument("--validator", default="kfold", choices=["kfold", "repeated_kfold"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--k_values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--data_dir", default=str(DATA_PATH), help="Directorio con los CSVs de tareas (text, label)")
    parser.add_argument("--results_dir", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument(
        "--task_prefix",
        nargs="+",
        default=["cti_", "attck_"],
        help="Prefijos de tareas a incluir (por defecto: CTI-to-MITRE y ATT&CK Enterprise)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    all_task_names = sorted(p.stem for p in data_dir.glob("*.csv"))
    task_names = [t for t in all_task_names if any(t.startswith(prefix) for prefix in args.task_prefix)]

    transformation_names = list(TRANSFORMATIONS.keys())
    combinations = list(itertools.product(task_names, transformation_names, args.k_values))
    total = len(combinations)

    print(f"Directorio de datos: {data_dir}")
    print(f"Tareas: {len(task_names)}/{len(all_task_names)} (prefijos: {args.task_prefix})")
    print(f"Transformaciones: {transformation_names}")
    print(f"k: {args.k_values} | Validador: {args.validator} (n_splits={args.n_splits})")
    print(f"Baselines: {args.run_baselines} | Total de combinaciones: {total} | workers: {args.max_workers}")
    print()

    completed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_combination,
                task_name,
                transformation_name,
                k,
                results_dir,
                args.validator,
                args.n_splits,
                args.n_repeats,
                args.random_state,
                args.run_baselines,
            ): (task_name, transformation_name, k)
            for task_name, transformation_name, k in combinations
        }

        for future in as_completed(futures):
            task_name, transformation_name, k = futures[future]
            completed += 1

            try:
                _, _, _, mean_ba = future.result()
                print(f"[{completed}/{total}] {task_name} | {transformation_name} | k={k} | BA={mean_ba:.4f}")
            except Exception as e:
                print(f"[{completed}/{total}] {task_name} | {transformation_name} | k={k} | ERROR: {e}")

    print()
    print(f"Listo: {total} combinaciones procesadas ({len(task_names)} tareas).")


if __name__ == "__main__":
    main()
