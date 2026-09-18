#!/usr/bin/env python3
"""Filtered logistic-regression analysis of the chr22 proteomics data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(data_dir / "sample_metadata.csv").set_index("sample_id")
    metadata["site"] = metadata.index.to_series().str.split("_", n=1).str[0].str.lower()
    matrices = []
    for path in sorted(data_dir.glob("site*_proteomics_log2.csv")):
        matrix = pd.read_csv(path).set_index("protein_id")
        matrices.append(matrix.drop(columns=["gene_symbol"]).T)
    expression = pd.concat(matrices, axis=0)
    expression.index.name = "sample_id"
    expression = expression.loc[metadata.index]
    return expression, metadata


def select_proteins(
    x_train: pd.DataFrame,
    min_detection: float,
    min_median: float,
    min_variance: float,
    max_correlation: float,
) -> tuple[list[str], dict[str, int]]:
    detection = x_train.notna().mean()
    median = x_train.median()
    variance = x_train.var()
    keep = (detection >= min_detection) & (median >= min_median) & (variance >= min_variance)
    candidates = list(x_train.columns[keep])

    # Greedy pruning makes the result deterministic and retains the more
    # variable member of highly correlated pairs.
    ordered = sorted(candidates, key=lambda col: (-variance[col], col))
    kept: list[str] = []
    correlation = x_train[ordered].corr().abs()
    for protein in ordered:
        if not kept or correlation.loc[protein, kept].max() <= max_correlation:
            kept.append(protein)
    counts = {
        "input": x_train.shape[1],
        "after_detection_median_variance": len(candidates),
        "after_correlation": len(kept),
    }
    return kept, counts


def make_pipeline(proteins: list[str], metadata_columns: list[str]) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [("proteins", numeric, proteins), ("metadata", categorical, metadata_columns)],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("proteomics/synthetic_proteomics_chr22"))
    parser.add_argument("--output-dir", type=Path, default=Path("proteomics/logistic_regression_results"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-detection", type=float, default=0.8)
    parser.add_argument("--min-median", type=float, default=6.0)
    parser.add_argument("--min-variance", type=float, default=0.01)
    parser.add_argument("--max-correlation", type=float, default=0.8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expression, metadata = load_data(args.data_dir)
    labels = metadata["phenotype"].astype(int)
    strata = metadata["site"].astype(str) + "_" + labels.astype(str)
    train_ids, test_ids = train_test_split(
        expression.index,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=strata,
    )
    x_train = expression.loc[train_ids]
    x_test = expression.loc[test_ids]
    y_train = labels.loc[train_ids].to_numpy()
    y_test = labels.loc[test_ids].to_numpy()
    proteins, filter_counts = select_proteins(
        x_train, args.min_detection, args.min_median, args.min_variance, args.max_correlation
    )

    feature_train = x_train[proteins].join(metadata.loc[train_ids, ["age", "sex", "site"]])
    feature_test = x_test[proteins].join(metadata.loc[test_ids, ["age", "sex", "site"]])
    model = make_pipeline(proteins, ["age", "sex", "site"])
    model.fit(feature_train, y_train)
    probability = model.predict_proba(feature_test)[:, 1]
    test_metrics = metrics(y_test, probability)
    cm = confusion_matrix(y_test, probability >= 0.5)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        args.output_dir / "test_confusion_matrix.csv"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_validate(
        make_pipeline(proteins, ["age", "sex", "site"]),
        expression.loc[train_ids, proteins].join(metadata.loc[train_ids, ["age", "sex", "site"]]),
        y_train,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
    )
    cv_summary = pd.DataFrame(
        {
            "metric": ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"],
            "mean": [cv_scores[f"test_{m}"].mean() for m in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]],
            "std": [cv_scores[f"test_{m}"].std(ddof=1) for m in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]],
        }
    )
    cv_summary.to_csv(args.output_dir / "cross_validation_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    pd.DataFrame({"sample_id": test_ids, "phenotype": y_test, "probability": probability}).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame({"protein_id": proteins}).to_csv(args.output_dir / "selected_proteins.csv", index=False)

    coefficients = model.named_steps["model"].coef_[0]
    names = model.named_steps["features"].get_feature_names_out()
    pd.DataFrame({"feature": names, "coefficient": coefficients}).sort_values(
        "coefficient", key=np.abs, ascending=False
    ).to_csv(args.output_dir / "model_coefficients.csv", index=False)
    summary = {
        "n_samples": len(expression),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "class_counts": labels.value_counts().sort_index().to_dict(),
        "filter_thresholds": {
            "min_detection": args.min_detection,
            "min_median_log2_expression": args.min_median,
            "min_variance": args.min_variance,
            "max_absolute_correlation": args.max_correlation,
        },
        "filter_counts": filter_counts,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted phenotype", ylabel="True phenotype", title="Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)

    top = pd.DataFrame({"feature": names, "coefficient": coefficients})
    top["abs_coefficient"] = top["coefficient"].abs()
    top = top.nlargest(20, "abs_coefficient").sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["coefficient"], color=np.where(top["coefficient"] > 0, "#d55e00", "#0072b2"))
    ax.set(xlabel="Standardized logistic-regression coefficient", ylabel="", title="Top model features")
    fig.tight_layout()
    fig.savefig(args.output_dir / "top_model_features.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(cv_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
