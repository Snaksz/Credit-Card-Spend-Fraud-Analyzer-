import argparse
import math
from typing import Dict

import numpy as np
import pandas as pd


def hypothesis_test(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    fraud_scores = scores[labels == 1]
    non_fraud_scores = scores[labels == 0]
    if len(fraud_scores) < 2 or len(non_fraud_scores) < 2:
        raise ValueError("At least two fraud and two non-fraud samples are required")

    fraud_mean = float(np.mean(fraud_scores))
    non_fraud_mean = float(np.mean(non_fraud_scores))
    fraud_var = float(np.var(fraud_scores, ddof=1))
    non_fraud_var = float(np.var(non_fraud_scores, ddof=1))

    standard_error = math.sqrt((fraud_var / len(fraud_scores)) + (non_fraud_var / len(non_fraud_scores)))
    z_score = (fraud_mean - non_fraud_mean) / standard_error
    p_value = math.erfc(abs(z_score) / math.sqrt(2))

    return {
        "fraud_mean": fraud_mean,
        "non_fraud_mean": non_fraud_mean,
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def _metrics_from_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())

    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "false_positive_count": float(false_positive),
    }


def tune_threshold(scores: np.ndarray, labels: np.ndarray, min_recall: float = 0.60) -> Dict[str, float]:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 200)
    candidates = [_metrics_from_threshold(scores, labels, threshold) for threshold in thresholds]

    valid_candidates = [candidate for candidate in candidates if candidate["recall"] >= min_recall]
    if not valid_candidates:
        valid_candidates = candidates

    valid_candidates.sort(
        key=lambda candidate: (
            candidate["false_positive_rate"],
            -candidate["precision"],
            -candidate["recall"],
        )
    )
    return valid_candidates[0]


def run(csv_path: str, score_column: str, label_column: str, min_recall: float) -> Dict[str, Dict[str, float]]:
    data = pd.read_csv(csv_path)
    if score_column not in data.columns or label_column not in data.columns:
        raise ValueError("Input file is missing required score or label columns")

    labels = data[label_column].astype(int).to_numpy()
    scores = data[score_column].astype(float).to_numpy()

    return {
        "hypothesis_test": hypothesis_test(scores, labels),
        "best_threshold": tune_threshold(scores, labels, min_recall=min_recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--score-column", default="fraud_score")
    parser.add_argument("--label-column", default="is_fraud")
    parser.add_argument("--min-recall", type=float, default=0.60)
    args = parser.parse_args()

    results = run(args.input, args.score_column, args.label_column, args.min_recall)
    for section, values in results.items():
        print(section)
        for key, value in values.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
