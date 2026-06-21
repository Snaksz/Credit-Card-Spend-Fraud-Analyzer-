import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def create_fraud_pattern_heatmap(data: pd.DataFrame, output_path: str) -> Path:
    if "transaction_hour" not in data.columns or "is_fraud" not in data.columns:
        raise ValueError("Data must include transaction_hour and is_fraud columns")
    pivot = data.pivot_table(index="transaction_hour", columns="is_fraud", values="transaction_id", aggfunc="count", fill_value=0)
    figure = plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def create_spend_trend_line(data: pd.DataFrame, output_path: str) -> Path:
    if "transaction_date" not in data.columns or "transaction_amount" not in data.columns:
        raise ValueError("Data must include transaction_date and transaction_amount columns")
    trend = data.copy()
    trend["transaction_date"] = pd.to_datetime(trend["transaction_date"])
    trend = trend.groupby(pd.Grouper(key="transaction_date", freq="M"), as_index=False)["transaction_amount"].sum()

    figure = plt.figure(figsize=(10, 5))
    sns.lineplot(data=trend, x="transaction_date", y="transaction_amount")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--fraud-output", default="visuals/output/fraud_pattern_heatmap.png")
    parser.add_argument("--spend-output", default="visuals/output/spend_trend_line.png")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    create_fraud_pattern_heatmap(data, args.fraud_output)
    create_spend_trend_line(data, args.spend_output)


if __name__ == "__main__":
    main()
