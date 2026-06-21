import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["transaction_id", "account_id", "transaction_amount", "transaction_date", "is_fraud"]


def load_transactions(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")
    return data.dropna(subset=["transaction_date"])


def build_account_level_features(transactions: pd.DataFrame) -> pd.DataFrame:
    grouped = transactions.groupby("account_id", as_index=False).agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("transaction_amount", "sum"),
        average_transaction_amount=("transaction_amount", "mean"),
        max_transaction_amount=("transaction_amount", "max"),
        fraud_transaction_count=("is_fraud", "sum"),
        first_transaction_date=("transaction_date", "min"),
        last_transaction_date=("transaction_date", "max"),
    )
    grouped["fraud_rate"] = grouped["fraud_transaction_count"] / grouped["transaction_count"].clip(lower=1)
    grouped["active_days"] = (grouped["last_transaction_date"] - grouped["first_transaction_date"]).dt.days + 1
    grouped["monthly_spend"] = grouped["total_spend"] / (grouped["active_days"] / 30.0).clip(lower=1)
    return grouped


def run_pipeline(input_csv: str, output_csv: str) -> Path:
    transactions = load_transactions(input_csv)
    features = build_account_level_features(transactions)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
