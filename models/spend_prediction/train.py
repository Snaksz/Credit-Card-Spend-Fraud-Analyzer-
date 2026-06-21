import argparse
from typing import Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "monthly_spend"


def _metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(feature_data: pd.DataFrame, random_state: int = 42) -> Dict[str, float]:
    if TARGET_COLUMN not in feature_data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    x = feature_data.drop(columns=[TARGET_COLUMN])
    y = feature_data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    categorical_columns = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ]
    )

    benchmark_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    challenger_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=250, random_state=random_state)),
        ]
    )

    benchmark_pipeline.fit(x_train, y_train)
    challenger_pipeline.fit(x_train, y_train)

    benchmark_predictions = benchmark_pipeline.predict(x_test)
    challenger_predictions = challenger_pipeline.predict(x_test)

    benchmark_metrics = _metrics(y_test, benchmark_predictions)
    challenger_metrics = _metrics(y_test, challenger_predictions)

    improvement = 100.0 * (benchmark_metrics["rmse"] - challenger_metrics["rmse"]) / max(benchmark_metrics["rmse"], 1e-9)

    return {
        "benchmark_rmse": benchmark_metrics["rmse"],
        "challenger_rmse": challenger_metrics["rmse"],
        "challenger_mae": challenger_metrics["mae"],
        "challenger_r2": challenger_metrics["r2"],
        "rmse_improvement_percent": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    args = parser.parse_args()
    feature_data = pd.read_csv(args.features)
    metrics = train_and_evaluate(feature_data)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
