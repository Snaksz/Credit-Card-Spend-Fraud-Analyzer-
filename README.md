# Credit Card Spend & Fraud Analyzer

Production-oriented data science project structure for spend prediction and fraud detection workflows.

## Objective
Build a modular analytics pipeline that transforms transaction-level data into account-level features, evaluates a spend prediction challenger model against a benchmark, and tunes fraud detection thresholds with a focus on minimizing false positives.

## Key Metrics
- Benchmark alignment target: match or exceed a 4% benchmark performance threshold for spend prediction.
- Fraud operations target: reduce false positives by 18% while preserving fraud recall.

## Repository Structure
- `/data_pipeline/feature_engineering.py`: Pandas feature engineering pipeline from transactions to account-level metrics.
- `/data_pipeline/sql/transaction_aggregation_template.sql`: SQL extraction and aggregation template.
- `/models/spend_prediction/train.py`: scikit-learn benchmark and challenger regression pipeline.
- `/models/spend_prediction/training_and_evaluation.ipynb`: notebook for training and comparative evaluation.
- `/models/fraud_detection/threshold_tuning.py`: hypothesis testing and threshold tuning workflow.
- `/visuals/plot_templates.py`: Seaborn-based proof-of-concept templates for fraud patterns and spend trends.

## How to Run
1. Generate account features from transaction data:
   ```bash
   python data_pipeline/feature_engineering.py --input data/transactions.csv --output data/output/account_features.csv
   ```
2. Train and evaluate spend prediction model:
   ```bash
   python models/spend_prediction/train.py --features data/output/account_features.csv
   ```
3. Run fraud threshold tuning:
   ```bash
   python models/fraud_detection/threshold_tuning.py --input data/transactions_with_scores.csv --score-column fraud_score --label-column is_fraud --min-recall 0.60
   ```
4. Generate visualization templates:
   ```bash
   python visuals/plot_templates.py --input data/transactions.csv
   ```

## Suggested Data Columns
- Transaction pipeline input: `transaction_id`, `account_id`, `transaction_amount`, `transaction_date`, `is_fraud`
- Fraud tuning input extras: `fraud_score`
- Visualization input extras: `transaction_hour`
