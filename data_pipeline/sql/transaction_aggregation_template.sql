WITH base_transactions AS (
    SELECT
        transaction_id,
        account_id,
        transaction_amount,
        transaction_date,
        is_fraud
    FROM {{source_schema}}.transactions
    WHERE transaction_date BETWEEN {{start_date}} AND {{end_date}}
),
account_aggregation AS (
    SELECT
        account_id,
        COUNT(*) AS transaction_count,
        SUM(transaction_amount) AS total_spend,
        AVG(transaction_amount) AS average_transaction_amount,
        MAX(transaction_amount) AS max_transaction_amount,
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transaction_count
    FROM base_transactions
    GROUP BY account_id
)
SELECT *
FROM account_aggregation;
