import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering steps used during model training.
    """

    # Bill amount columns
    bill_cols = [
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'
    ]

    # Payment amount columns
    pay_cols = [
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    # Average bill amount
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)

    # Average payment amount
    df['AVG_PAY_AMT'] = df[pay_cols].mean(axis=1)

    # Credit utilization
    df['CREDIT_UTILIZATION'] = df['AVG_BILL_AMT'] / df['LIMIT_BAL']
    df['CREDIT_UTILIZATION'] = df['CREDIT_UTILIZATION'].replace([np.inf, -np.inf], 0)
    df['CREDIT_UTILIZATION'] = df['CREDIT_UTILIZATION'].fillna(0)

    return df
