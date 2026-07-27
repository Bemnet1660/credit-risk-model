"""
Synthetic data generator for credit risk modeling.
"""

import numpy as np
import pandas as pd


def generate_credit_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit risk data.

    Args:
        n_samples: Number of samples
        random_state: Random seed

    Returns:
        DataFrame with features and target
    """
    np.random.seed(random_state)

    # Features
    loan_amount = np.random.uniform(1000, 50000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)

    income = np.random.uniform(20000, 200000, n_samples)
    employment_length = np.random.randint(0, 40, n_samples)
    debt_to_income = np.random.uniform(0, 0.6, n_samples)

    # Create target based on a logistic function of features
    logit = (
        -0.5
        + 0.3 * (loan_amount / 10000)
        - 0.02 * credit_score
        + 0.01 * (income / 10000)
        + 0.1 * debt_to_income
        + 0.05 * employment_length
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = np.random.binomial(1, prob_default)

    data = pd.DataFrame({
        'loan_amount': loan_amount,
        'credit_score': credit_score,
        'income': income,
        'employment_length': employment_length,
        'debt_to_income_ratio': debt_to_income,
        'target': target
    })

    return data


if name == "main":
    # Generate and save sample data
    df = generate_credit_data(5000)
    df.to_csv('data/credit_data.csv', index=False)
    print("Sample data saved to data/credit_data.csv")
