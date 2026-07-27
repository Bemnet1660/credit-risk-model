"""
Main entry point for training and evaluation.
"""

import argparse
import pandas as pd
from credit_risk_model import CreditRiskPredictor, ModelConfig
from data_generator import generate_credit_data


def main():
    parser = argparse.ArgumentParser(description='Credit Risk Model Training')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to generate')
    args = parser.parse_args()

    if args.generate:
        print(f"Generating {args.samples} samples...")
        df = generate_credit_data(args.samples)
        df.to_csv('credit_data.csv', index=False)
        data_path = 'credit_data.csv'
    elif args.data:
        data_path = args.data
    else:
        print("Please provide --data or --generate")
        return

    # Load and train
    print(f"Loading data from {data_path}")
    model = CreditRiskPredictor(ModelConfig())
    data = model.load_data(data_path)
    model.split_data(data)
    print("Training model...")
    model.train()
    print("Model trained successfully!")

    # Print metrics
    print("\n--- Model Performance ---")
    print(f"Accuracy: {model.metrics.accuracy:.4f}")
    print(f"ROC-AUC:   {model.metrics.roc_auc:.4f}")
    print(f"F1-Score:  {model.metrics.f1_score:.4f}")

    # Feature importance
    importance = model.get_feature_importance()
    print("\n--- Feature Importance ---")
    for feat, imp in importance.items():
        print(f"{feat}: {imp:.4f}")


if name == "main":
    main()
