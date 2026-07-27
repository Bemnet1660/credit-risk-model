"""
Credit Risk Prediction Module
Author: Your Name
Date: 2026
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


@dataclass
class ModelConfig:
    """Configuration for the credit risk model."""
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    test_size: float = 0.2


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class CreditRiskPredictor:
    """A production-ready credit risk prediction model."""

    def init(self, config: Optional[ModelConfig] = None):
        """
        Initialize the credit risk predictor.

        Args:
            config: Model configuration parameters
        """
        self.config = config or ModelConfig()
        self.model = None
        self._is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the dataset.

        Args:
            file_path: Path to the data file

        Returns:
            pandas DataFrame with cleaned data

        Raises:
            ValueError: If data validation fails
        """
        data = pd.read_csv(file_path)

        # Validate required columns
        required_columns = {'loan_amount', 'credit_score', 'income', 'target'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        return self._clean_data(data)

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        # Drop rows with missing target
        data = data.dropna(subset=['target'])

        # Handle missing values for features (simple imputation with median)
        for col in data.columns:
            if col != 'target':
                data[col].fillna(data[col].median(), inplace=True)

        # Remove outliers using IQR method for numerical columns
        num_cols = data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col != 'target':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

        return data

    def split_data(self, data: pd.DataFrame) -> None:
        """
        Split data into train/test sets.

        Args:
            data: Full dataset with target column
        """
        X = data.drop('target', axis=1)
        y = data['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

    def train(self, X: pd.DataFrame = None, y: pd.Series = None) -> None:
        """
        Train the model. If X and y are provided, use them; otherwise use the internal train split.

        Args:
            X: Feature matrix (optional)
            y: Target variable (optional)

        Raises:
            ValueError: If no data is available
        """
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
             X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
        elif self.X_train is None:
            raise ValueError("No training data available. Call split_data() or provide X,y.")

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced'
        )
        self.model.fit(self.X_train, self.y_train)

        # Evaluate
        self.metrics = self._evaluate_model()
        self._is_trained = True

    def _evaluate_model(self) -> ModelMetrics:
        """Evaluate model performance on test set."""
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        return ModelMetrics(
            accuracy=accuracy_score(self.y_test, y_pred),
            precision=report['weighted avg']['precision'],
            recall=report['weighted avg']['recall'],
            f1_score=report['weighted avg']['f1-score'],
            roc_auc=roc_auc_score(self.y_test, y_prob)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix for prediction

        Returns:
            Array of predictions

        Raises:
            RuntimeError: If model hasn't been trained
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.

        Returns:
            pandas Series with feature importance
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_
        ).sort_values(ascending=False)
