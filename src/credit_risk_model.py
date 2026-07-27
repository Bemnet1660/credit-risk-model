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
