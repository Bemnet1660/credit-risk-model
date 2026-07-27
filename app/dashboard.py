"""
Interactive Credit Risk Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.credit_risk_model import CreditRiskPredictor, ModelConfig
import shap
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(file), '..'))

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


class Dashboard:
    def init(self):
        self.model = None
        self.data = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.explainer = None
        self.shap_values = None

    def load_model(self, data_path):
        """Load and train the model."""
        try:
            self.model = CreditRiskPredictor(ModelConfig())
            self.data = self.model.load_data(data_path)
            self.model.split_data(self.data)
            self.model.train()
            self.X_test = self.model.X_test
            self.y_test = self.model.y_test
            self.predictions = self.model.predict(self.X_test)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def compute_shap(self):
        """Compute SHAP values if not already done."""
        if self.explainer is None and self.model is not None and self.model._is_trained:
            with st.spinner("Computing SHAP explanations..."):
                self.explainer = shap.TreeExplainer(self.model.model)
                self.shap_values = self.explainer.shap_values(self.X_test.iloc[:100])  # Limit for speed
        return self.shap_values is not None

    def run(self):
        st.title("🏦 Credit Risk Analytics Dashboard")
        st.markdown("---")

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Controls")
            uploaded_file = st.file_uploader(
                "Upload Credit Data (CSV)",
                type=['csv']
            )

            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("temp_data.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if self.load_model("temp_data.csv"):
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model.")

            st.markdown("---")
            st.header("📖 About")
            st.markdown("""
                This dashboard analyzes credit risk using a Random Forest model.

                Features used:
                - Loan Amount
                - Credit Score
                - Income
                - Employment Length
                - Debt-to-Income Ratio

                Data format required:
                - Columns: loan_amount, credit_score, income,
                  employment_length, debt_to_income_ratio, target
                - target: 0 = no default, 1 = default
            """)

        # Main content
        if self.model is not None and self.model._is_trained:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "🎯 Accuracy",
                    f"{self.model.metrics.accuracy:.2%}",
                )
              with col2:
                st.metric(
                    "📈 ROC-AUC",
                    f"{self.model.metrics.roc_auc:.2%}",
                )
            with col3:
                st.metric(
                    "🎯 F1 Score",
                    f"{self.model.metrics.f1_score:.2%}",
                )
            with col4:
                st.metric(
                    "📊 Test Samples",
                    len(self.X_test),
                )

            st.markdown("---")

            # Feature Importance
            st.header("🔍 Feature Importance Analysis")
            importance = self.model.get_feature_importance()
            fig = px.bar(
                x=importance.values,
                y=importance.index,
                orientation='h',
                title="Global Feature Importance",
                labels={'x': 'Importance Score', 'y': 'Feature'},
                color=importance.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

            # SHAP Analysis
            st.header("🤖 Model Explainability with SHAP")
            if st.button("Generate SHAP Explanations"):
                if self.compute_shap():
                    # Summary plot
                    st.subheader("SHAP Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(
                        self.shap_values[1],  # for class 1 (default)
                        self.X_test.iloc[:100],
                        show=False
                    )
                    st.pyplot(fig)

                    # Force plot for first prediction
                    st.subheader("Individual Prediction Explanation")
                    selected_index = st.slider(
                        "Select sample index",
                        0, min(99, len(self.X_test)-1), 0
                    )
                    fig = shap.force_plot(
                        self.explainer.expected_value[1],
                        self.shap_values[1][selected_index:selected_index+1],
                        self.X_test.iloc[selected_index:selected_index+1],
                        show=False,
                        matplotlib=True
                    )
                    st.pyplot(fig)

            # Prediction Explorer
            st.header("🔮 Prediction Explorer")

            col1, col2 = st.columns(2)
            with col1:
                loan_amount = st.slider(
                    "Loan Amount ($)",
                    1000, 100000, 25000,
                    step=1000
                )
                credit_score = st.slider(
                    "Credit Score",
                    300, 850, 700,
                    step=5
                )
            with col2:
                income = st.slider(
                    "Annual Income ($)",
                    20000, 500000, 75000,
                    step=5000
                )
                employment_length = st.slider(
                    "Employment Length (years)",
                    0, 40, 5,
                    step=1
                )
            debt_to_income = st.slider(
                "Debt-to-Income Ratio",
                0.0, 0.6, 0.3,
                step=0.01
            )

            if st.button("Predict Risk"):
                input_data = pd.DataFrame({
                    'loan_amount': [loan_amount],
                    'credit_score': [credit_score],
                    'income': [income],
                    'employment_length': [employment_length],
                    'debt_to_income_ratio': [debt_to_income]
                })

                prediction = self.model.predict(input_data)
                probability = self.model.predict_proba(input_data)[0][1]

                if prediction[0] == 1:
                    st.error(f"🚨 High Credit Risk Detected! Probability: {probability:.2%}")
                else:
                    st.success(f"✅ Low Credit Risk Detected! Probability: {probability:.2%}")
                  # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    title={'text': "Risk Score"},
                    delta={'reference': 60},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Please upload a CSV file to get started!")
            st.markdown("""
                ### Sample Data Format
                Your CSV should include:
                - loan_amount: Float
                - credit_score: Integer
                - income: Float
                - employment_length: Integer
                - debt_to_income_ratio: Float
                - target: Integer (0 = no default, 1 = default)
            """)
            # Option to generate sample data
            if st.button("Generate Sample Data and Load"):
                from src.data_generator import generate_credit_data
                df = generate_credit_data(2000)
                df.to_csv("sample_data.csv", index=False)
                if self.load_model("sample_data.csv"):
                    st.success("✅ Sample data loaded and model trained!")
                    st.rerun()


if name == "main":
    dashboard = Dashboard()
    dashboard.run()
