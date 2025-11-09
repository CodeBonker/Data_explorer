import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_model.model_selector import ModelSelector
from ml_model.auto_trainer import AutoTrainer
from ml_model.meta_recommender import MetaFeatureExtractor, MetaLogger, MetaRecommender

st.set_page_config(page_title="AI Data Explorer", layout="wide")
st.title("AI Data Explorer")
st.caption("Upload your dataset, explore interactively, and train models automatically.")

# Helper functions


def plot_correlation(df):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        st.warning("Not enough numeric data to plot correlation heatmap.")
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def plot_feature_distribution(df, column, target_col=None):
    if column not in df.columns:
        st.error(f"Column '{column}' not found in dataset.")
        return

    # Categorical features
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        fig = px.bar(
            counts,
            x=column,
            y='count',
            title=f"Distribution of {column}",
            text='count'
        )
        st.plotly_chart(fig, use_container_width=True)
    # Numeric features
    else:
        if target_col and target_col in df.columns:
            try:
                fig = px.histogram(df, x=column, color=target_col, nbins=30, title=f"Distribution of {column} by {target_col}", marginal="box")
            except Exception:
                fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
        else:
            fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

# Initialize session state

if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = None
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False
if "leaderboard" not in st.session_state:
    st.session_state["leaderboard"] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "target" not in st.session_state:
    st.session_state["target"] = None


tab1, tab2, tab3 = st.tabs(["EDA & Visualization", "Model Training", "Meta Learning"])

# Tab 1: EDA

with tab1:
    st.header("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            st.session_state["dataframe"] = df
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.write("Shape:", df.shape)
            st.write("Missing values (total):", int(df.isnull().sum().sum()))

            with st.expander("Statistical Summary"):
                try:
                    st.dataframe(df.describe(include="all").T)
                except Exception:
                    st.write("Could not compute full describe table.")

            with st.expander("Correlation Heatmap (numeric only)"):
                plot_correlation(df)

            cols = list(df.columns)
            if cols:
                selected_col = st.selectbox("Select a feature to visualize", cols, index=0)
                target_col = st.selectbox("Select target column (optional)", [None] + cols)
                if st.button("Plot Feature Distribution"):
                    plot_feature_distribution(df, selected_col, target_col if target_col else None)

            st.session_state["analyzed"] = True
        else:
            st.session_state["analyzed"] = False


# Tab 2: Model Training

with tab2:
    st.header("AutoML Model Training")
    if st.session_state["dataframe"] is None:
        st.warning("Please upload a dataset in the EDA tab first.")
    else:
        df = st.session_state["dataframe"]
        cols = list(df.columns)
        # Prevent user from selecting invalid target
        target = st.selectbox("Select target column", cols, index=0 if len(cols) else None)

        # Train only once per upload unless user clears
        if st.button("Analyze and Train Models"):
            st.session_state["trained"] = False  # reset so we always run training after clicking
            try:
                with st.spinner("Analyzing dataset..."):
                    selector = ModelSelector(df, target)
                    analysis = selector.analyze()
                    suggestions = selector.suggest_models(top_k=5)
                st.success("Analysis complete.")
                st.json(analysis)
            except Exception as e:
                st.error(f"Dataset analysis failed: {e}")
                suggestions = None

            if suggestions:
                try:
                    with st.spinner("Training models (this may take a while)..."):
                        trainer = AutoTrainer(df, target, suggestions["problem_type"], suggestions["candidates"])
                        leaderboard = trainer.train_and_evaluate()
                    st.subheader("Leaderboard Results")
                    st.dataframe(leaderboard)
                    fig = px.bar(leaderboard, x="model", y="accuracy", title="Model Accuracy Comparison",
                                 color="model", text="accuracy")
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state["leaderboard"] = leaderboard
                    st.session_state["target"] = target
                    st.session_state["trained"] = True
                except Exception as e:
                    st.error(f"Model training failed: {e}")
                    st.session_state["trained"] = False


# Tab 3: Meta Learning

with tab3:
    st.header("Meta-Learning Model Recommendations")
    if not st.session_state["trained"] or st.session_state["leaderboard"] is None:
        st.warning("Train models first in the Model Training tab to view meta-learning insights.")
    else:
        df = st.session_state["dataframe"]
        target = st.session_state["target"]
        leaderboard = st.session_state["leaderboard"]

        try:
            with st.spinner("Updating meta logs and training recommender..."):
                meta_features = MetaFeatureExtractor.extract(df, target)
                MetaLogger.log_run(meta_features, leaderboard)
                logs = MetaLogger.load_logs()

                meta_rec = MetaRecommender()
                meta_rec.train_or_update(logs)

                candidate_models = leaderboard["model"].tolist() if "model" in leaderboard.columns else leaderboard.iloc[:, 0].tolist()
                predicted = meta_rec.predict_best_models(df, target, candidate_models, MetaFeatureExtractor)

                st.subheader("Predicted Best Models (Meta-Learning)")
                st.dataframe(predicted)

                fig = px.bar(predicted, x="model_name", y="predicted_accuracy",
                             title="Meta-Predicted Accuracy by Model", color="predicted_accuracy")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Meta-learning failed: {e}")
