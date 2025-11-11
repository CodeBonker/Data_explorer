import sys
import os
import math
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
from ml_model.data_preprocessor import DataPreprocessor
from ml_model.data_quality_analyzer import DataQualityAnalyzer



st.set_page_config(page_title="AI Data Explorer 2.0", layout="wide")
st.title("AI Data Explorer 2.0")
st.caption("Upload dataset, explore interactively, then train models when ready.")


# Helper functions: EDA tools

def compute_missing_stats(df):
    total_cells = df.size
    total_missing = int(df.isnull().sum().sum())
    missing_by_col = (df.isnull().sum() / len(df)).sort_values(ascending=False)
    return {
        "total_missing": total_missing,
        "percent_missing": round(100 * total_missing / total_cells, 3),
        "missing_by_col": missing_by_col
    }


def plot_missing_heatmap(df):
    """Visualize missing values using Seaborn (safe for large data)"""
    # Sample for performance if large
    sample = df if len(df) <= 2000 else df.sample(2000, random_state=42)
    missing = sample.isnull().astype(int)
    # Reorder columns by missingness
    ordered_cols = missing.sum().sort_values(ascending=False).index
    missing = missing[ordered_cols]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(missing.T, cbar=False, cmap="Reds", ax=ax)
    ax.set_xlabel("Sampled Rows")
    ax.set_ylabel("Columns (sorted by missingness)")
    ax.set_title("Missing Value Heatmap (sampled up to 2000 rows)")
    st.pyplot(fig)


def plot_correlation(df):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns to compute a correlation matrix")
        return
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu",
                    title="Correlation matrix (numeric columns)")
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_distribution(df, column, target_col=None):
    counts = df[column].value_counts().reset_index()

    # Ensure unique column names
    counts.columns = [f"{column}_value", "count"]
    counts = counts.loc[:, ~counts.columns.duplicated()]

    fig = px.bar(
        counts.head(200),
        x=f"{column}_value",
        y="count",
        title=f"Category counts for {column}"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_outliers_box(df, column):
    if column not in df.columns:
        st.error(f"Column '{column}' not found.")
        return
    ser = df[column]
    if not pd.api.types.is_numeric_dtype(ser):
        st.warning("Boxplot outliers only available for numeric columns.")
        return
    fig = px.box(df, y=column, points="outliers", title=f"Boxplot for {column} (outliers shown)")
    st.plotly_chart(fig, use_container_width=True)
    # compute IQR outliers count
    q1 = ser.quantile(0.25)
    q3 = ser.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = int(((ser < lower) | (ser > upper)).sum())
    st.write(f"Outliers detected (IQR method): {n_out} rows")


def scatter_matrix_sample(df, cols=None, sample_n=1000):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        st.warning("No numeric columns available for scatter matrix.")
        return
    if cols is None:
        cols = numeric_df.columns.tolist()
    cols = [c for c in cols if c in numeric_df.columns]
    if len(cols) < 2:
        st.warning("Need at least two numeric columns for scatter matrix.")
        return
    if len(cols) > 6:
        cols = cols[:6]
        st.info("Scatter matrix limited to first 6 numeric columns for performance.")
    sample = df[cols] if len(df) <= sample_n else df[cols].sample(sample_n, random_state=42)
    fig = px.scatter_matrix(sample, dimensions=cols, title="Scatter matrix (sampled)")
    st.plotly_chart(fig, use_container_width=True)


def detect_id_like_columns(df):
    id_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            id_cols.append(col)
        elif df[col].nunique() / max(len(df), 1) > 0.95 and df[col].dtype == object:
            # high-cardinality string column might be an identifier
            id_cols.append(col)
    return id_cols


def compute_skewness_flags(df):
    skew_flags = {}
    numeric = df.select_dtypes(include=["number"])
    for col in numeric.columns:
        ser = numeric[col].dropna()
        if len(ser) < 3:
            continue
        skew = float(ser.skew())
        skew_flags[col] = skew
    return skew_flags


def generate_eda_insights(df, target_col=None):
    insights = []
    # missingness
    ms = compute_missing_stats(df)
    if ms["percent_missing"] > 5:
        insights.append(f"Dataset has {ms['percent_missing']}% missing values overall.")
    top_missing = ms["missing_by_col"].head(5)
    for col, frac in top_missing.items():
        pct = round(100 * float(frac), 2)
        if pct > 0:
            insights.append(f"Column '{col}' has {pct}% missing values.")
    # id-like
    id_cols = detect_id_like_columns(df)
    if id_cols:
        insights.append(f"Detected possible identifier columns: {', '.join(id_cols)}. Consider dropping them.")
    # low variance
    low_var = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            low_var.append(col)
    if low_var:
        insights.append(f"Columns with single unique value: {', '.join(low_var)}. Consider removing.")
    # cardinality
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        u = df[col].nunique(dropna=True)
        if u > 100:
            insights.append(f"Categorical column '{col}' has high cardinality ({u} unique). Consider encoding strategy or grouping.")
    # skewness
    skew_map = compute_skewness_flags(df)
    for c, s in list(skew_map.items())[:5]:
        if abs(s) > 1:
            insights.append(f"Numeric column '{c}' is highly skewed (skew={round(s,2)}). Consider log/box-cox transform.")
    # target imbalance
    if target_col and target_col in df.columns:
        try:
            vc = df[target_col].value_counts(normalize=True, dropna=True)
            if len(vc) > 1:
                top = float(vc.iloc[0])
                if top > 0.8:
                    insights.append(f"Target '{target_col}' is imbalanced: dominant class has {round(100*top,1)}% of samples.")
        except Exception:
            pass
    if not insights:
        insights.append("No critical issues detected. Data looks reasonably clean.")
    return insights


# Session state initialization
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

# Tab 1: Enhanced EDA
with tab1:
    st.header("Exploratory Data Analysis and Automated Insights")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            st.session_state["dataframe"] = df

            st.subheader("Data Quality Assessment")
            qa = DataQualityAnalyzer(df)
            score, quality_report = qa.calculate_health_score()
            st.progress(int(score))
            st.write(f"**Data Health Score:** {score}/100")
            st.json(quality_report["scores"])

            st.subheader("Preview")
            st.dataframe(df.head())
            
            st.write("Shape:", df.shape)

            ms = compute_missing_stats(df)
            st.write("Missing values (total):", int(ms["total_missing"]), f"({ms['percent_missing']}% of cells)")

            # Insights panel
            with st.expander("Automated Insights"):
                insights = generate_eda_insights(df, target_col=None)
                for i, insight in enumerate(insights, 1):
                    st.write(f"{i}. {insight}")

            # Missing heatmap
            with st.expander("Missing Values Heatmap"):
                plot_missing_heatmap(df)

            # Correlation
            with st.expander("Correlation Matrix (numeric)"):
                plot_correlation(df)

            # Outliers / distributions
            cols = list(df.columns)
            if cols:
                col1, col2 = st.columns([3, 2])
                with col1:
                    sel = st.selectbox("Feature for distribution/outlier check", cols)
                    if st.button("Show distribution and boxplot"):
                        plot_feature_distribution(df, sel)
                        plot_outliers_box(df, sel)
                with col2:
                    sel_target = st.selectbox("Optional target column", [None] + cols)
                    if st.button("Show target distribution") and sel_target:
                        plot_feature_distribution(df, sel_target)

            # Scatter matrix (sampled)
            with st.expander("Scatter matrix (sampled)"):
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    chosen = st.multiselect("Select numeric columns (max 6)", numeric_cols, default=numeric_cols[:4])
                    if st.button("Plot scatter matrix"):
                        scatter_matrix_sample(df, chosen, sample_n=1000)
                else:
                    st.write("No numeric columns available for scatter matrix.")

            st.session_state["analyzed"] = True


# Tab 2: Model Training
with tab2:
    st.header("AutoML Model Training")
    if st.session_state["dataframe"] is None:
        st.warning("Please upload a dataset in the EDA tab first.")
    else:
        df = st.session_state["dataframe"]
        cols = list(df.columns)
        target = st.selectbox("Select target column", cols, index=0)
        if st.button("Analyze and Train Models"):
            st.session_state["trained"] = False
            try:
                with st.spinner("Analyzing dataset..."):
                    selector = ModelSelector(df, target)

                    st.info("Running automated preprocessing...")
                    pre = DataPreprocessor(df, target)
                    df_processed, y, prep_report = pre.run()
                    st.success("Data preprocessing complete!")
                    st.json(prep_report)
                    df = pd.concat([df_processed, y], axis=1)

                    analysis = selector.analyze()
                    suggestions = selector.suggest_models(top_k=5)

                st.success("Analysis complete.")
                st.json(analysis)
            except Exception as e:
                st.error(f"Dataset analysis failed: {e}")
                suggestions = None

            if suggestions:
                try:
                    with st.spinner("Training models..."):
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
