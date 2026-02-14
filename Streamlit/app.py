import json
import os
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="ML Classification Models - Assignment 2", layout="wide")


@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), "model")
    with open(os.path.join(base, "models.pkl"), "rb") as f:
        models = pickle.load(f)
    with open(os.path.join(base, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(base, "confusion_matrix.json"), "r", encoding="utf-8") as f:
        cm = json.load(f)
    with open(os.path.join(base, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return models, metrics, cm, feature_names


def metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T
    df = df[["accuracy", "auc", "precision", "recall", "f1", "mcc"]]
    df.columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    return df.round(4)


def make_template_csv(feature_names: List[str]) -> pd.DataFrame:
    # Create a small dummy template with correct columns
    # Values can be replaced by user
    return pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)


def main():
    st.title("Assignment 2 — End-to-End ML: Models + Streamlit UI + Deployment")

    models, metrics, cm_store, feature_names = load_artifacts()
    model_names = list(models.keys())

    st.sidebar.header("Controls")
    selected_model = st.sidebar.selectbox("Select a model", model_names, index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload (Test CSV)")
    uploaded = st.sidebar.file_uploader("Upload CSV (must match feature columns)", type=["csv"])

    st.sidebar.markdown("You can also download a template CSV and fill it with your test rows.")
    template_df = make_template_csv(feature_names)
    st.sidebar.download_button(
        "Download template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="test_template.csv",
        mime="text/csv",
    )

    # Layout
    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Model Metrics Comparison (All 6 Models)")
        df_metrics = metrics_table(metrics)
        st.dataframe(df_metrics, use_container_width=True)

        st.subheader(f"Selected Model: {selected_model}")
        one = df_metrics.loc[selected_model].to_dict()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{one['Accuracy']:.4f}")
        c2.metric("AUC", f"{one['AUC']:.4f}")
        c3.metric("Precision", f"{one['Precision']:.4f}")
        c4.metric("Recall", f"{one['Recall']:.4f}")
        c5.metric("F1", f"{one['F1']:.4f}")
        c6.metric("MCC", f"{one['MCC']:.4f}")

    with right:
        st.subheader("Confusion Matrix / Classification Report (Held-out Test Split)")
        cm = cm_store[selected_model]["confusion_matrix"]
        st.write("Confusion Matrix (rows=true, cols=pred):")
        st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]), use_container_width=True)

        st.write("Classification Report:")
        report = cm_store[selected_model]["classification_report"]
        # show key rows in a neat table
        report_df = pd.DataFrame(report).T
        if {"precision", "recall", "f1-score"}.issubset(set(report_df.columns)):
            st.dataframe(report_df[["precision", "recall", "f1-score", "support"]].round(4), use_container_width=True)
        else:
            st.json(report)

    st.markdown("---")
    st.subheader("Run Predictions on Your Uploaded Test CSV")

    if uploaded is None:
        st.info("Upload a CSV from the sidebar to get predictions. Use the template CSV if needed.")
        return

    try:
        user_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    missing = [c for c in feature_names if c not in user_df.columns]
    extra = [c for c in user_df.columns if c not in feature_names]

    if missing:
        st.error(f"Your CSV is missing required feature columns: {missing}")
        st.stop()

    if extra:
        st.warning(f"Extra columns found (they will be ignored): {extra}")

    X_user = user_df[feature_names].values
    model = models[selected_model]

    preds = model.predict(X_user)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_user)[:, 1]
    else:
        probs = None

    out = user_df.copy()
    out["prediction"] = preds
    if probs is not None:
        out["prob_class_1"] = probs

    st.dataframe(out.head(50), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
