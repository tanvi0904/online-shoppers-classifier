"""Streamlit app: upload test CSV, pick a saved model, see metrics + confusion matrix.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from shopper_utils import FEATURES, MODEL_FILES, TARGET, clean_features, compute_metrics, to_binary

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"

st.set_page_config(page_title="Will this shopper buy?", page_icon="🛒", layout="wide")


# cached so each model is only loaded from disk once
@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


# only predicts, never fits, so uploaded data can't change a model
def evaluate(name: str, X: pd.DataFrame, y):
    model = load_model(name)
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score, compute_metrics(y, y_pred, y_score)


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------- sidebar ----------
st.sidebar.header("⚙️ Setup")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
if SAMPLE_TEST.exists():
    st.sidebar.download_button("Download sample test_data.csv", SAMPLE_TEST.read_bytes(), "test_data.csv", "text/csv")
st.sidebar.caption("CSV needs all 17 feature columns plus the `Revenue` target column.")

# ---------- header ----------
st.title("🛒 Online Shopper Purchase Prediction")
st.write(
    "Predicts whether a browsing session on an e-commerce site ends in a purchase (`Revenue`), "
    "using the UCI *Online Shoppers Purchasing Intention* dataset. Models were trained offline; "
    "this app only loads them and scores the uploaded test data."
)

# nothing to show until a csv is uploaded
if uploaded is None:
    st.info("👈 Upload a test CSV from the sidebar to get started (you can download the sample one there).")
    st.stop()

df = pd.read_csv(uploaded)
# stop with a clear message if any column is missing
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {', '.join(missing)}")
    st.stop()

# same cleaning as in training
X, y = clean_features(df), to_binary(df[TARGET])
st.success(f"Loaded {len(df):,} rows — {y.sum():,} purchases ({y.mean():.1%}).")

tab_single, tab_compare, tab_data = st.tabs(["📊 Selected model", "🏁 Compare all models", "🔎 Data preview"])

# tab 1: one model at a time
with tab_single:
    y_pred, y_score, metrics = evaluate(model_name, X, y)
    st.subheader(f"{model_name} — evaluation on uploaded data")
    for col, (k, v) in zip(st.columns(len(metrics)), metrics.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.pyplot(plot_confusion(y, y_pred, f"Confusion matrix — {model_name}"))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, target_names=["No purchase", "Purchase"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # uploaded rows plus the prediction and purchase probability
    out = df.assign(Predicted=y_pred.astype(bool), Purchase_Probability=y_score.round(4))
    st.download_button("Download predictions", out.to_csv(index=False), f"predictions_{MODEL_FILES[model_name][:-7]}.csv")

# tab 2: all five models side by side
with tab_compare:
    st.subheader("All five models on the same uploaded data")
    table = pd.DataFrame({name: evaluate(name, X, y)[2] for name in MODEL_FILES}).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"), use_container_width=True)
    st.bar_chart(table[["Accuracy", "AUC", "F1", "MCC"]])
    best = table["MCC"].idxmax()
    st.markdown(f"Best by MCC (fairest single number for this imbalanced data): **{best}**")

# tab 3: a look at the uploaded rows
with tab_data:
    st.dataframe(df.head(50), use_container_width=True)
