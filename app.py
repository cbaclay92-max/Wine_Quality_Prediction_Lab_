from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "wine_quality.csv"
MODEL_PATH = APP_DIR / "wine_model.pkl"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET_COLUMN = "quality"
GOOD_QUALITY_THRESHOLD = 6

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the local dataset, downloading it from UCI when needed."""
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        data = pd.read_csv(DATA_URL, sep=";")
        data.to_csv(DATA_PATH, index=False)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    return data


@st.cache_resource(show_spinner="Preparing the wine quality model...")
def load_model():
    """Load the saved pipeline, or train it on first launch."""
    if not MODEL_PATH.exists():
        # Importing this module does not train; the script's main guard does.
        from train_model import train_and_evaluate

        train_and_evaluate(data_path=DATA_PATH, model_path=MODEL_PATH)
    return joblib.load(MODEL_PATH)


st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary red-wine quality classification")

try:
    data = load_dataset()
    model = load_model()
except Exception as exc:  # Give Streamlit users an actionable setup error.
    st.error(
        "The dataset or model could not be prepared. Check your internet connection, "
        "install the packages in requirements.txt, or run `python train_model.py` first."
    )
    st.exception(exc)
    st.stop()

feature_columns = list(getattr(model, "feature_names_in_", []))
if not feature_columns:
    st.error("The saved model does not expose its expected feature columns. Retrain it with `python train_model.py`.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Dataset records", f"{len(data):,}")
c2.metric("Average quality score", f"{data[TARGET_COLUMN].mean():.2f}")
c3.metric("Good-quality records", f"{int((data[TARGET_COLUMN] >= GOOD_QUALITY_THRESHOLD).sum()):,}")

with st.expander("About the training data", expanded=False):
    st.dataframe(data.head(10), use_container_width=True)
    st.write(f"Features used: {', '.join(feature_columns)}")

st.subheader("Predict wine samples")
st.write("Upload a CSV with one row per sample and the 11 physicochemical measurements.")

sample = data.loc[:, feature_columns].head(5)
st.download_button(
    "Download sample input CSV",
    data=sample.to_csv(index=False).encode("utf-8"),
    file_name="sample_wine_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose a wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

    missing_columns = [column for column in feature_columns if column not in new_data.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        st.stop()
    if new_data.empty:
        st.error("The uploaded CSV has a header but no sample rows.")
        st.stop()

    inputs = new_data.loc[:, feature_columns].copy()
    for column in feature_columns:
        converted = pd.to_numeric(inputs[column], errors="coerce")
        invalid = inputs[column].notna() & converted.isna()
        if invalid.any():
            st.error(f"Column '{column}' contains non-numeric values. Please correct the CSV and upload it again.")
            st.stop()
        inputs[column] = converted

    try:
        predictions = model.predict(inputs)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    results = new_data.copy()
    results["Prediction"] = [
        "GOOD QUALITY" if int(value) == 1 else "NEEDS IMPROVEMENT"
        for value in predictions
    ]

    st.subheader("Prediction results")
    st.dataframe(results, use_container_width=True)
    good_count = int((predictions == 1).sum())
    st.metric("Predicted good-quality samples", f"{good_count:,} / {len(predictions):,}")
    st.download_button(
        "Download prediction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="wine_quality_predictions.csv",
        mime="text/csv",
    )

st.info(
    "Responsible use: this classroom prototype is based on a public dataset. A prediction is not a substitute "
    "for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions."
)
