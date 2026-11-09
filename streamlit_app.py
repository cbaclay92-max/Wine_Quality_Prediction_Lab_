"""Streamlit interface for the wine quality prediction lab.

Run with: streamlit run streamlit_app.py
The app loads the red-wine dataset from a local CSV/ZIP when available, or
falls back to the public UCI dataset, then trains its classroom model on demand.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DATA_CANDIDATES = ("wine_quality.csv", "winequality-red.csv")
ZIP_CANDIDATES = ("wine-quality-prediction.zip", "winequality.zip")


def _read_wine_csv(source: str | BytesIO) -> pd.DataFrame:
    """Read either the semicolon-delimited UCI file or a regular CSV copy."""
    frame = pd.read_csv(source, sep=";")
    if "quality" not in frame.columns and isinstance(source, (str, Path)):
        frame = pd.read_csv(source)
    if "quality" not in frame.columns:
        raise ValueError("The dataset must contain a 'quality' column.")
    return frame


@st.cache_data(show_spinner="Loading the red wine dataset...")
def load_dataset() -> pd.DataFrame:
    root = Path(__file__).resolve().parent

    for filename in DATA_CANDIDATES:
        path = root / filename
        if path.is_file():
            return _read_wine_csv(str(path))

    # The repository may include the original UCI ZIP instead of extracted CSVs.
    for filename in ZIP_CANDIDATES:
        path = root / filename
        if not path.is_file():
            continue
        with zipfile.ZipFile(path) as archive:
            match = next(
                (name for name in archive.namelist() if name.lower().endswith("winequality-red.csv")),
                None,
            )
            if match:
                with archive.open(match) as csv_file:
                    return _read_wine_csv(BytesIO(csv_file.read()))

    # Last resort allows deployment without committing the dataset to the repo.
    return _read_wine_csv(DATA_URL)


@st.cache_resource(show_spinner="Training the wine quality model...")
def train_model(dataset: pd.DataFrame):
    clean = dataset.drop_duplicates().copy()
    features = clean.drop(columns=["quality"])
    target = (clean["quality"] >= 6).astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model


st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification")
st.info(
    "This model predicts GOOD QUALITY when the dataset's quality score is at least 6. "
    "The original quality score is used only as the training target, never as an input feature."
)

try:
    data = load_dataset()
    if data.empty:
        st.error("The wine dataset is empty.")
        st.stop()
    if not data.select_dtypes(include="number").columns.size:
        st.error("The wine dataset has no numeric measurements to use as predictors.")
        st.stop()
    model = train_model(data)
except Exception as exc:
    st.error(
        "Could not load the wine dataset. Place `wine_quality.csv` or "
        "`winequality-red.csv` beside this app, or make the UCI dataset available."
    )
    st.exception(exc)
    st.stop()

first, second, third = st.columns(3)
first.metric("Records", f"{len(data):,}")
second.metric("Average quality score", f"{data['quality'].mean():.2f}")
third.metric("Good-quality records (score ≥ 6)", f"{int((data['quality'] >= 6).sum()):,}")

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV with the same numeric measurement columns used to train the model.")
feature_columns = list(model.feature_names_in_)
template = data[feature_columns].head(5)
st.download_button(
    "Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_wine_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        if new_data.empty:
            st.error("The uploaded CSV has no rows.")
        else:
            missing = [column for column in feature_columns if column not in new_data.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                inputs = new_data[feature_columns].apply(pd.to_numeric, errors="coerce")
                if inputs.isna().all(axis=None):
                    st.error("The uploaded measurement columns do not contain usable numeric values.")
                else:
                    predictions = model.predict(inputs)
                    results = new_data.copy()
                    results["Prediction"] = [
                        "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                        for value in predictions
                    ]
                    st.subheader("Prediction results")
                    st.dataframe(results, use_container_width=True)
                    st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
                    st.download_button(
                        "Download prediction results",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="wine_quality_predictions.csv",
                        mime="text/csv",
                    )
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")

st.warning(
    "Responsible use: this is a classroom prototype. A prediction is not a substitute "
    "for laboratory testing, sensory evaluation, food-safety procedures, or professional "
    "quality-control decisions."
)
