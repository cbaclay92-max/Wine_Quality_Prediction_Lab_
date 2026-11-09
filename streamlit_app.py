"""Streamlit interface for the wine quality prediction lab.

Run with: streamlit run streamlit_app.py
The app accepts comma- or semicolon-delimited CSV files. If a local model
artifact is not available, it trains and evaluates the lab model on startup.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DATA_CANDIDATES = ("wine_quality.csv", "winequality-red.csv")
ZIP_CANDIDATES = ("wine-quality-prediction.zip", "winequality.zip")
LABELS = ["NEEDS IMPROVEMENT", "GOOD QUALITY"]


def _read_csv_flexible(source: str | BytesIO) -> pd.DataFrame:
    """Read comma- or semicolon-separated CSV and clean its column headers."""
    frame = pd.read_csv(
        source,
        sep=None,
        engine="python",
        skipinitialspace=True,
        encoding="utf-8-sig",
    )
    frame.columns = [str(column).strip().strip('"').strip("'") for column in frame.columns]
    return frame


def _read_wine_csv(source: str | BytesIO) -> pd.DataFrame:
    """Read a wine dataset and check its required target column."""
    frame = _read_csv_flexible(source)
    if "quality" not in frame.columns:
        raise ValueError(
            "The dataset must contain a 'quality' column. Check the header and "
            "that the file uses commas or semicolons as delimiters."
        )
    return frame


@st.cache_data(show_spinner="Loading the red wine dataset...")
def load_dataset() -> pd.DataFrame:
    """Load a local dataset, dataset in a ZIP, or the public UCI dataset."""
    root = Path(__file__).resolve().parent
    for filename in DATA_CANDIDATES:
        path = root / filename
        if path.is_file():
            return _read_wine_csv(str(path))

    for filename in ZIP_CANDIDATES:
        path = root / filename
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                match = next(
                    (
                        name
                        for name in archive.namelist()
                        if name.lower().endswith("winequality-red.csv")
                    ),
                    None,
                )
                if match:
                    with archive.open(match) as csv_file:
                        return _read_wine_csv(BytesIO(csv_file.read()))

    return _read_wine_csv(DATA_URL)


def build_model() -> Pipeline:
    """Create the Random Forest pipeline specified in the laboratory."""
    return Pipeline(
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


@st.cache_resource(show_spinner="Training and evaluating the wine quality model...")
def train_and_evaluate(dataset: pd.DataFrame):
    """Train on a stratified split and return model plus holdout metrics."""
    clean = dataset.drop_duplicates().copy()
    y = (clean["quality"] >= 6).astype(int)
    # Keep quality out of X: it is only the target used to create y.
    X = clean.drop(columns=["quality"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = build_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )
    metrics = pd.DataFrame(
        {
            "Class": LABELS,
            "Precision": [report[label]["precision"] for label in LABELS],
            "Recall": [report[label]["recall"] for label in LABELS],
            "F1-score": [report[label]["f1-score"] for label in LABELS],
            "Support": [int(report[label]["support"]) for label in LABELS],
        }
    ).round(3)
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=[f"Actual: {label}" for label in LABELS],
        columns=[f"Predicted: {label}" for label in LABELS],
    )
    return model, accuracy_score(y_test, predictions), metrics, matrix, len(y_test)


st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification")
st.info(
    "Prediction rule: quality ≥ 6 = GOOD QUALITY; quality < 6 = NEEDS IMPROVEMENT. "
    "The original quality score is used only to create the training target, never as an input feature."
)

try:
    data = load_dataset()
    if data.empty:
        st.error("The wine dataset is empty.")
        st.stop()
    if not pd.api.types.is_numeric_dtype(pd.to_numeric(data["quality"], errors="coerce")):
        data["quality"] = pd.to_numeric(data["quality"], errors="coerce")
    data = data.dropna(subset=["quality"])
    if data.empty:
        st.error("The dataset's quality column has no numeric values.")
        st.stop()
    model, accuracy, metrics, matrix, test_count = train_and_evaluate(data)
except Exception as exc:
    st.error(
        "Could not load or train the wine model. Check the dataset and required "
        "dependencies, then try again."
    )
    st.exception(exc)
    st.stop()

clean_data = data.drop_duplicates()
first, second, third = st.columns(3)
first.metric("Records", f"{len(data):,}")
second.metric("Average quality score", f"{data['quality'].mean():.2f}")
third.metric("Good-quality records (score ≥ 6)", f"{int((data['quality'] >= 6).sum()):,}")

tab_overview, tab_data, tab_evaluation, tab_predict, tab_limits = st.tabs(
    ["Overview (SAD)", "Dataset", "Model evaluation", "Predict", "Limitations"]
)

with tab_overview:
    st.subheader("SAD mini-analysis")
    st.table(
        pd.DataFrame(
            {
                "SAD element": ["Problem", "Users", "Inputs", "Process", "Outputs", "Limitation"],
                "Description": [
                    "Screen wine samples preliminarily using physicochemical measurements.",
                    "Quality-control staff, production supervisor, or data analyst.",
                    "The 11 physicochemical measurement columns in the wine dataset.",
                    "Validate columns → preprocess → apply classifier → return a category.",
                    "GOOD QUALITY / NEEDS IMPROVEMENT and evaluation results.",
                    "Trained on one red-wine dataset; results may not generalize to other settings.",
                ],
            }
        ).set_index("SAD element")
    )
    st.markdown(
        "**Mini use-case:** enter measurements or upload a CSV → validate required "
        "columns → predict the category → review or download results."
    )

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing values", int(data.isnull().sum().sum()))
    c2.metric("Duplicate rows", int(data.duplicated().sum()))
    c3.metric("Columns", data.shape[1])
    st.subheader("Quality score distribution")
    st.bar_chart(data["quality"].value_counts().sort_index())
    st.caption("Source: UCI Machine Learning Repository — Wine Quality (red wine).")

with tab_evaluation:
    st.subheader("Random Forest holdout evaluation")
    st.write(
        "Median imputation → scaling → 150-tree Random Forest. "
        "Stratified 80/20 split; random_state=42."
    )
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("Test samples", test_count)
    st.markdown("**Precision, recall, and F1-score per class**")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    st.dataframe(matrix, use_container_width=True)
    st.info(
        "Review precision, recall, and F1-score as well as accuracy. Low recall for "
        "GOOD QUALITY means some samples that meet the dataset's threshold may be missed."
    )

with tab_predict:
    st.subheader("Predict uploaded wine samples")
    st.write(
        "Upload a CSV with the 11 measurement columns. Both comma- and "
        "semicolon-delimited files are accepted. A quality column may be present; "
        "it is ignored for prediction."
    )
    feature_columns = list(model.feature_names_in_)
    sample = data[feature_columns].head(5)
    st.download_button(
        "Download sample input CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_wine_input.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
    if uploaded is not None:
        try:
            new_data = _read_csv_flexible(BytesIO(uploaded.getvalue()))
            if new_data.empty:
                st.error("The uploaded CSV has no rows.")
            else:
                missing = [column for column in feature_columns if column not in new_data.columns]
                if missing:
                    st.error(
                        "Missing required columns: "
                        + ", ".join(missing)
                        + ". Check that the first row contains the exact measurement headers. "
                        "Semicolon-delimited UCI files are supported; download the sample CSV above."
                    )
                else:
                    inputs = new_data[feature_columns].apply(pd.to_numeric, errors="coerce")
                    all_invalid = inputs.isna().all(axis=1)
                    if all_invalid.any():
                        st.error(
                            f"{int(all_invalid.sum())} row(s) have no usable numeric measurements. "
                            "Correct or remove those rows and upload again."
                        )
                    else:
                        valid_inputs = inputs.loc[~all_invalid]
                        predictions = model.predict(valid_inputs)
                        results = new_data.loc[~all_invalid].copy()
                        results["Prediction"] = [LABELS[int(value)] for value in predictions]
                        st.subheader("Prediction results")
                        st.dataframe(results, use_container_width=True)
                        st.metric(
                            "Predicted good-quality samples",
                            int((predictions == 1).sum()),
                        )
                        st.download_button(
                            "Download prediction results",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="wine_quality_predictions.csv",
                            mime="text/csv",
                        )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

with tab_limits:
    st.subheader("Limitations and responsible use")
    st.markdown(
        "- The model uses one red-wine dataset and may not generalize to other types, producers, regions, or conditions.\n"
        "- Sensory quality ratings are subjective and the dataset is limited.\n"
        "- Predictions do not replace laboratory testing, sensory evaluation, food-safety procedures, or professional decisions.\n"
        "- More producers, regions, vintages, tasters, cross-validation, and pilot testing would help assess reliability."
    )

st.warning(
    "Classroom prototype only: a prediction is a preliminary screening result, "
    "not a final quality decision."
)
