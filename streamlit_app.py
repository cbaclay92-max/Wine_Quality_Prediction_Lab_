"""Streamlit app for the Wine Quality Prediction laboratory.

Run with: streamlit run streamlit_app.py
Accepts comma- and semicolon-delimited CSV files, including UCI's quoted
semicolon-separated red-wine dataset.
"""

from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
import re
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
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
LABELS = ["NEEDS IMPROVEMENT", "GOOD QUALITY"]


def _normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim quoting/spacing and map common header variants to canonical names."""
    rename = {}
    canonical = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in FEATURE_COLUMNS}
    canonical["quality"] = "quality"
    for column in frame.columns:
        cleaned = str(column).replace("\ufeff", "").strip().strip('"').strip("'").strip()
        key = re.sub(r"[^a-z0-9]", "", cleaned.lower())
        rename[column] = canonical.get(key, cleaned)
    frame = frame.rename(columns=rename)
    # Duplicate headers after normalization are ambiguous and cannot be safely used.
    return frame.loc[:, ~frame.columns.duplicated()]


def _read_csv_flexible(source: str | BytesIO) -> pd.DataFrame:
    """Read a comma-, semicolon-, or tab-delimited CSV robustly."""
    if isinstance(source, BytesIO):
        raw = source.getvalue()
        text = raw.decode("utf-8-sig")
    elif str(source).startswith(("http://", "https://")):
        # The public UCI URL has a known delimiter and avoids re-downloading per attempt.
        text = pd.read_csv(source, sep=";", encoding="utf-8-sig").to_csv(index=False)
    else:
        text = Path(source).read_text(encoding="utf-8-sig")

    sample = text[:8192]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        # Choose the delimiter occurring most often in the header line.
        first_line = sample.splitlines()[0] if sample else ""
        delimiter = max((";", ",", "\t"), key=first_line.count)

    candidates = [delimiter] + [sep for sep in (";", ",", "\t") if sep != delimiter]
    best_frame = None
    best_score = -1
    for sep in candidates:
        try:
            frame = pd.read_csv(StringIO(text), sep=sep, skipinitialspace=True)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        frame = _normalize_headers(frame)
        available = set(frame.columns)
        score = sum(column in available for column in FEATURE_COLUMNS)
        if "quality" in available:
            score += 1
        if score > best_score:
            best_frame, best_score = frame, score
        if score == len(FEATURE_COLUMNS) + 1:
            break

    if best_frame is None or best_score == 0:
        raise ValueError("Could not detect a valid CSV header or delimiter.")
    return best_frame


def _read_wine_csv(source: str | BytesIO) -> pd.DataFrame:
    """Read a training dataset and validate its target and feature columns."""
    frame = _read_csv_flexible(source)
    missing = [column for column in [*FEATURE_COLUMNS, "quality"] if column not in frame.columns]
    if missing:
        raise ValueError("Training data is missing required columns: " + ", ".join(missing))
    return frame


@st.cache_data(show_spinner="Loading the red wine dataset...")
def load_dataset() -> pd.DataFrame:
    """Load a local dataset, a matching CSV from a ZIP, or the UCI dataset."""
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
                    (name for name in archive.namelist() if name.lower().endswith("winequality-red.csv")),
                    None,
                )
                if match:
                    with archive.open(match) as csv_file:
                        return _read_wine_csv(BytesIO(csv_file.read()))

    return _read_wine_csv(DATA_URL)


def build_model() -> Pipeline:
    """Create the lab Random Forest preprocessing/classification pipeline."""
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
    """Train on a stratified split and return model and holdout metrics."""
    clean = dataset.drop_duplicates().copy()
    clean["quality"] = pd.to_numeric(clean["quality"], errors="coerce")
    clean = clean.dropna(subset=["quality"])
    y = (clean["quality"] >= 6).astype(int)
    # quality is the target only; it is deliberately excluded from predictors.
    X = clean[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
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
    "The original quality score creates the training target and is never an input feature."
)

with st.expander("App features", expanded=True):
    st.markdown(
        "- Dataset preview\n"
        "- Missing-value and duplicate checks\n"
        "- Quality distribution view\n"
        "- Model evaluation metrics\n"
        "- Upload CSV prediction flow\n"
        "- Download prediction results\n"
        "- Responsible-use warning"
    )

try:
    data = load_dataset()
    data["quality"] = pd.to_numeric(data["quality"], errors="coerce")
    data = data.dropna(subset=["quality"])
    if data.empty:
        st.error("The dataset has no rows with a numeric quality score.")
        st.stop()
    model, accuracy, metrics, matrix, test_count = train_and_evaluate(data)
except Exception as exc:
    st.error("Could not load or train the wine model. Check the dataset and try again.")
    st.exception(exc)
    st.stop()

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
                    "Eleven physicochemical measurements from the red-wine dataset.",
                    "Validate columns → impute missing values → classify → display results.",
                    "GOOD QUALITY / NEEDS IMPROVEMENT and evaluation results.",
                    "Trained on one red-wine dataset; results may not generalize to other settings.",
                ],
            }
        ).set_index("SAD element")
    )
    st.markdown(
        "**Mini use-case:** upload a CSV → validate required measurement columns → "
        "predict each row → review or download results."
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
    st.write("Median imputation → scaling → 150-tree Random Forest; stratified 80/20 split; random_state=42.")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("Test samples", test_count)
    st.markdown("**Precision, recall, and F1-score per class**")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    st.dataframe(matrix, use_container_width=True)
    st.info("Review precision, recall, and F1-score as well as accuracy. Low GOOD QUALITY recall means qualifying samples may be missed.")

with tab_predict:
    st.subheader("Predict uploaded wine samples")
    st.write(
        "Upload a CSV containing the 11 measurement columns. Comma- and semicolon-separated "
        "files are supported, including the original quoted UCI file. The optional `quality` "
        "column is ignored during prediction."
    )
    sample = data[FEATURE_COLUMNS].head(5)
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
            missing = [column for column in FEATURE_COLUMNS if column not in new_data.columns]
            if new_data.empty:
                st.error("The uploaded CSV has no data rows.")
            elif missing:
                st.error(
                    "Missing required columns: " + ", ".join(missing)
                    + ". Ensure the file has a header row with the 11 wine measurements. "
                    "The original semicolon-delimited UCI CSV is supported; you can also "
                    "download the sample input CSV above."
                )
            else:
                inputs = new_data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
                invalid_rows = inputs.isna().all(axis=1)
                if invalid_rows.any():
                    st.error(f"{int(invalid_rows.sum())} row(s) have no usable numeric measurement values.")
                else:
                    # Partial missing measurements are handled by the pipeline's median imputer.
                    predictions = model.predict(inputs)
                    results = new_data.copy()
                    results["Prediction"] = [LABELS[int(value)] for value in predictions]
                    st.success(f"Predictions completed for {len(results):,} uploaded row(s).")
                    st.dataframe(results, use_container_width=True)
                    st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
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
