"""Streamlit app for the Wine Quality Prediction laboratory.

Run with: streamlit run streamlit_app.py
Upload a labeled CSV (with quality) to replace the active dataset and refresh
model evaluation, or an unlabeled CSV to predict using the reference model.
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
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
LABELS = ["NEEDS IMPROVEMENT", "GOOD QUALITY"]


def _normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim quoting/spacing and map common header variants to canonical names."""
    canonical = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in FEATURE_COLUMNS}
    canonical["quality"] = "quality"
    renamed = {}
    for column in frame.columns:
        cleaned = str(column).replace("\ufeff", "").strip().strip('"').strip("'").strip()
        key = re.sub(r"[^a-z0-9]", "", cleaned.lower())
        renamed[column] = canonical.get(key, cleaned)
    frame = frame.rename(columns=renamed)
    return frame.loc[:, ~frame.columns.duplicated()]


def _read_csv_flexible(source: str | BytesIO) -> pd.DataFrame:
    """Read comma-, semicolon-, or tab-delimited CSV, including quoted UCI headers."""
    if isinstance(source, BytesIO):
        text = source.getvalue().decode("utf-8-sig")
    elif str(source).startswith(("http://", "https://")):
        text = pd.read_csv(source, sep=";", encoding="utf-8-sig").to_csv(index=False)
    else:
        text = Path(source).read_text(encoding="utf-8-sig")

    try:
        delimiter = csv.Sniffer().sniff(text[:8192], delimiters=";,\t").delimiter
    except csv.Error:
        header = text.splitlines()[0] if text else ""
        delimiter = max((";", ",", "\t"), key=header.count)

    best_frame, best_score = None, -1
    separators = [delimiter] + [sep for sep in (";", ",", "\t") if sep != delimiter]
    for separator in separators:
        try:
            frame = _normalize_headers(
                pd.read_csv(StringIO(text), sep=separator, skipinitialspace=True)
            )
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        columns = set(frame.columns)
        score = sum(name in columns for name in FEATURE_COLUMNS) + int("quality" in columns)
        if score > best_score:
            best_frame, best_score = frame, score
        if score == len(FEATURE_COLUMNS) + 1:
            break

    if best_frame is None or best_score == 0:
        raise ValueError("Could not detect a valid CSV header or delimiter.")
    return best_frame


def _read_wine_csv(source: str | BytesIO) -> pd.DataFrame:
    """Read and validate the labeled training dataset."""
    frame = _read_csv_flexible(source)
    missing = [name for name in [*FEATURE_COLUMNS, "quality"] if name not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
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
    """Create the laboratory's imputation/scaling/Random Forest pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )


@st.cache_resource(show_spinner="Training and evaluating the model on the active dataset...")
def train_and_evaluate(dataset: pd.DataFrame):
    """Train using active labeled data and calculate stratified holdout metrics."""
    clean = dataset.drop_duplicates().copy()
    clean["quality"] = pd.to_numeric(clean["quality"], errors="coerce")
    clean = clean.dropna(subset=["quality"])
    y = (clean["quality"] >= 6).astype(int)
    X = clean[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("The labeled dataset needs at least two records in each quality class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = build_model().fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=LABELS,
        zero_division=0, output_dict=True,
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
    "The original quality score creates the target and is never an input feature."
)

with st.expander("App features", expanded=True):
    st.markdown(
        "- Dataset preview\n- Missing-value and duplicate checks\n- Quality distribution view\n"
        "- Model evaluation metrics\n- Upload CSV prediction flow\n- Download prediction results\n"
        "- Responsible-use warning"
    )

# Upload before rendering dashboard metrics so labeled uploads can become the
# active dataset and refresh the dataset view and holdout evaluation together.
uploaded = st.file_uploader(
    "Upload a CSV dataset or prediction file",
    type=["csv"],
    help=(
        "A CSV with all 11 measurement columns and a quality column replaces the active "
        "dataset and recalculates model evaluation. Without quality, it is prediction-only."
    ),
)
upload_data = None
upload_error = None
if uploaded is not None:
    try:
        upload_data = _read_csv_flexible(BytesIO(uploaded.getvalue()))
        missing_features = [name for name in FEATURE_COLUMNS if name not in upload_data.columns]
        if missing_features:
            upload_error = (
                "Upload is missing required measurement columns: " + ", ".join(missing_features)
            )
        elif upload_data.empty:
            upload_error = "The uploaded CSV has no data rows."
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        upload_error = f"Could not read the uploaded CSV: {exc}"

try:
    reference_data = load_dataset()
    active_data = reference_data
    dataset_source = "Reference UCI red-wine dataset"
    if upload_data is not None and not upload_error and "quality" in upload_data.columns:
        upload_data["quality"] = pd.to_numeric(upload_data["quality"], errors="coerce")
        labeled_upload = upload_data.dropna(subset=["quality"])
        if labeled_upload.empty:
            upload_error = "The uploaded quality column has no numeric quality scores."
        else:
            active_data = labeled_upload
            dataset_source = f"Uploaded dataset: {uploaded.name}"
            st.success(
                "Labeled dataset loaded. The dataset preview, summary, model, and holdout "
                "evaluation below are now based on this uploaded CSV."
            )
    elif upload_data is not None and not upload_error:
        st.info(
            "This upload has no `quality` column, so it will be used for predictions only. "
            "Dataset statistics and evaluation remain based on the labeled reference dataset; "
            "evaluation cannot be recalculated without true quality labels."
        )

    if upload_error:
        st.error(upload_error)
    if active_data.empty:
        st.error("The active dataset has no valid records.")
        st.stop()
    model, accuracy, metrics, matrix, test_count = train_and_evaluate(active_data)
except Exception as exc:
    st.error("Could not load or evaluate the active dataset. Check the file and try again.")
    st.exception(exc)
    st.stop()

st.caption(f"Current dataset and evaluation source: **{dataset_source}**")
first, second, third = st.columns(3)
first.metric("Records", f"{len(active_data):,}")
second.metric("Average quality score", f"{active_data['quality'].mean():.2f}")
third.metric("Good-quality records (score ≥ 6)", f"{int((active_data['quality'] >= 6).sum()):,}")

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

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(active_data.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing values", int(active_data.isnull().sum().sum()))
    c2.metric("Duplicate rows", int(active_data.duplicated().sum()))
    c3.metric("Columns", active_data.shape[1])
    st.subheader("Quality score distribution")
    st.bar_chart(active_data["quality"].value_counts().sort_index())

with tab_evaluation:
    st.subheader("Random Forest holdout evaluation")
    st.write(
        "Evaluation is recalculated from the active labeled dataset using a stratified "
        "80/20 split (random_state=42), median imputation, scaling, and a 150-tree Random Forest."
    )
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
        "The upload above is used here for predictions. It may be semicolon- or comma-delimited. "
        "If it contains `quality`, that column is ignored as an input feature."
    )
    sample = active_data[FEATURE_COLUMNS].head(5)
    st.download_button(
        "Download sample input CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_wine_input.csv",
        mime="text/csv",
    )
    if uploaded is None:
        st.info("Upload a CSV above to generate predictions.")
    elif upload_error:
        st.warning("Fix the upload issue shown above before generating predictions.")
    elif upload_data is not None:
        inputs = upload_data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        invalid_rows = inputs.isna().all(axis=1)
        if invalid_rows.any():
            st.error(f"{int(invalid_rows.sum())} row(s) have no usable numeric measurement values.")
        else:
            predictions = model.predict(inputs)
            results = upload_data.copy()
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

with tab_limits:
    st.subheader("Limitations and responsible use")
    st.markdown(
        "- The model uses red-wine data and may not generalize to other types, producers, regions, or conditions.\n"
        "- Sensory quality ratings are subjective and the dataset is limited.\n"
        "- Predictions do not replace laboratory testing, sensory evaluation, food-safety procedures, or professional decisions.\n"
        "- More producers, regions, vintages, tasters, cross-validation, and pilot testing would help assess reliability."
    )

st.warning(
    "Classroom prototype only: a prediction is a preliminary screening result, "
    "not a final quality decision."
)
