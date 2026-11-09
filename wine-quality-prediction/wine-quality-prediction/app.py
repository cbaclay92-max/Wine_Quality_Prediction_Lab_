# app.py - Wine Quality Prediction System (Streamlit)
# Section B classroom prototype for preliminary quality classification.
#
# Run from the project folder:   streamlit run app.py
# It needs wine_quality.csv and wine_model.pkl (both created by train_model.py).
# If wine_model.pkl is missing or unreadable, the app trains the same model itself.

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "wine_model.pkl"
LABELS = ["Needs improvement", "Good quality"]

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show_df(df):
    """Show a table full-width (works on old and new Streamlit versions)."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


@st.cache_data
def load_data():
    """Read wine_quality.csv (made by train_model.py) or the original red-wine CSV."""
    clean = BASE / "wine_quality.csv"
    original = BASE / "winequality-red.csv"
    if clean.exists():
        return pd.read_csv(clean)
    if original.exists():
        return pd.read_csv(original, sep=";")
    return None


def make_xy(df):
    """Remove duplicates, build the binary target, drop quality from the inputs."""
    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)      # 1 = Good; 0 = Needs improvement
    X = df.drop(columns=["quality"])          # never use quality as a predictor
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])


@st.cache_resource
def load_model(data):
    """Load wine_model.pkl; if it is missing/unreadable, train the same model here."""
    try:
        model = joblib.load(MODEL_FILE)
        if hasattr(model, "feature_names_in_"):
            return model, "Loaded wine_model.pkl"
    except Exception:
        pass
    X, y = make_xy(data)
    X_train, _, y_train, _ = split_data(X, y)
    model = build_pipeline().fit(X_train, y_train)
    try:
        joblib.dump(model, MODEL_FILE)
    except Exception:
        pass
    return model, "wine_model.pkl was not usable, so the model was trained again inside the app"


@st.cache_data
def evaluate(_model, data):
    """Evaluate the model on the same 20% test split used in train_model.py."""
    X, y = make_xy(data)
    _, X_test, _, y_test = split_data(X, y)
    pred = _model.predict(X_test[list(_model.feature_names_in_)])
    acc = accuracy_score(y_test, pred)
    rep = classification_report(y_test, pred, target_names=LABELS,
                                zero_division=0, output_dict=True)
    table = pd.DataFrame({
        "Class": LABELS,
        "Precision": [rep[c]["precision"] for c in LABELS],
        "Recall": [rep[c]["recall"] for c in LABELS],
        "F1-score": [rep[c]["f1-score"] for c in LABELS],
        "Support": [int(rep[c]["support"]) for c in LABELS],
    }).round(3)
    cm = pd.DataFrame(
        confusion_matrix(y_test, pred),
        index=["Actual: " + c for c in LABELS],
        columns=["Predicted: " + c for c in LABELS],
    )
    return acc, table, cm, len(y_test)


def label_of(value):
    return "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"


# ----------------------------------------------------------------------------
# Load data and model
# ----------------------------------------------------------------------------
data = load_data()
if data is None:
    st.error("Dataset not found. Put wine_quality.csv (or winequality-red.csv) in the "
             "same folder as app.py, or run train_model.py first.")
    st.stop()

model, model_note = load_model(data)
features = list(model.feature_names_in_)
X_all, _ = make_xy(data)

# ----------------------------------------------------------------------------
# Title and headline numbers
# ----------------------------------------------------------------------------
st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

tab_overview, tab_data, tab_eval, tab_predict, tab_limits = st.tabs(
    ["📋 Overview (SAD)", "📊 Dataset", "✅ Model evaluation", "🔮 Predict", "⚠️ Limitations"]
)

# ----------------------------------------------------------------------------
# Tab 1 - SAD overview
# ----------------------------------------------------------------------------
with tab_overview:
    st.subheader("SAD mini-analysis")
    sad = pd.DataFrame({
        "SAD element": ["Problem statement", "Users", "Inputs", "Process", "Outputs", "Key limitation"],
        "Response": [
            "A producer needs a consistent preliminary way to screen wine samples using "
            "recorded physicochemical measurements.",
            "Quality-control staff, production supervisor, or data analyst.",
            "Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total "
            "sulfur dioxide, density, pH, sulphates, and alcohol.",
            "Validate input columns → preprocess data → apply trained model → classify quality category.",
            "GOOD QUALITY / NEEDS IMPROVEMENT label and model evaluation summary.",
            "The model is trained on a particular red-wine dataset and may not generalize to other "
            "wine types, producers, regions, or testing conditions.",
        ],
    })
    st.table(sad.set_index("SAD element"))
    st.markdown(
        "**Prediction rule:** quality ≥ 6 = **GOOD QUALITY**; quality < 6 = **NEEDS IMPROVEMENT**. "
        "The original quality score is the target only and is never used as an input feature."
    )
    st.markdown(
        "**Mini use-case:** the user enters sample measurements or uploads a CSV file → the system "
        "checks the required columns → the model predicts the quality category → the system "
        "displays and exports the results."
    )
    st.caption(f"Model status: {model_note}.")

# ----------------------------------------------------------------------------
# Tab 2 - Dataset
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset preview")
    show_df(data.head(10))

    d1, d2, d3 = st.columns(3)
    d1.metric("Missing values", int(data.isnull().sum().sum()))
    d2.metric("Duplicate rows", int(data.duplicated().sum()))
    d3.metric("Columns", data.shape[1])

    st.subheader("Quality score distribution")
    st.bar_chart(data["quality"].value_counts().sort_index())
    st.caption("Source: UCI Machine Learning Repository - Wine Quality (red wine). "
               "Duplicate rows are removed before training.")

# ----------------------------------------------------------------------------
# Tab 3 - Model evaluation
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Model evaluation summary")
    st.write("Random Forest pipeline (median imputer → scaler → 150 trees), evaluated on the "
             "20% test split (random_state = 42).")
    acc, table, cm, n_test = evaluate(model, data)
    e1, e2 = st.columns(2)
    e1.metric("Accuracy", f"{acc:.3f}")
    e2.metric("Test samples", n_test)

    st.markdown("**Precision, recall and F1-score per class**")
    show_df(table)
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    show_df(cm)
    st.info("Do not judge the model by accuracy alone. Recall for GOOD QUALITY shows how many truly "
            "good wines the model finds; low recall means good batches may be wrongly flagged as "
            "NEEDS IMPROVEMENT.")

# ----------------------------------------------------------------------------
# Tab 4 - Predict (manual entry + CSV upload)
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Option A - Enter one wine sample")
    with st.form("manual_form"):
        cols = st.columns(3)
        values = {}
        for i, name in enumerate(features):
            lo = float(X_all[name].min())
            hi = float(X_all[name].max())
            med = float(X_all[name].median())
            with cols[i % 3]:
                values[name] = st.number_input(
                    name,
                    min_value=0.0,
                    value=med,
                    step=max((hi - lo) / 200.0, 0.001),
                    format="%.4f",
                    help=f"Range in the training data: {lo:g} to {hi:g}",
                )
        submitted = st.form_submit_button("Predict quality")

    if submitted:
        row = pd.DataFrame([values])[features]
        pred = int(model.predict(row)[0])
        proba_good = float(model.predict_proba(row)[0][list(model.classes_).index(1)])
        if pred == 1:
            st.success(f"Prediction: GOOD QUALITY  (model confidence for good quality: {proba_good:.0%})")
        else:
            st.warning(f"Prediction: NEEDS IMPROVEMENT  (model confidence for good quality: {proba_good:.0%})")
        outside = [n for n in features
                   if values[n] < X_all[n].min() or values[n] > X_all[n].max()]
        if outside:
            st.caption("Note: these values are outside the training range, so the prediction is "
                       "less reliable: " + ", ".join(outside))

    st.divider()
    st.subheader("Option B - Predict uploaded wine samples")
    st.write("Upload a CSV containing the same predictor columns as the training data.")

    template = data.drop(columns=["quality"]).head(5)
    st.download_button(
        "Download sample input CSV",
        template.to_csv(index=False),
        "sample_wine_input.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])

    if uploaded is not None:
        new_data = pd.read_csv(uploaded)
        required_columns = list(model.feature_names_in_)
        missing = [c for c in required_columns if c not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [label_of(v) for v in predictions]

            st.subheader("Prediction results")
            show_df(results)
            st.metric(
                "Predicted good-quality samples",
                int((predictions == 1).sum())
            )
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "wine_quality_predictions.csv",
                "text/csv"
            )

# ----------------------------------------------------------------------------
# Tab 5 - Limitations and responsible use
# ----------------------------------------------------------------------------
with tab_limits:
    st.subheader("Limitations and responsible use")
    st.markdown(
        "- The model is trained on **one red-wine dataset** and may not generalize to other wine "
        "types, producers, regions, or testing conditions.\n"
        "- Quality scores are **subjective sensory ratings**, and the dataset is limited in size.\n"
        "- The prediction uses only physicochemical measurements, not taste, smell, or food-safety checks.\n"
        "- A prediction is a **preliminary screening result**, not a final quality decision."
    )
    st.markdown("**More data and testing that would improve reliability:** more samples (especially "
                "very low and very high quality), several producers, regions and vintages, ratings "
                "from more tasters, cross-validation, comparison with other models, and a pilot "
                "against laboratory and sensory results.")

st.info(
    "Responsible use: this is a classroom prototype. A prediction is not a substitute "
    "for laboratory testing, sensory evaluation, food-safety procedures, or professional "
    "quality-control decisions."
)
