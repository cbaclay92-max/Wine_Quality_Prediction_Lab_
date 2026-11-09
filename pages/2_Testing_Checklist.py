"""In-app checklist for verifying the wine prediction lab deliverables."""

import streamlit as st

st.set_page_config(page_title="Lab Testing Checklist", page_icon="🧪", layout="wide")
st.title("🧪 Laboratory Testing Checklist")
st.caption("Wine Quality Prediction System — Section B")

st.info(
    "Use this page to record your own checks. The checkboxes are not automated test results "
    "and reset when the app session restarts. Only mark an item after you verify it locally."
)

checks = [
    (
        "Run `python train_model.py`",
        "The script finishes without errors and creates `wine_quality.csv` and `wine_model.pkl`.",
    ),
    (
        "Run the Streamlit app",
        "Run `streamlit run streamlit_app.py`; the dashboard opens in your browser.",
    ),
    (
        "Upload a sample CSV",
        "Upload a CSV with the 11 measurement columns; the app displays predictions.",
    ),
    (
        "Upload a CSV missing a required column",
        "The app displays a helpful error identifying the missing measurement columns.",
    ),
    (
        "Download prediction results",
        "The downloaded CSV contains the uploaded rows and a `Prediction` column.",
    ),
    (
        "Open the GitHub repository",
        "The source files and README are visible in the repository.",
    ),
]

st.subheader("Verification")
for title, expected in checks:
    st.checkbox(f"{title} — {expected}", key=f"check_{title}")

st.subheader("Upload and evaluation note")
st.markdown(
    "- A labeled CSV with the 11 required measurement columns and numeric `quality` values "
    "becomes the active dataset; its preview, summary, model, and holdout evaluation refresh.\n"
    "- An unlabeled CSV can be used for predictions, but evaluation metrics remain tied to the "
    "labeled reference data because accuracy and related metrics require true labels.\n"
    "- The prediction page is the main app. Return to it using the page navigation in the sidebar."
)

st.warning(
    "Do not treat these checkboxes as evidence that tests passed. Keep test evidence such as "
    "terminal output or screenshots for your lab submission."
)
