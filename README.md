# Wine Quality Prediction Lab

A Streamlit app that demonstrates preliminary red-wine quality classification from physicochemical measurements. The app trains a Random Forest pipeline when it starts and lets you predict from an uploaded CSV.

> **Responsible use:** This is a classroom prototype, not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## Run locally

Requirements: Python 3.10 or newer and Git.

```bash
git clone https://github.com/cbaclay92-max/Wine_Quality_Prediction_Lab_.git
cd Wine_Quality_Prediction_Lab_
python -m venv .venv
```

Activate the virtual environment, then install and launch:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**.
3. Choose this repository, the `main` branch, and `streamlit_app.py` as the **Main file path**.
4. Deploy. The root-level `requirements.txt` lists the app dependencies.

The app looks for a wine CSV beside `streamlit_app.py`, then checks `wine-quality-prediction.zip` for the dataset, and finally falls back to the public UCI red-wine CSV. No model artifact needs to be committed; the model is trained and cached at runtime.

## Dataset and prediction

The red-wine dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). It contains physicochemical measurements and a sensory quality score. The app labels records with score **6 or higher** as `GOOD QUALITY`; lower scores are labeled `NEEDS IMPROVEMENT`. The quality score is used as the training target and is never used as an input feature.

For predictions, use **Download sample input CSV** in the app to get the required feature columns, fill in sample measurements, and upload the CSV. The app displays and allows downloading the predictions.

## Repository layout

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py       # Streamlit Community Cloud entry point
├── wine-quality-prediction.zip
└── Wine_Quality_Prediction_Lab_.ipynb
```

The notebook and extracted project materials are retained as reference/classroom assets. The deployed Streamlit entry point is the root-level `streamlit_app.py`.

## Limitations

- Trained on one red-wine dataset; performance may not generalize to other wine types, producers, regions, or testing conditions.
- Quality ratings are subjective and the data is limited.
- Predictions are preliminary screening results, not final quality judgments.
