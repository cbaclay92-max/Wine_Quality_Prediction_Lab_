# Wine Quality Prediction System

A Section B Data Mining Laboratory prototype that uses physicochemical measurements to classify red-wine samples as **GOOD QUALITY** or **NEEDS IMPROVEMENT**. It follows the lab's binary target rule: quality score **≥ 6** is GOOD QUALITY; score **< 6** is NEEDS IMPROVEMENT. The original `quality` column is used only to create the target and is excluded from model input features.

> **Responsible use:** This is a classroom prototype based on a public dataset. Predictions are preliminary screening results, not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## SAD mini-analysis

| SAD element | Description |
| --- | --- |
| Problem | Provide a consistent preliminary screen using recorded wine measurements. |
| Users | Quality-control staff, production supervisor, or data analyst. |
| Inputs | The 11 physicochemical measurement columns in the red-wine dataset. |
| Process | Validate uploaded columns → impute missing values → classify with a Random Forest pipeline. |
| Outputs | GOOD QUALITY / NEEDS IMPROVEMENT label and model evaluation summary. |
| Limitation | The model uses one red-wine dataset and may not generalize to other wines or conditions. |

## Files

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py     # Streamlit app and upload-prediction flow
├── train_model.py       # download, train, evaluate, and save model
└── wine-quality-prediction/  # original extracted classroom project assets
```

The app can run without pre-generated artifacts: it loads the dataset and trains its model at startup. `train_model.py` is included to match the lab exercise and to create `wine_quality.csv` and `wine_model.pkl` locally.

## Requirements and local setup

Use Python 3.10 or later. Clone this repository and install dependencies:

```bash
git clone https://github.com/cbaclay92-max/Wine_Quality_Prediction_Lab_.git
cd Wine_Quality_Prediction_Lab_
python -m venv .venv
```

Activate the virtual environment (`source .venv/bin/activate` on macOS/Linux, or `.venv\\Scripts\\Activate.ps1` in PowerShell), then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

The training script downloads the UCI red-wine data and saves `wine_quality.csv` plus `wine_model.pkl` in the repository root. It prints accuracy, a confusion matrix, and precision/recall/F1-score for both classes. The model uses a stratified 80/20 holdout split, `random_state=42`, median imputation, scaling, and a 150-tree Random Forest with balanced class weights.

## Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
2. Select this repository, branch `main`, and main file path `streamlit_app.py`.
3. Deploy. The root `requirements.txt` contains the dependencies.

For this public deployment, no secrets are required. The app searches for `wine_quality.csv` or `winequality-red.csv` beside the app, then for a matching file in `wine-quality-prediction.zip`, and finally downloads the UCI red-wine CSV. It supports comma- and semicolon-delimited uploads; the semicolon-delimited UCI `winequality-red.csv` may include the `quality` target, which the app ignores for prediction.

## Dataset and predictions

Dataset source: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality). The app shows dataset details and quality distribution, evaluates the Random Forest using a stratified holdout split, accepts uploaded CSV samples, validates required measurement columns, shows predictions, and offers a results download. Use **Download sample input CSV** to get a correctly formatted prediction template.

## Limitations

- The model is trained on one red-wine dataset; it may not generalize to other wine types, producers, regions, or testing conditions.
- Quality scores are subjective sensory ratings, and the dataset is limited.
- Low recall for GOOD QUALITY could mean qualifying samples are missed; accuracy alone should not be used to judge performance.
- The prototype does not replace professional quality-control procedures or food-safety assessments.
