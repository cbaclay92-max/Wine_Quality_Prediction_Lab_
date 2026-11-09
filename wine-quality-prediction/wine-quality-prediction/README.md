# 🍷 Wine Quality Prediction System

## Problem
A small wine-production business needs a consistent, preliminary way to screen wine samples using recorded physicochemical measurements. This classroom prototype classifies a sample as **GOOD QUALITY** or **NEEDS IMPROVEMENT**.

## Dataset
UCI Machine Learning Repository – Wine Quality (red wine):
https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction rule
- `quality >= 6` → GOOD QUALITY (1)
- `quality < 6` → NEEDS IMPROVEMENT (0)

The original `quality` score is used only to create the target. It is **not** used as an input feature (to avoid target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest pipeline), joblib, Streamlit, GitHub.

## Installation and run steps
```bash
git clone https://github.com/YOUR-USERNAME/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
python train_model.py      # reads the red-wine CSV, trains the model, creates wine_quality.csv and wine_model.pkl
streamlit run app.py       # opens http://localhost:8501
```

## What the app shows
- **Overview (SAD):** problem, users, inputs, process, outputs, key limitation, prediction rule.
- **Dataset:** preview, missing/duplicate counts, quality distribution.
- **Model evaluation:** accuracy, precision, recall, F1-score and confusion matrix.
- **Predict:** enter one wine sample, or upload a CSV (column check, predictions, downloadable results).
- **Limitations:** responsible-use notes.

## Limitations
- Trained on one red-wine dataset; may not generalize to other wine types, producers, regions, or testing conditions.
- Predictions are preliminary screening only and are not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
- Quality scores are subjective sensory ratings, and the dataset is limited in size.
