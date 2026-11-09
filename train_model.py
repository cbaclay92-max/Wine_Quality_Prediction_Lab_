"""Train and evaluate the Wine Quality Prediction System model.

Run from the repository root with: python train_model.py
This downloads the UCI red-wine dataset, writes wine_quality.csv and
wine_model.pkl, and prints metrics for a stratified holdout test set.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "wine_quality.csv"
MODEL_FILE = ROOT / "wine_model.pkl"


def build_model() -> Pipeline:
    """Build the lab's preprocessing and Random Forest pipeline."""
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


def main() -> None:
    """Download data, train/evaluate the model, and save the artifacts."""
    df = pd.read_csv(DATA_URL, sep=";")
    df.to_csv(DATA_FILE, index=False)

    print("Rows and columns:", df.shape)
    print(df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nQuality distribution:\n", df["quality"].value_counts().sort_index())

    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    # Exclude the original quality value to prevent target leakage.
    X = df.drop(columns=["quality"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
    print(
        "\nClassification report:\n",
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Needs improvement", "Good quality"],
            zero_division=0,
        ),
    )

    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")
    print(
        "The saved model was fit on the training split, matching the app's "
        "holdout evaluation workflow."
    )


if __name__ == "__main__":
    main()
