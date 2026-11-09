from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "wine_quality.csv"
DEFAULT_MODEL_PATH = ROOT / "wine_model.pkl"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET_COLUMN = "quality"


def load_data(data_path: Path) -> pd.DataFrame:
    """Load a prepared CSV or retrieve the official red-wine CSV from UCI."""
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        print("Local dataset not found; downloading the UCI red-wine dataset...")
        data = pd.read_csv(DATA_URL, sep=";")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    return data


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Train, evaluate, and persist the leakage-safe binary classifier."""
    data = load_data(Path(data_path))
    print("Rows and columns:", data.shape)
    print("Missing values per column:\n", data.isna().sum())
    print("Duplicate rows:", int(data.duplicated().sum()))
    print("Quality distribution:\n", data[TARGET_COLUMN].value_counts().sort_index())

    data = data.drop_duplicates().copy()
    y = (data[TARGET_COLUMN] >= 6).astype(int)
    # Exclude quality: it is used only to derive y, never as a feature.
    X = data.drop(columns=[TARGET_COLUMN])
    if X.empty or y.nunique() != 2:
        raise ValueError("Training data must include features and both target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
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
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Needs improvement", "Good quality"],
            zero_division=0,
        )
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the wine quality classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to wine_quality.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for the trained model")
    args = parser.parse_args()
    train_and_evaluate(data_path=args.data, model_path=args.model)


if __name__ == "__main__":
    main()
