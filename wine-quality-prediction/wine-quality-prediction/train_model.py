# train_model.py - Activity A + B
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# ---------- Activity A: retrieve and inspect ----------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
source = "winequality-red.csv" if os.path.exists("winequality-red.csv") else url
df = pd.read_csv(source, sep=";")
df.to_csv("wine_quality.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Quality distribution:")
print(df["quality"].value_counts().sort_index())

# ---------- Activity B: preprocess, train, evaluate ----------
# Remove duplicate records and define the binary target
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)  # 1 = Good; 0 = Needs improvement

# Do not include the original quality score as a predictor
X = df.drop(columns=["quality"])

# Split the data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Preprocessing and classification in one pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print(classification_report(
    y_test, predictions,
    target_names=["Needs improvement", "Good quality"],
    zero_division=0
))

joblib.dump(model, "wine_model.pkl")
print("Saved wine_model.pkl")
