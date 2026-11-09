# 🍷 Wine Quality Prediction System

A classroom decision-support prototype that classifies red-wine samples as **GOOD QUALITY** or **NEEDS IMPROVEMENT** from physicochemical measurements.

> **Responsible use:** This is a classroom prototype based on a public dataset. Its predictions are not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.

## Project files

```text
.
├── app.py                 # Streamlit dashboard and CSV prediction flow
├── train_model.py         # Downloads/loads data, trains and evaluates the model
├── requirements.txt       # Python dependencies
├── wine_quality.csv       # Created by training (not required in Git)
└── wine_model.pkl         # Created by training (not required in Git)
```

The repository also includes the original lab notebook and ZIP archive. The runnable scripts are at the repository root for convenience.

## Requirements

- Python 3.10 or later
- Internet access the first time you train/run, unless `wine_quality.csv` is already present

## Run locally

From the repository folder:

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Train and evaluate the model (this downloads the UCI red-wine CSV if it is not already in the folder):

```bash
python train_model.py
```

Start the app:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually <http://localhost:8501>. The app can also prepare the CSV and train the model automatically on first launch if the generated files are missing.

## Using the app

1. Download the sample input CSV from the dashboard, or prepare a CSV with the 11 physicochemical feature columns shown in the sample.
2. Upload the CSV. The app checks for required fields and numeric values before prediction.
3. Review the labels and download the results CSV.

The original `quality` column is the answer key used only to create the training target (`quality >= 6`). It is never a model input. If an uploaded file includes `quality`, the app excludes it from prediction features.

## Dataset and model

- **Dataset:** [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), red wine CSV.
- **Target rule:** quality ≥ 6 → GOOD QUALITY; quality < 6 → NEEDS IMPROVEMENT.
- **Preprocessing/model:** duplicate removal, median imputation, standard scaling, and a class-balanced Random Forest with 150 trees.
- **Evaluation:** stratified 80/20 train/test split (`random_state=42`); the training script prints accuracy, confusion matrix, and precision/recall/F1 for each class.

Performance on the reference split is about 0.77 accuracy. Results can vary with data or library versions, and accuracy should not be considered by itself.

## SAD mini-analysis

- **Problem:** provide a consistent preliminary screen from recorded wine measurements.
- **Users:** quality-control staff, production supervisors, or data analysts.
- **Inputs:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, and alcohol.
- **Process:** validate columns and numeric values → preprocess → apply the trained classifier.
- **Outputs:** predicted quality category and evaluation summary.
- **Limitation:** trained on a particular red-wine dataset; it may not generalize to other wine types, producers, regions, or testing conditions.

## Reflection prompts

1. The target is the binary category derived from `quality`; excluding the original score prevents target leakage (giving the model the answer).
2. Good-quality recall is the fraction of truly good samples that the model identifies as good. Low recall means some genuinely good samples may be missed.
3. Producer, region, process, and measurement differences can shift the data distribution and reduce performance.
4. The model is a preliminary screen trained on limited historical data, not a validated food-safety or quality-control instrument.
5. More representative samples across producers and vintages, reliable laboratory measurements, external validation, and ongoing monitoring could improve confidence.

## GitHub deployment note

For Streamlit Community Cloud, select this repository and set `app.py` as the entry point. The app needs outbound access to the UCI dataset on its first run unless the generated CSV and model artifacts are committed or otherwise provided. Generated artifacts are excluded by `.gitignore` by default.
