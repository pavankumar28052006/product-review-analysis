import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import nltk
from nltk.corpus import stopwords


TEXT_COLUMN = "review"
TARGET_COLUMN = "sentiment"

DATA_PATH = Path("data") / "product_reviews.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_svm.pkl"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please create a 'data' folder and place your CSV as 'product_reviews.csv'."
        )

    df = pd.read_csv(DATA_PATH)

    if TEXT_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected columns '{TEXT_COLUMN}' and '{TARGET_COLUMN}' in the dataset. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with missing values in critical columns
    df = df[[TEXT_COLUMN, TARGET_COLUMN]].dropna()

    # Basic cleanup: strip whitespace
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    return df


def build_pipeline(stop_words):
    """
    Build an sklearn Pipeline that does:
    TF-IDF vectorization -> Linear SVM classifier.
    """
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words=stop_words,
                    max_df=0.9,
                    min_df=2,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "svm",
                SVC(kernel="linear", probability=True, random_state=42),
            ),
        ]
    )
    return pipeline


def train_and_evaluate():
    # Ensure NLTK stopwords are available
    try:
        stop_words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        stop_words = stopwords.words("english")

    df = load_dataset()

    # Create synthetic sentiment classes for demonstration
    # Subsample for performance
    SAMPLE_SIZE = 500
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    n = len(df)
    indices = np.arange(n)
    df[TARGET_COLUMN] = np.where(
        indices < n / 3,
        "Negative",
        np.where(indices < 2 * n / 3, "Positive", "Neutral"),
    )

    X = df[TEXT_COLUMN].values
    y = df[TARGET_COLUMN].values


    print("Label distribution in full dataset (after synthetic labelling):")
    print(pd.Series(y).value_counts())


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )


    unique_train_labels = np.unique(y_train)
    if unique_train_labels.shape[0] < 2:
        raise ValueError(
            f"Training split ended up with only one class: {unique_train_labels}. "
            "Please check the dataset or adjust the split."
        )

    pipeline = build_pipeline(stop_words)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    print("=== Sentiment Analysis using SVM ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train_and_evaluate()


