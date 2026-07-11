import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_and_save(output_path=None):

    output_path = os.path.join(BASE_DIR, "data", "raw", "heart.csv")

    df = pd.read_csv(URL, names=COLUMNS, na_values="?")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved {df.shape[0]} rows to {output_path}")

    return df


if __name__ == "__main__":
    download_and_save()
