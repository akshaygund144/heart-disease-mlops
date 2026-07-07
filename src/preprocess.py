import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path):
    """
    Load the processed dataset.
    """
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Perform preprocessing:
    - Split features and target
    - One-hot encode categorical variables
    - Train-test split
    - Scale numerical features
    - Save scaler
    """

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # One-hot encode categorical columns
    X = pd.get_dummies(
        X,
        columns=["cp", "restecg", "slope", "thal"],
        drop_first=True,
        dtype=int
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "../artifacts/scaler.pkl")

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )

