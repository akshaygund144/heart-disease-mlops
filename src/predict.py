import os
import joblib
import pandas as pd

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# ==========================================================
# Load Saved Model
# ==========================================================


def load_model():
    """
    Load the trained machine learning model.
    """
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
    return joblib.load(model_path)


# ==========================================================
# Load Saved Scaler
# ==========================================================


def load_scaler():
    """
    Load the saved StandardScaler.
    """
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    return joblib.load(scaler_path)


# ==========================================================
# Prediction Function
# ==========================================================


def predict(model, scaler, input_data):
    """
    Predict heart disease from new patient data.

    Parameters
    ----------
    model : Trained machine learning model
    scaler : Fitted StandardScaler
    input_data : pandas DataFrame

    Returns
    -------
    prediction : int
        0 = No Heart Disease
        1 = Heart Disease

    probability : float
        Probability of heart disease
    """

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Convert back to DataFrame to preserve feature names
    input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability


# ==========================================================
# Test Prediction
# ==========================================================

if __name__ == "__main__":

    # Load saved model and scaler
    model = load_model()
    scaler = load_scaler()

    # Sample patient (18 features)
    sample = pd.DataFrame(
        [[63, 1, 145, 233, 1, 150, 0, 2.3, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]],
        columns=[
            "age",
            "sex",
            "trestbps",
            "chol",
            "fbs",
            "thalach",
            "exang",
            "oldpeak",
            "ca",
            "cp_2.0",
            "cp_3.0",
            "cp_4.0",
            "restecg_1.0",
            "restecg_2.0",
            "slope_2.0",
            "slope_3.0",
            "thal_6.0",
            "thal_7.0",
        ],
    )

    prediction, probability = predict(model, scaler, sample)

    print("=" * 50)

    if prediction == 1:
        print("Prediction : Heart Disease Detected")
    else:
        print("Prediction : No Heart Disease")

    print(f"Probability : {probability:.4f}")

    print("=" * 50)
