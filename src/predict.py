import joblib
import pandas as pd


def load_model(model_path="../artifacts/best_model.pkl"):
    """
    Load the trained machine learning model.
    """
    return joblib.load(model_path)


def load_scaler(scaler_path="../artifacts/scaler.pkl"):
    """
    Load the saved scaler.
    """
    return joblib.load(scaler_path)


def predict(model, scaler, input_data):
    """
    Predict heart disease from new patient data.

    Parameters
    ----------
    model : trained ML model
    scaler : fitted StandardScaler
    input_data : pandas DataFrame

    Returns
    -------
    prediction, probability
    """

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict class
    prediction = model.predict(input_scaled)[0]

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability