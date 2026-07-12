from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------

model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

print("Loaded Feature Names:")
print(feature_names)
print("Model expects:", model.n_features_in_)
print("Scaler expects:", len(scaler.feature_names_in_))

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0",
    description="Heart Disease Prediction using Machine Learning"
)

# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class Patient(BaseModel):
    age: int
    sex: int
    trestbps: float
    chol: float
    fbs: int
    thalach: float
    exang: int
    oldpeak: float
    ca: int
    cp: int
    restecg: int
    slope: int
    thal: int

# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running!"
    }

# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(patient: Patient):

    try:

        # Create empty dataframe with EXACT training columns
        sample = pd.DataFrame(
            [[0.0] * len(feature_names)],
            columns=feature_names
        )

        # Numerical features
        sample.loc[0, "age"] = patient.age
        sample.loc[0, "sex"] = patient.sex
        sample.loc[0, "trestbps"] = patient.trestbps
        sample.loc[0, "chol"] = patient.chol
        sample.loc[0, "fbs"] = patient.fbs
        sample.loc[0, "thalach"] = patient.thalach
        sample.loc[0, "exang"] = patient.exang
        sample.loc[0, "oldpeak"] = patient.oldpeak
        sample.loc[0, "ca"] = patient.ca

        # One-hot encoding
        cp_col = f"cp_{float(patient.cp):.1f}"
        if cp_col in sample.columns:
            sample.loc[0, cp_col] = 1

        restecg_col = f"restecg_{float(patient.restecg):.1f}"
        if restecg_col in sample.columns:
            sample.loc[0, restecg_col] = 1

        slope_col = f"slope_{float(patient.slope):.1f}"
        if slope_col in sample.columns:
            sample.loc[0, slope_col] = 1

        thal_col = f"thal_{float(patient.thal):.1f}"
        if thal_col in sample.columns:
            sample.loc[0, thal_col] = 1

        print("=" * 60)
        print("Input Sample")
        print(sample)

        # Scale
        sample_scaled = scaler.transform(sample)

        # Convert back to DataFrame
        sample_scaled = pd.DataFrame(
            sample_scaled,
            columns=feature_names
        )

        print("=" * 60)
        print("Scaled Shape:", sample_scaled.shape)

        # Prediction
        prediction = int(model.predict(sample_scaled)[0])

        probability = float(
            model.predict_proba(sample_scaled)[0][1]
        )

        return {
            "prediction": prediction,
            "probability": round(probability, 4)
        }

    except Exception as e:

        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }