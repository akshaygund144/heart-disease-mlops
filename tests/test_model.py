import os
import sys
import joblib
import pandas as pd

# -------------------------------------------------------
# Add project root to Python path
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def test_model_prediction():
    """
    Test that the trained model can make predictions
    and returns a valid probability.
    """

    # -------------------------------------------------------
    # Load saved artifacts
    # -------------------------------------------------------
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))

    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    # -------------------------------------------------------
    # Create sample input
    # -------------------------------------------------------
    sample = pd.DataFrame(0, index=[0], columns=feature_names)

    # Numerical features
    sample.loc[0, "age"] = 63
    sample.loc[0, "sex"] = 1
    sample.loc[0, "trestbps"] = 145
    sample.loc[0, "chol"] = 233
    sample.loc[0, "fbs"] = 1
    sample.loc[0, "thalach"] = 150
    sample.loc[0, "exang"] = 0
    sample.loc[0, "oldpeak"] = 2.3
    sample.loc[0, "ca"] = 0

    # -------------------------------------------------------
    # One-hot encoded categorical features
    # -------------------------------------------------------
    if "cp_2" in sample.columns:
        sample.loc[0, "cp_2"] = 1

    if "cp_3" in sample.columns:
        sample.loc[0, "cp_3"] = 0

    if "cp_4" in sample.columns:
        sample.loc[0, "cp_4"] = 0

    if "restecg_1" in sample.columns:
        sample.loc[0, "restecg_1"] = 1

    if "slope_2" in sample.columns:
        sample.loc[0, "slope_2"] = 1

    if "slope_3" in sample.columns:
        sample.loc[0, "slope_3"] = 0

    if "thal_6" in sample.columns:
        sample.loc[0, "thal_6"] = 1

    if "thal_7" in sample.columns:
        sample.loc[0, "thal_7"] = 0

    # -------------------------------------------------------
    # Scale input
    # -------------------------------------------------------
    sample_scaled = scaler.transform(sample)

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    # -------------------------------------------------------
    # Assertions
    # -------------------------------------------------------
    assert prediction in [0, 1]
    assert 0.0 <= probability <= 1.0

