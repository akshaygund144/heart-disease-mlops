from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Artifacts folder
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Load artifacts
model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read form inputs
    age = float(request.form["age"])
    sex = int(request.form["sex"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = int(request.form["fbs"])
    thalach = float(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    ca = float(request.form["ca"])

    cp = int(request.form["cp"])
    restecg = int(request.form["restecg"])
    slope = int(request.form["slope"])
    thal = int(request.form["thal"])

    # Create dataframe with zeros
    data = pd.DataFrame(0, index=[0], columns=feature_names)

    # Fill numerical features
    data["age"] = age
    data["sex"] = sex
    data["trestbps"] = trestbps
    data["chol"] = chol
    data["fbs"] = fbs
    data["thalach"] = thalach
    data["exang"] = exang
    data["oldpeak"] = oldpeak
    data["ca"] = ca

    # One-hot encoding
    if cp == 2:
        data["cp_2.0"] = 1
    elif cp == 3:
        data["cp_3.0"] = 1
    elif cp == 4:
        data["cp_4.0"] = 1

    if restecg == 1:
        data["restecg_1.0"] = 1
    elif restecg == 2:
        data["restecg_2.0"] = 1

    if slope == 2:
        data["slope_2.0"] = 1
    elif slope == 3:
        data["slope_3.0"] = 1

    if thal == 6:
        data["thal_6.0"] = 1
    elif thal == 7:
        data["thal_7.0"] = 1

    # Scale data
    scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        result = f"Heart Disease Detected (Probability: {probability:.2%})"
    else:
        result = f"No Heart Disease Detected (Probability: {(1-probability):.2%})"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)