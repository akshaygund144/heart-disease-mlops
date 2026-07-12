import os
import joblib
import pandas as pd

from preprocess import load_data, preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "heart_processed.csv"
)

# ----------------------------------------------------
# Load & preprocess data
# ----------------------------------------------------

df = load_data(DATA_PATH)

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = preprocess_data(df)

print("Training features:", X_train.shape[1])
print(X_train.columns.tolist())

# ----------------------------------------------------
# Hyperparameter tuning
# ----------------------------------------------------

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ----------------------------------------------------
# Save model
# ----------------------------------------------------

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

joblib.dump(
    best_rf,
    os.path.join(ARTIFACTS_DIR, "best_model.pkl")
)

joblib.dump(
    best_rf,
    os.path.join(ARTIFACTS_DIR, "random_forest.pkl")
)

print("Model saved successfully.")


import os
print(os.getcwd())