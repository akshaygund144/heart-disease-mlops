import sys
import os
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_data


def test_preprocess_data():
    """
    Test preprocessing pipeline.
    """

    data = pd.DataFrame(
        {
            "age": [63, 67, 70, 55, 60, 58, 62, 59, 65, 61],
            "sex": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "cp": [1, 2, 3, 4, 1, 2, 3, 4, 2, 3],
            "trestbps": [145, 160, 130, 120, 140, 135, 150, 128, 138, 142],
            "chol": [233, 286, 250, 210, 240, 220, 260, 230, 245, 255],
            "fbs": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "restecg": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "thalach": [150, 108, 140, 170, 155, 145, 160, 150, 148, 152],
            "exang": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "oldpeak": [2.3, 1.5, 0.5, 0.0, 1.2, 0.8, 2.0, 1.0, 0.4, 1.6],
            "slope": [1, 2, 3, 2, 1, 2, 3, 1, 2, 3],
            "ca": [0, 1, 2, 0, 1, 0, 2, 1, 0, 2],
            "thal": [3, 6, 7, 3, 6, 7, 3, 6, 7, 3],
            "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler) = (
        preprocess_data(data)
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None