from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained machine learning model.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    return y_pred, y_prob


def plot_confusion_matrix(model, X_test, y_test):

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(model, X_test, y_test):

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title("ROC Curve")
    plt.show()

