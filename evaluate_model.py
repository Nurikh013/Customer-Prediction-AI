
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from prepare_data import load_training_data, split_data


def main():
    # Load our dataset
    X, y = load_training_data()

    # Recreate the same train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Load the model used by Streamlit
    model_path = (
        Path(__file__).parent
        / "models"
        / "customer_model.joblib"
    )

    model_package = joblib.load(model_path)

    pipeline = model_package["pipeline"]
    threshold = model_package["threshold"]

    # Predict using the saved model
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\nSAVED MODEL - TEST RESULTS")

    print(f"Threshold: {threshold:.2f}")
    print(f"Test records: {len(X_test)}")

    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
