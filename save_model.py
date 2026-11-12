
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from prepare_data import (
    load_training_data,
    split_data,
    build_preprocessor
)


def main():
    # Load customer data
    X, y = load_training_data()

    # Keep the test data separate
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create the preprocessing system
    preprocessor = build_preprocessor(X_train)

    # Create our balanced model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced"
        ))
    ])

    # Train using the training data only
    print("\nTraining the final model...")
    model.fit(X_train, y_train)

    # Set our selected prediction threshold
    threshold = 0.60

    # Specify where to save the model
    model_path = (
        Path(__file__).parent
        / "models"
        / "customer_model.joblib"
    )

    # Make sure the models folder exists
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model and configuration together
    model_package = {
        "pipeline": model,
        "threshold": threshold,
        "feature_columns": X_train.columns.tolist()
    }

    joblib.dump(model_package, model_path)

    print("\nModel saved successfully!")
    print(f"Location: {model_path}")
    print(f"Prediction threshold: {threshold}")
    print(f"Input features: {len(X_train.columns)}")


if __name__ == "__main__":
    main()
