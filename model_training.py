
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


from prepare_data import (
    load_training_data,
    split_data,
    build_preprocessor
)


def main():
    # Load customer data
    X, y = load_training_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create preprocessing system
    preprocessor = build_preprocessor(X_train)

    # Create machine learning pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        ))
    ])

    # Train the model
    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    print("Model training completed successfully!")

    # Test predictions on five training examples
    predictions = model.predict(X_train.head(5))

    print("\nFirst 5 predictions:")
    print(predictions)

    print("\nActual answers:")
    print(y_train.head(5).to_numpy())

    # Predict outcomes on unseen test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nMODEL EVALUATION RESULTS")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))



if __name__ == "__main__":
    main()
