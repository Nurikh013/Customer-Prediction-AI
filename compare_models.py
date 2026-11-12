
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from prepare_data import (
    load_training_data,
    split_data,
    build_preprocessor
)


def main():
    X, y = load_training_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    models = [
        ("Baseline", None),
        ("Balanced", "balanced")
    ]

    for name, weight in models:
        print(f"\nTraining {name} model...")

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_fit)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight=weight
            ))
        ])

        pipeline.fit(X_fit, y_fit)

        predictions = pipeline.predict(X_val)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        print(f"\n{name} Model Results:")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1-score:  {f1:.2%}")


        if name == "Balanced":
            probabilities = pipeline.predict_proba(X_val)[:, 1]

            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

            print("\nBALANCED MODEL - THRESHOLD COMPARISON")

            for threshold in thresholds:
                new_predictions = (
                    probabilities >= threshold
                ).astype(int)

                new_precision = precision_score(
                    y_val, new_predictions, zero_division=0
                )

                new_recall = recall_score(
                    y_val, new_predictions, zero_division=0
                )

                new_f1 = f1_score(
                    y_val, new_predictions, zero_division=0
                )

                print(
                    f"Threshold: {threshold:.1f} | "
                    f"Precision: {new_precision:.2%} | "
                    f"Recall: {new_recall:.2%} | "
                    f"F1: {new_f1:.2%}"
                )



if __name__ == "__main__":
    main()
