
from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path(__file__).parent / "models" / "customer_model.joblib"


def predict_customer(customer):
    # Load our trained model
    model_package = joblib.load(MODEL_PATH)

    pipeline = model_package["pipeline"]
    threshold = model_package["threshold"]
    feature_columns = model_package["feature_columns"]

    # Convert customer information into a DataFrame
    customer_df = pd.DataFrame(
        [customer],
        columns=feature_columns
    )

    # Calculate probability of subscription
    probability = pipeline.predict_proba(customer_df)[0][1]

    # Apply our selected threshold
    prediction = int(probability >= threshold)

    return prediction, probability


if __name__ == "__main__":
    example_customer = {
        "age": 40,
        "job": "management",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1000,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "jan",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }

    prediction, probability = predict_customer(example_customer)

    print("Prediction:", prediction)
    print(f"Subscription probability: {probability:.2%}")
