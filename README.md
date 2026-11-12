
# Customer Prediction AI

An end-to-end machine learning application that predicts whether a bank customer is likely to subscribe to a term deposit.

Built with Python, scikit-learn, Streamlit, and FastAPI.

## Project Overview

This project uses historical bank marketing data to train a binary classification model.

The application allows users to enter customer information and receive a prediction from a trained Logistic Regression model.

## Dataset

Source: UCI Machine Learning Repository - Bank Marketing

https://archive.ics.uci.edu/dataset/222/bank+marketing

- 45,211 customer records
- 15 input features used for prediction
- Target: Subscription (Yes / No)
- Training records: 36,168
- Testing records: 9,043

The `duration` feature was excluded to avoid using information unavailable before a phone call ends.

## Technologies

- Python
- Pandas and NumPy
- scikit-learn
- Matplotlib
- Streamlit
- FastAPI
- Pydantic
- pytest
- joblib

## Machine Learning Pipeline

1. Load and explore the dataset.
2. Analyze missing values and class imbalance.
3. Split data into training and testing sets.
4. Encode categorical features using OneHotEncoder.
5. Scale numerical features using StandardScaler.
6. Train Logistic Regression models.
7. Compare baseline and balanced models.
8. Select a prediction threshold using validation data.
9. Evaluate the saved model on test data.
10. Serve predictions through Streamlit and FastAPI.

## Model Performance

The saved model uses Balanced Logistic Regression with a decision threshold of 0.60.

### Test Results

| Metric | Result |
|---|---|
| Accuracy | 85.47% |
| Precision | 39.92% |
| Recall | 47.92% |
| F1-score | 43.56% |

### Confusion Matrix

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 7,222 | 763 |
| Yes | 551 | 507 |

The balanced model improved recall compared with the original baseline, but reduced precision.

Because the model uses class weighting, its output score should not be interpreted as a calibrated real-world probability.

## Features

- Customer information input form
- Machine learning prediction
- Model subscription score
- Interactive Streamlit interface
- FastAPI prediction endpoint
- Input validation and error handling
- Automated unit and integration tests

## Installation

Clone the repository and create a Python virtual environment.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the Bank Marketing dataset from UCI and place `bank-full.csv` inside the `data/` folder.

Train and save the model:

```bash
python save_model.py
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Run the FastAPI server:

```bash
python -m uvicorn api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

## Testing

Run all automated tests:

```bash
python -m pytest tests/ -v
```

Seven automated tests currently pass in the local development environment.

## Project Structure

- `app.py` - Streamlit user interface
- `api.py` - FastAPI backend
- `predict.py` - Prediction function
- `prepare_data.py` - Data preparation
- `model_training.py` - Initial model training
- `compare_models.py` - Model comparison
- `save_model.py` - Train and save the selected model
- `evaluate_model.py` - Saved model evaluation
- `train.py` - Exploratory data analysis
- `tests/` - Automated tests
- `data/` - Local dataset files
- `models/` - Saved machine learning models

## Limitations

- The model is trained on historical marketing data from one dataset.
- Predictions may not generalize to other banks or populations.
- Model scores are not calibrated probabilities.
- The application is an educational prototype, not a production banking decision system.
- Dataset CSV files and trained model artifacts are excluded from Git and must be generated locally.

## Future Improvements

- Compare additional machine learning algorithms.
- Evaluate probability calibration.
- Add more comprehensive automated tests.
- Improve the application interface.
- Deploy the application online.




## Author

**Irisbaev Nuriddin**

AI & Machine Learning Developer

GitHub: [Nurikh013](https://github.com/Nurikh013)

