

from fastapi import FastAPI, HTTPException


from pydantic import BaseModel, Field
from predict import predict_customer

from typing import Literal



app = FastAPI(
    title="Customer Prediction AI API",
    version="1.0.0"
)



class CustomerInput(BaseModel):
    age: int = Field(ge=18, le=95)

    job: Literal[
        "management", "technician", "blue-collar",
        "admin.", "services", "retired",
        "student", "self-employed", "entrepreneur",
        "unemployed", "housemaid", "unknown"
    ]

    marital: Literal["married", "single", "divorced"]

    education: Literal[
        "primary", "secondary", "tertiary", "unknown"
    ]

    default: Literal["yes", "no"]

    balance: int

    housing: Literal["yes", "no"]

    loan: Literal["yes", "no"]

    contact: Literal["cellular", "telephone", "unknown"]

    day: int = Field(ge=1, le=31)

    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    campaign: int = Field(ge=1)

    pdays: int = Field(ge=-1)

    previous: int = Field(ge=0)

    poutcome: Literal[
        "unknown", "failure", "other", "success"
    ]




@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Customer Prediction AI"
    }



@app.post("/predict")
def predict_subscription(customer: CustomerInput):
    customer_data = customer.model_dump()

    try:
        prediction, score = predict_customer(customer_data)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable."
        ) from error

    return {
        "prediction": prediction,
        "label": "yes" if prediction == 1 else "no",
        "model_score": round(float(score), 4)
    }

