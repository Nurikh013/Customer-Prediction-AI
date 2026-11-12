
import streamlit as st

from predict import predict_customer



st.set_page_config(
    page_title="Customer Prediction AI",
    layout="centered"
)

st.title("Customer Prediction AI")

st.write(
    "A machine learning application for predicting "
    "bank term deposit subscriptions."
)

st.divider()

st.subheader("Project Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Customer Records", "45,211")

with col2:
    st.metric("Input Features", "15")

st.subheader("Machine Learning Model")

st.write("Algorithm: Logistic Regression")
st.write("Training data: 36,168 customer records")
st.write("Validation threshold: 0.60 (provisional)")


st.info(
    "Enter customer information below to predict whether "
    "a customer is likely to subscribe to a bank term deposit. "
    "Predictions are generated using our trained Logistic Regression model."
)


st.divider()


st.subheader("Customer Information")

# Basic customer information
age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=95,
    value=40
)

job = st.selectbox(
    "Customer Occupation",
    [
        "management", "technician", "blue-collar",
        "admin.", "services", "retired",
        "student", "self-employed", "entrepreneur",
        "unemployed", "housemaid", "unknown"
    ]
)

balance = st.number_input(
    "Account Balance",
    value=1000,
    step=100
)

# Additional customer information
with st.expander("Additional Customer Details", expanded=True):

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

    education = st.selectbox(
        "Education Level",
        ["secondary", "tertiary", "primary", "unknown"]
    )

    default = st.selectbox(
        "Credit Default",
        ["no", "yes"]
    )

    housing = st.selectbox(
        "Housing Loan",
        ["no", "yes"]
    )

    loan = st.selectbox(
        "Personal Loan",
        ["no", "yes"]
    )

    contact = st.selectbox(
        "Contact Method",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
    )

    campaign = st.number_input(
        "Current Campaign Contacts",
        min_value=1,
        value=1
    )

    pdays = st.number_input(
        "Days Since Previous Campaign Contact (-1 = none)",
        min_value=-1,
        value=-1
    )

    previous = st.number_input(
        "Previous Campaign Contacts",
        min_value=0,
        value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )

# Review entered information

if st.button("Predict Subscription"):


    customer = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }

    try:
        prediction, probability = predict_customer(customer)

        st.subheader("AI Prediction Result")

        st.metric(
            "Model Subscription Score",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

        if prediction == 1:
            st.success("Prediction: YES - Customer may subscribe")
        else:
            st.warning("Prediction: NO - Customer may not subscribe")

        st.caption(
            "Decision threshold: 60%. "
            "This model score is not a calibrated probability."
        )

        with st.expander("View Customer Details"):
            st.json(customer)

    except FileNotFoundError:
        st.error(
            "Trained model not found. Run python save_model.py."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")



st.caption("Customer Prediction AI | AI Engineering Portfolio")

