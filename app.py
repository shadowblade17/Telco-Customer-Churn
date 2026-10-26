import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate whether the customer is likely to churn. "
    "Streamlit sends the request to a FastAPI backend."
)

with st.form("churn_form"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with right:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=150.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Keep the payload as ordinary Python strings, ints, and floats for JSON.
    customer_payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        with st.spinner("Requesting prediction from FastAPI..."):
            response = requests.post(API_URL, json=customer_payload, timeout=15)

        if response.status_code == 422:
            st.error("The backend rejected the input. Check the validation details below.")
            st.json(response.json())
        else:
            response.raise_for_status()
            result = response.json()

            if result["prediction"] == 1:
                st.warning("Prediction: Customer is likely to churn.")
            else:
                st.success("Prediction: Customer is likely to stay.")

            st.metric(
                "Model-estimated churn score",
                f"{result['churn_probability']:.1%}",
            )
            st.caption(
                "Educational model output only. This score is not a guarantee "
                "of future behavior and may not be calibrated as a real-world probability."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. Start the backend in another terminal with:\n\n"
            "python -m uvicorn backend.main:app --reload"
        )
    except requests.exceptions.Timeout:
        st.error("The API request timed out. Check whether the backend is still running.")
    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", str(exc))
        except ValueError:
            detail = str(exc)
        st.error(f"Backend error: {detail}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
