from functools import lru_cache
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict whether a telecom customer is likely to churn. "
        "Incoming customer features are validated with Pydantic."
    ),
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    """The 19 feature columns expected by the trained model."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=150, allow_inf_nan=False)
    TotalCharges: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    prediction_label: Literal["Likely to stay", "Likely to churn"]
    churn_probability: float = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def load_model():
    """Load the saved Scikit-learn pipeline once per server process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python scripts/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    model_ready = MODEL_PATH.is_file()
    return {
        "status": "ok" if model_ready else "model_missing",
        "model_file_found": model_ready,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not load the serialized model")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load the model. Check the model file and the "
                "Python/Scikit-learn versions used to create it."
            ),
        ) from exc

    input_df = pd.DataFrame([customer.model_dump()])

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        class_labels = list(model.classes_)
        churn_index = class_labels.index(1)
        churn_probability = float(probabilities[churn_index])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Confirm that the saved pipeline expects "
                "the 19 input features defined by this API."
            ),
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        prediction_label="Likely to churn" if prediction == 1 else "Likely to stay",
        churn_probability=round(churn_probability, 4),
    )
