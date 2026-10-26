# Customer Churn Prediction — Streamlit + FastAPI

This project uses a Scikit-learn pipeline to predict customer churn. Streamlit provides the user interface; FastAPI exposes the prediction endpoint; Pydantic validates incoming request data.

## Folder structure

```text
customer_churn_project/
├── app.py                              # Streamlit user interface
├── requirements.txt                    # Python dependencies
├── README.md                           # Setup and usage guide
├── .gitignore                          # Files not intended for version control
├── backend/
│   ├── __init__.py
│   └── main.py                          # FastAPI + Pydantic endpoints
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv      # Original dataset
├── models/
│   └── churn_pipeline.joblib             # Trained Scikit-learn pipeline
├── notebooks/
│   └── Telco-Customer-Churn.ipynb        # EDA, preprocessing, and training work
└── scripts/
    └── train_model.py                    # Rebuilds and saves the model
```

## 1. Open the project

Extract the project ZIP, then open the extracted `customer_churn_project` folder in VS Code. Open a terminal in that folder (the directory containing `app.py` and `requirements.txt`).

## 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, you can still install packages using `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use that same interpreter to run the commands below.

The included model was verified with Scikit-learn 1.8.0. Keep the Scikit-learn version aligned with the version used to train a serialized model.

## 3. Start the FastAPI backend

In terminal 1, from the project root:

```powershell
python -m uvicorn backend.main:app --reload
```

API pages:

- Health check: http://127.0.0.1:8000/health
- Interactive Swagger docs: http://127.0.0.1:8000/docs
- Root status: http://127.0.0.1:8000/

Use the Swagger docs to test `POST /predict`. Pydantic validates the feature names, types, categories, and numeric ranges before the model is called.

## 4. Start the Streamlit frontend

Keep terminal 1 running. Open terminal 2 in the same project root and run:

```powershell
python -m streamlit run app.py
```

Streamlit normally opens at http://localhost:8501. Submit a customer profile; Streamlit sends it to the FastAPI endpoint at `http://127.0.0.1:8000/predict`.

## 5. Retrain the model (optional)

The supplied model is already in `models/`. To reproduce the training process from the CSV, run from the project root:

```powershell
python scripts/train_model.py
```

The script cleans the data, compares Logistic Regression, Decision Tree, and Random Forest with cross-validation, tunes the leading model using GridSearchCV, evaluates it on a held-out test set, and writes the new `models/churn_pipeline.joblib` file.

## Common errors

- **`Could not connect to FastAPI`:** start the API in terminal 1 before submitting the Streamlit form.
- **Model file not found:** ensure `models/churn_pipeline.joblib` exists. Retrain with `python scripts/train_model.py` if necessary.
- **Model deserialization error:** use the same Python and Scikit-learn versions used to create the `.joblib` file.
- **HTTP 422:** the input request did not pass Pydantic validation; inspect the error details shown in Streamlit or `/docs`.
- **HTTP 500 prediction error:** check the model's feature columns and the backend terminal traceback.

## Important note

This is an educational ML project. The estimated churn score is not guaranteed to be calibrated as a real-world probability and should not be used as the sole basis for business decisions.
