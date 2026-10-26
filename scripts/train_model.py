"""Train, compare, tune, and save the customer churn classification pipeline."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"


def load_and_clean_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Telco-Customer-Churn.csv in data/raw/."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    df = df.drop_duplicates().copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Blank TotalCharges for zero-tenure customers represent no accumulated charges.
    zero_tenure_mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[zero_tenure_mask, "TotalCharges"] = 0
    return df


def main() -> None:
    df = load_and_clean_data()
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn", "customerID"])

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
    }

    results = {}
    print("Comparing baseline models with 5-fold cross-validation...")
    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", estimator)]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        results[name] = {
            "Mean F1": scores["test_f1"].mean(),
            "Mean ROC-AUC": scores["test_roc_auc"].mean(),
        }

    results_df = pd.DataFrame(results).T.sort_values("Mean F1", ascending=False)
    print("\nCross-validation results:")
    print(results_df.round(4).to_string())
    best_name = results_df.index[0]
    print(f"\nSelected model for tuning: {best_name}")

    param_grids = {
        "Logistic Regression": {"classifier__C": [0.1, 1, 10]},
        "Decision Tree": {
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2],
        },
    }

    tuning_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", models[best_name]),
        ]
    )
    search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid=param_grids[best_name],
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    print("\nTuning selected model...")
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(X_test)
    probabilities = best_pipeline.predict_proba(X_test)[:, list(best_pipeline.classes_).index(1)]

    print("Best parameters:", search.best_params_)
    print(f"Best cross-validation F1: {search.best_score_:.4f}")
    print(f"Test F1: {f1_score(y_test, predictions):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nSaved model pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
