"""Train and export the spam classifier without the biased Email_Length feature."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/email_spam_detection.csv")
MODELS_DIR = Path("models")
URGENT_SUBJECTS = {"Security Alert", "Account Verification", "Win Prize"}
CATEGORICAL_FEATURES = ["email_domain", "Subject"]
CONTINUOUS_FEATURES = ["Num_Links", "Num_Special_Chars", "Capital_Words", "subject_length"]
BINARY_FEATURES = ["Has_Attachment", "is_urgent_subject"]
MODEL_FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES + BINARY_FEATURES


def add_engineered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["email_domain"] = frame["Sender_Email"].apply(
        lambda value: value.split("@")[-1] if isinstance(value, str) and "@" in value else "unknown"
    )
    frame["subject_length"] = frame["Subject"].astype(str).str.len()
    frame["is_urgent_subject"] = frame["Subject"].isin(URGENT_SUBJECTS).astype(int)
    return frame


def main() -> None:
    data = add_engineered_columns(pd.read_csv(DATA_PATH))
    X = data[MODEL_FEATURES]
    y = data["Spam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = pipeline.predict(X_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "spam_pipeline.joblib")

    metrics = {
        "best_model": "logistic_regression_without_email_length",
        "excluded_features": ["Email_Length"],
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "roc_curve": {"fpr": roc_curve(y_test, probabilities)[0].tolist(), "tpr": roc_curve(y_test, probabilities)[1].tolist()},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    coefficients = pipeline.named_steps["clf"].coef_[0]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    importance = sorted(zip(names, np.abs(coefficients).tolist()), key=lambda item: -item[1])
    (MODELS_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    print(f"Saved model without Email_Length. Test ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
