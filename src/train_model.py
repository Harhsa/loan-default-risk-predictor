import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from features import create_features


def main():
    # Load data
    data_path = "data/credit_data.xls"
    df = pd.read_excel(data_path)

    # Fix column headers
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)

    # Drop ID column
    df = df.drop(columns=["ID"])

    # Rename target
    df = df.rename(columns={"default payment next month": "default"})

    # Convert numeric columns
    df = df.apply(pd.to_numeric)

    # Feature engineering
    df = create_features(df)

    # Split features and target
    X = df.drop(columns=["default"])
    y = df["default"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # ✅ SAVE MODEL AND FEATURE NAMES (INSIDE main)
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

    print("Model saved to models/rf_model.pkl")
    print("Feature names saved to models/feature_names.pkl")


if __name__ == "__main__":
    main()
