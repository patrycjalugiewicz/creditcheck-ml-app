# pylint: disable=invalid-name
"""
Model treningowy dla aplikacji CreditCheck.
Porównuje kilka modeli klasyfikacyjnych i zapisuje najlepszy model do pliku.
"""
import os
import sys

import joblib
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from data.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_data():
    """Wczytuje dane treningowe."""
    df = pd.read_csv(
        "../data/raw/train_u6lujuX_CVtuZ9i.csv"
    )

    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    return df


def split_data(df):
    """Dzieli dane na zbiór treningowy i testowy."""

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Trenuje modele i wybiera najlepszy."""

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print("\n" + "=" * 40)
        print(name)
        print("=" * 40)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = model

    return best_model, best_score


def save_model(model):
    """Zapisuje model do pliku PKL."""

    joblib.dump(model, "credit_model.pkl")

    print("\nModel zapisany jako credit_model.pkl")


def test_saved_model():
    """Wczytuje zapisany model i wykonuje przykładową predykcję."""

    loaded_model = joblib.load("credit_model.pkl")

    print("\nTest pliku .pkl")
    print(type(loaded_model))
    print("Liczba cech:", loaded_model.n_features_in_)

    sample = pd.DataFrame([{
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 1,
        "TotalIncome": 5000,
        "IncomeToLoanRatio": 5000 / 120
    }])

    print("Predykcja:", loaded_model.predict(sample))

    return loaded_model


def show_feature_importance(model, X):
    """Wyświetla znaczenie cech wykorzystanych przez model."""

    print("\nWpływ cech:")

    if hasattr(model, "coef_"):

        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_[0]
        })

        print(
            feature_importance.sort_values(
                by="Coefficient",
                ascending=False
            )
        )

    elif hasattr(model, "feature_importances_"):

        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        print(
            feature_importance.sort_values(
                by="Importance",
                ascending=False
            )
        )


def test_income_impact(model):
    """Sprawdza wpływ dochodu na decyzję modelu."""

    print("\n" + "=" * 40)
    print("TEST WPŁYWU DOCHODU")
    print("=" * 40)

    test_low_income = pd.DataFrame([{
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 1,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 1,
        "TotalIncome": 1,
        "IncomeToLoanRatio": 1 / 120
    }])

    test_high_income = pd.DataFrame([{
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 50000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 1,
        "TotalIncome": 50000,
        "IncomeToLoanRatio": 50000 / 120
    }])

    print("\nDochód = 1")
    print("Predykcja:", model.predict(test_low_income))
    print(
        "Prawdopodobieństwo:",
        model.predict_proba(test_low_income)
    )

    print("\nDochód = 50000")
    print("Predykcja:", model.predict(test_high_income))
    print(
        "Prawdopodobieństwo:",
        model.predict_proba(test_high_income)
    )


def test_credit_history(model):
    """Sprawdza wpływ historii kredytowej na decyzję modelu."""

    print("\n" + "=" * 40)
    print("TEST CREDIT HISTORY")
    print("=" * 40)

    credit_good = pd.DataFrame([{
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 1,
        "TotalIncome": 5000,
        "IncomeToLoanRatio": 5000 / 120
    }])

    credit_bad = pd.DataFrame([{
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 0,
        "Property_Area": 1,
        "TotalIncome": 5000,
        "IncomeToLoanRatio": 5000 / 120
    }])

    print("\nCredit_History = 1")
    print("Predykcja:", model.predict(credit_good))
    print(
        "Prawdopodobieństwo:",
        model.predict_proba(credit_good)
    )

    print("\nCredit_History = 0")
    print("Predykcja:", model.predict(credit_bad))
    print(
        "Prawdopodobieństwo:",
        model.predict_proba(credit_bad)
    )


def main():
    """Uruchamia pełny proces trenowania, ewaluacji i zapisu modelu."""

    df = load_data()

    df = preprocess_data(df)

    X, y, X_train, X_test, y_train, y_test = split_data(df)

    best_model, best_score = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nNajlepszy model:")
    print(type(best_model).__name__)
    print(f"Accuracy: {best_score:.4f}")

    save_model(best_model)

    loaded_model = test_saved_model()

    show_feature_importance(
        best_model,
        X
    )

    test_income_impact(loaded_model)

    test_credit_history(loaded_model)


if __name__ == "__main__":
    main()
