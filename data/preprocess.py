import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    df = df.copy()

    df.drop("Loan_ID", axis=1, inplace=True)

    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(
        df["Self_Employed"].mode()[0]
    )

    df["LoanAmount"] = df["LoanAmount"].fillna(
        df["LoanAmount"].median()
    )

    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
        df["Loan_Amount_Term"].median()
    )

    df["Credit_History"] = df["Credit_History"].fillna(
        df["Credit_History"].mode()[0]
    )

    df["TotalIncome"] = (
        df["ApplicantIncome"] +
        df["CoapplicantIncome"]
    )

    df["IncomeToLoanRatio"] = (
        df["TotalIncome"] /
        df["LoanAmount"]
    )

    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status"
    ]

    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    return df
