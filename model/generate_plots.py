import pandas as pd
import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting"
]

accuracy = [
    0.7805,
    0.7724,
    0.7886
]

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracy)

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")

plt.ylim(0.77, 0.79)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.savefig(
    "model_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

features = [
    "Credit_History",
    "TotalIncome",
    "IncomeToLoanRatio",
    "LoanAmount",
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Property_Area",
    "Married",
    "Dependents",
    "Gender",
    "Education",
    "Self_Employed"
]

importance = [
    0.413731,
    0.137843,
    0.126187,
    0.112263,
    0.083051,
    0.035308,
    0.035307,
    0.029023,
    0.009331,
    0.006931,
    0.006344,
    0.002585,
    0.002096
]

plt.figure(figsize=(10, 6))

plt.barh(features, importance)

plt.title("Feature Importance")

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

df = pd.read_csv("../data/raw/train_u6lujuX_CVtuZ9i.csv")

missing = df.isnull().sum()
missing = missing[missing > 0]

plt.figure(figsize=(8, 5))

plt.bar(
    missing.index,
    missing.values
)

plt.title("Missing Values Before Preprocessing")
plt.ylabel("Missing Count")

plt.xticks(rotation=45)

for i, value in enumerate(missing.values):
    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

plt.savefig(
    "missing_values.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#LoanStatusDistribution

loan_counts = df["Loan_Status"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    loan_counts.index,
    loan_counts.values
)

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")

for i, value in enumerate(loan_counts.values):
    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

plt.savefig(
        "loan_status_distribution.png",
        dpi=300,
        bbox_inches="tight"
)

plt.show()

#CreditHistvsCreditLoan
credit_vs_loan = pd.crosstab(
    df["Credit_History"],
    df["Loan_Status"]
)

credit_vs_loan.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.legend(title="Loan Status")
plt.title("Credit History vs Loan Status")
plt.xlabel("Credit History")
plt.ylabel("Count")

plt.savefig(
    "credit_history_vs_loan_status.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#Dochodwplyw
income_df = df.copy()

income_df["TotalIncome"] = (
    income_df["ApplicantIncome"] +
    income_df["CoapplicantIncome"]
)

income_df["IncomeGroup"] = pd.cut(
    income_df["TotalIncome"],
    bins=[0, 2500, 5000, 10000, float("inf")],
    labels=[
        "0-2500",
        "2500-5000",
        "5000-10000",
        "10000+"
    ]
)

approval_rate = (
    income_df.groupby("IncomeGroup")["Loan_Status"]
    .apply(lambda x: (x == "Y").mean() * 100)
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    approval_rate.index.astype(str),
    approval_rate.values
)

plt.title("Loan Approval Rate by Total Income")
plt.xlabel("Total Income")
plt.ylabel("Approval Rate (%)")

for bar, value in zip(bars, approval_rate.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )

plt.ylim(0, 100)

plt.savefig(
    "total_income_approval_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
