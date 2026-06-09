import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Wczytanie danych

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Analiza datasetu

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Usunięcie zbędnej kolumny

df.drop("Loan_ID", axis=1, inplace=True)

# Uzupełnienie braków danych

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].median()
)

df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mode()[0]
)

# Kodowanie danych kategorycznych

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

# Podział na dane wejściowe i wyjściowe

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Podział na zbiór treningowy i testowy

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modele

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Porównanie modeli

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

# Wynik końcowy

print("\nNajlepszy model:")
print(type(best_model).__name__)
print(f"Accuracy: {best_score:.4f}")

# Zapis modelu

joblib.dump(best_model, "credit_model.pkl")

print("\nModel zapisany jako credit_model.pkl")

# Test pliku .pkl

loaded_model = joblib.load("credit_model.pkl")

print("\nTest pliku .pkl")
print(type(loaded_model))
print("Liczba cech:", loaded_model.n_features_in_)

# Testowa predykcja

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
    "Property_Area": 1
}])

print("Predykcja:", loaded_model.predict(sample))