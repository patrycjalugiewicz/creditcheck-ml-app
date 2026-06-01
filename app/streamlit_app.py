"""Aplikacja Streamlit do predykcji decyzji kredytowej."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("model/credit_model.pkl")


def load_model():
    """Wczytuje wytrenowany model, jeżeli plik istnieje."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def prepare_input_data(form_data):
    """Przygotowuje dane wejściowe w formacie zgodnym z modelem."""
    return pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Married": form_data["married"],
                "Dependents": form_data["dependents"],
                "Education": form_data["education"],
                "Self_Employed": form_data["self_employed"],
                "ApplicantIncome": form_data["applicant_income"],
                "CoapplicantIncome": form_data["coapplicant_income"],
                "LoanAmount": form_data["loan_amount"],
                "Loan_Amount_Term": form_data["loan_term"],
                "Credit_History": form_data["credit_history"],
                "Property_Area": form_data["property_area"],
            }
        ]
    )


def generate_comments(form_data):
    """Generuje krótką interpretację czynników wpływających na wynik."""
    comments = []
    total_income = (
        form_data["applicant_income"] + form_data["coapplicant_income"]
    )

    if form_data["credit_history"] == 1:
        comments.append(
            "Brak problemów ze spłatą zobowiązań pozytywnie wpływa na ocenę."
        )
    else:
        comments.append(
            "Problemy ze spłatą zobowiązań mogą obniżać szansę akceptacji."
        )

    if total_income >= 8000:
        comments.append(
            "Wysoki łączny dochód zwiększa przewidywaną szansę "
            "uzyskania kredytu."
        )
    elif total_income < 4000:
        comments.append(
            "Niski łączny dochód może obniżać przewidywaną szansę "
            "akceptacji."
        )

    if total_income > 0 and form_data["loan_amount_full"] > total_income * 20:
        comments.append(
            "Wysoka kwota kredytu względem dochodu może negatywnie wpływać "
            "na wynik."
        )

    if form_data["dependents"] == "3+":
        comments.append(
            "Większa liczba osób na utrzymaniu może wpływać na ocenę "
            "zdolności kredytowej."
        )

    if form_data["education"] == "Graduate":
        comments.append(
            "Wykształcenie wyższe może pozytywnie wpływać na ocenę wniosku."
        )

    return comments


def show_result(prediction, probability, comments):
    """Wyświetla wynik predykcji oraz komentarze."""
    accepted_values = ("Y", 1, True)

    st.divider()
    st.subheader("Wynik analizy")

    if prediction in accepted_values:
        st.success("Przewidywana decyzja: wniosek prawdopodobnie zaakceptowany")
    else:
        st.error("Przewidywana decyzja: wniosek prawdopodobnie odrzucony")

    if probability is not None:
        st.metric(
            "Szacowane prawdopodobieństwo akceptacji",
            f"{probability:.0%}",
        )

    st.subheader("Najważniejsze czynniki wpływające na wynik")
    for comment in comments:
        st.write(f"• {comment}")


def get_applicant_data():
    """Pobiera dane wnioskodawcy z formularza."""
    st.header("Dane wnioskodawcy")

    applicant_income = st.number_input(
        "Miesięczny dochód netto wnioskodawcy",
        min_value=0,
        value=5000,
        step=500,
    )

    has_coapplicant = st.radio(
        "Czy kredyt będzie brany wspólnie z drugą osobą?",
        ["Nie", "Tak"],
        horizontal=True,
    )

    if has_coapplicant == "Tak":
        coapplicant_income = st.number_input(
            "Miesięczny dochód netto drugiego wnioskodawcy",
            min_value=0,
            value=3000,
            step=500,
        )
        married = "Yes"
    else:
        coapplicant_income = 0
        married = "No"

    employment_label = st.selectbox(
        "Forma zatrudnienia",
        [
            "Umowa o pracę / umowa zlecenie / inna forma zatrudnienia",
            "Samozatrudnienie / JDG / B2B",
        ],
    )

    self_employed = (
        "Yes"
        if employment_label == "Samozatrudnienie / JDG / B2B"
        else "No"
    )

    has_higher_education = st.radio(
        "Czy posiadasz wykształcenie wyższe?",
        ["Tak", "Nie"],
        horizontal=True,
    )

    education = "Graduate" if has_higher_education == "Tak" else "Not Graduate"

    dependents_label = st.selectbox(
        "Ile osób pozostaje na Twoim utrzymaniu?",
        ["0", "1", "2", "3 lub więcej"],
    )

    dependents = (
        "3+"
        if dependents_label == "3 lub więcej"
        else dependents_label
    )

    return {
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "married": married,
        "self_employed": self_employed,
        "education": education,
        "dependents": dependents,
    }


def get_loan_data():
    """Pobiera dane kredytu z formularza."""
    st.header("Dane kredytu")

    loan_amount_full = st.number_input(
        "Wnioskowana kwota kredytu",
        min_value=0,
        value=120000,
        step=10000,
    )

    loan_amount = loan_amount_full / 1000

    loan_amount_term = st.selectbox(
        "Na jaki okres chcesz wziąć kredyt?",
        [
            "5 lat",
            "10 lat",
            "15 lat",
            "20 lat",
            "25 lat",
            "30 lat",
        ],
    )

    term_mapping = {
        "5 lat": 60,
        "10 lat": 120,
        "15 lat": 180,
        "20 lat": 240,
        "25 lat": 300,
        "30 lat": 360,
    }

    loan_term = term_mapping[loan_amount_term]

    debt_answer = st.radio(
        "Czy w przeszłości zdarzały Ci się opóźnienia lub problemy "
        "ze spłatą zobowiązań?",
        ["Nie", "Tak"],
        horizontal=True,
    )

    credit_history = 1 if debt_answer == "Nie" else 0

    property_area_label = st.selectbox(
        "Gdzie znajduje się nieruchomość?",
        ["Miasto", "Okolice miasta", "Wieś"],
    )

    property_area_mapping = {
        "Miasto": "Urban",
        "Okolice miasta": "Semiurban",
        "Wieś": "Rural",
    }

    property_area = property_area_mapping[property_area_label]

    return {
        "loan_amount_full": loan_amount_full,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": credit_history,
        "property_area": property_area,
    }


def get_demo_prediction(form_data):
    """Zwraca przykładową predykcję przed podłączeniem modelu."""
    is_accepted = (
        form_data["credit_history"] == 1
        and form_data["applicant_income"] >= 3000
    )

    prediction = "Y" if is_accepted else "N"
    probability = 0.78 if is_accepted else 0.34

    return prediction, probability


def handle_prediction(model, form_data):
    """Obsługuje predykcję dla danych wpisanych przez użytkownika."""
    comments = generate_comments(form_data)

    if model is None:
        st.warning(
            "Model nie został jeszcze podłączony. Wyświetlany jest przykładowy "
            "wynik testowy interfejsu."
        )
        prediction, probability = get_demo_prediction(form_data)
    else:
        input_data = prepare_input_data(form_data)
        prediction = model.predict(input_data)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]

    show_result(
        prediction=prediction,
        probability=probability,
        comments=comments,
    )


def main():
    """Uruchamia aplikację Streamlit."""
    st.set_page_config(
        page_title="CreditCheck",
        page_icon="💳",
        layout="centered",
    )

    st.title("CreditCheck")
    st.write("Wypełnij dane, aby sprawdzić przewidywaną decyzję kredytową.")

    model = load_model()

    form_data = {}
    form_data.update(get_applicant_data())
    form_data.update(get_loan_data())

    if st.button("Sprawdź decyzję kredytową"):
        handle_prediction(model, form_data)


if __name__ == "__main__":
    main()