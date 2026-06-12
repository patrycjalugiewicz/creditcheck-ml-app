"""Aplikacja Streamlit do predykcji decyzji kredytowej."""

import logging
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "credit_model.pkl"

FEATURE_NAMES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "TotalIncome",
    "IncomeToLoanRatio",
]

CATEGORY_ENCODINGS = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}


class ModelLoadError(RuntimeError):
    """Błąd uniemożliwiający bezpieczne wczytanie modelu."""


class PredictionError(RuntimeError):
    """Błąd uniemożliwiający wykonanie predykcji."""


def load_model(model_path=MODEL_PATH):
    """Wczytuje model i sprawdza jego zgodność z aplikacją."""
    if not model_path.is_file():
        raise ModelLoadError(
            "Nie znaleziono pliku modelu `model/credit_model.pkl`. "
            "Predykcja jest obecnie niedostępna."
        )

    try:
        model = joblib.load(model_path)
    except Exception as error:  # Model files can fail with library-specific errors.
        LOGGER.exception("Nie udało się wczytać modelu z %s", model_path)
        raise ModelLoadError(
            "Nie udało się wczytać modelu. Plik może być uszkodzony albo "
            "niezgodny z zainstalowaną wersją bibliotek."
        ) from error

    if not callable(getattr(model, "predict", None)):
        raise ModelLoadError(
            "Wczytany plik nie zawiera obsługiwanego modelu predykcyjnego."
        )

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is None or list(model_feature_names) != FEATURE_NAMES:
        raise ModelLoadError(
            "Wczytany model jest niezgodny z aplikacją. "
            "Model wykorzystuje inny zestaw lub kolejność cech."
        )

    return model


def prepare_input_data(form_data):
    """Przygotowuje dane wejściowe w formacie zgodnym z modelem."""
    total_income = (
        form_data["applicant_income"] + form_data["coapplicant_income"]
    )
    loan_amount = form_data["loan_amount"]
    income_to_loan_ratio = total_income / loan_amount if loan_amount > 0 else 0

    input_data = pd.DataFrame(
        [
            {
                "Gender": CATEGORY_ENCODINGS["Gender"][form_data["gender"]],
                "Married": CATEGORY_ENCODINGS["Married"][form_data["married"]],
                "Dependents": CATEGORY_ENCODINGS["Dependents"][
                    form_data["dependents"]
                ],
                "Education": CATEGORY_ENCODINGS["Education"][
                    form_data["education"]
                ],
                "Self_Employed": CATEGORY_ENCODINGS["Self_Employed"][
                    form_data["self_employed"]
                ],
                "ApplicantIncome": form_data["applicant_income"],
                "CoapplicantIncome": form_data["coapplicant_income"],
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": form_data["loan_term"],
                "Credit_History": form_data["credit_history"],
                "Property_Area": CATEGORY_ENCODINGS["Property_Area"][
                    form_data["property_area"]
                ],
                "TotalIncome": total_income,
                "IncomeToLoanRatio": income_to_loan_ratio,
            }
        ],
        columns=FEATURE_NAMES,
    )
    return input_data


def get_acceptance_probability(model, input_data):
    """Zwraca prawdopodobieństwo klasy oznaczającej akceptację."""
    if not hasattr(model, "predict_proba"):
        return None

    classes = list(model.classes_)
    accepted_class = 1 if 1 in classes else "Y"
    if accepted_class not in classes:
        return None

    accepted_class_index = classes.index(accepted_class)
    return model.predict_proba(input_data)[0][accepted_class_index]


def predict_credit_decision(model, form_data):
    """Wykonuje predykcję i zamienia błędy modelu na błąd aplikacji."""
    try:
        input_data = prepare_input_data(form_data)
        prediction = model.predict(input_data)[0]
        probability = get_acceptance_probability(model, input_data)
    except Exception as error:  # Estimators can raise library-specific errors.
        LOGGER.exception("Nie udało się wykonać predykcji")
        raise PredictionError(
            "Nie udało się obliczyć predykcji. Sprawdź zgodność modelu "
            "z aplikacją i spróbuj ponownie."
        ) from error

    return prediction, probability


def validate_form_data(form_data):
    """Zwraca komunikaty dla danych, które nie pozwalają na predykcję."""
    errors = []

    if form_data["applicant_income"] <= 0:
        errors.append("Dochód wnioskodawcy musi być większy od zera.")

    if form_data["has_coapplicant"] and form_data["coapplicant_income"] <= 0:
        errors.append(
            "Podaj dochód współwnioskodawcy albo zaznacz brak "
            "współwnioskodawcy."
        )

    if form_data["loan_amount_full"] <= 0:
        errors.append("Wnioskowana kwota kredytu musi być większa od zera.")

    return errors


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

    st.subheader("Komentarz do podanych danych")
    for comment in comments:
        st.write(f"• {comment}")


def get_applicant_data():
    """Pobiera dane wnioskodawcy z formularza."""
    st.header("Dane wnioskodawcy")

    gender_label = st.radio(
        "Płeć",
        ["Kobieta", "Mężczyzna"],
        horizontal=True,
    )
    gender = "Female" if gender_label == "Kobieta" else "Male"

    married_label = st.radio(
        "Czy pozostajesz w związku małżeńskim?",
        ["Nie", "Tak"],
        horizontal=True,
    )
    married = "Yes" if married_label == "Tak" else "No"

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
            "Miesięczny dochód netto współwnioskodawcy",
            min_value=0,
            value=3000,
            step=500,
        )
    else:
        coapplicant_income = 0

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
        "gender": gender,
        "married": married,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "has_coapplicant": has_coapplicant == "Tak",
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


def handle_prediction(model, form_data):
    """Obsługuje predykcję dla danych wpisanych przez użytkownika."""
    if model is None:
        st.error(
            "Predykcja jest niedostępna, ponieważ model nie został wczytany."
        )
        return False

    validation_errors = validate_form_data(form_data)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        return False

    try:
        prediction, probability = predict_credit_decision(model, form_data)
    except PredictionError as error:
        st.error(str(error))
        return False

    show_result(
        prediction=prediction,
        probability=probability,
        comments=generate_comments(form_data),
    )
    return True


def main():
    """Uruchamia aplikację Streamlit."""
    st.set_page_config(
        page_title="CreditCheck",
        page_icon="💳",
        layout="centered",
    )

    st.title("CreditCheck")
    st.write("Wypełnij dane, aby sprawdzić przewidywaną decyzję kredytową.")

    try:
        model = load_model()
    except ModelLoadError as error:
        model = None
        st.error(str(error))
        st.info(
            "Formularz pozostaje dostępny, ale obliczenie wyniku będzie "
            "możliwe dopiero po przywróceniu prawidłowego modelu."
        )

    form_data = {}
    form_data.update(get_applicant_data())
    form_data.update(get_loan_data())

    if st.button(
        "Sprawdź decyzję kredytową",
        disabled=model is None,
    ):
        handle_prediction(model, form_data)


if __name__ == "__main__":
    main()
