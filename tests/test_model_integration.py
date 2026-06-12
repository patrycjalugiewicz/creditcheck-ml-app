"""Testy zgodności formularza Streamlit z zapisanym modelem."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from app import streamlit_app
from app.streamlit_app import (
    FEATURE_NAMES,
    ModelLoadError,
    get_acceptance_probability,
    handle_prediction,
    load_model,
    prepare_input_data,
    validate_form_data,
)


class ModelIntegrationTest(unittest.TestCase):
    """Sprawdza przygotowanie danych i predykcję zapisanego modelu."""

    def setUp(self):
        """Tworzy przykładowe dane zwracane przez formularz Streamlit."""
        self.form_data = {
            "gender": "Male",
            "married": "No",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 5000,
            "coapplicant_income": 0,
            "has_coapplicant": False,
            "loan_amount": 120.0,
            "loan_term": 360,
            "credit_history": 1,
            "property_area": "Urban",
            "loan_amount_full": 120000,
        }

    def test_prepare_input_data_matches_training_encoding(self):
        """Sprawdza zgodność kodowania formularza z kodowaniem treningowym."""
        input_data = prepare_input_data(self.form_data)

        self.assertEqual(list(input_data.columns), FEATURE_NAMES)
        self.assertEqual(
            input_data.iloc[0].to_dict(),
            {
                "Gender": 1.0,
                "Married": 0.0,
                "Dependents": 0.0,
                "Education": 0.0,
                "Self_Employed": 0.0,
                "ApplicantIncome": 5000.0,
                "CoapplicantIncome": 0.0,
                "LoanAmount": 120.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": 2.0,
                "TotalIncome": 5000.0,
                "IncomeToLoanRatio": 5000.0 / 120.0,
            },
        )

    def test_prepare_input_data_uses_selected_gender(self):
        """Sprawdza przekazanie do modelu płci wybranej w formularzu."""
        self.form_data["gender"] = "Female"

        input_data = prepare_input_data(self.form_data)

        self.assertEqual(input_data.iloc[0]["Gender"], 0)

    def test_validation_rejects_zero_values(self):
        """Sprawdza błędy walidacji dla zerowego dochodu i kwoty kredytu."""
        self.form_data["applicant_income"] = 0
        self.form_data["loan_amount_full"] = 0
        self.form_data["loan_amount"] = 0

        errors = validate_form_data(self.form_data)

        self.assertEqual(len(errors), 2)

    def test_zero_loan_amount_does_not_cause_division_error(self):
        """Sprawdza zabezpieczenie przed dzieleniem przez zero."""
        self.form_data["loan_amount_full"] = 0
        self.form_data["loan_amount"] = 0

        input_data = prepare_input_data(self.form_data)

        self.assertEqual(input_data.iloc[0]["IncomeToLoanRatio"], 0)

    def test_validation_requires_coapplicant_income(self):
        """Sprawdza wymaganie dochodu zadeklarowanego współwnioskodawcy."""
        self.form_data["has_coapplicant"] = True

        errors = validate_form_data(self.form_data)

        self.assertEqual(len(errors), 1)
        self.assertIn("współwnioskodawcy", errors[0])

    def test_saved_model_accepts_form_data(self):
        """Sprawdza predykcję zapisanego modelu dla danych formularza."""
        model = load_model()
        input_data = prepare_input_data(self.form_data)

        self.assertIsNotNone(model)
        self.assertEqual(list(model.feature_names_in_), FEATURE_NAMES)
        self.assertIn(model.predict(input_data)[0], (0, 1))

        probability = get_acceptance_probability(model, input_data)
        self.assertIsNotNone(probability)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_load_model_reports_missing_file(self):
        """Sprawdza kontrolowany błąd dla brakującego pliku modelu."""
        missing_path = Path("model/does_not_exist.pkl")

        with self.assertRaisesRegex(ModelLoadError, "Nie znaleziono"):
            load_model(missing_path)

    def test_load_model_reports_corrupted_file(self):
        """Sprawdza kontrolowany błąd dla uszkodzonego pliku modelu."""
        model_path = Mock(spec=Path)
        model_path.is_file.return_value = True

        with patch.object(
            streamlit_app.joblib,
            "load",
            side_effect=ValueError("corrupted file"),
        ):
            with patch.object(streamlit_app.LOGGER, "exception"):
                with self.assertRaisesRegex(ModelLoadError, "uszkodzony"):
                    load_model(model_path)

    def test_load_model_rejects_incompatible_object(self):
        """Sprawdza odrzucenie pliku, który nie zawiera modelu."""
        model_path = Mock(spec=Path)
        model_path.is_file.return_value = True

        with patch.object(
            streamlit_app.joblib,
            "load",
            return_value={"not": "a model"},
        ):
            with self.assertRaisesRegex(ModelLoadError, "modelu predykcyjnego"):
                load_model(model_path)

    def test_load_model_rejects_different_features(self):
        """Sprawdza odrzucenie modelu wykorzystującego inne cechy."""
        model_path = Mock(spec=Path)
        model_path.is_file.return_value = True
        incompatible_model = Mock()
        incompatible_model.feature_names_in_ = ["DifferentFeature"]

        with patch.object(
            streamlit_app.joblib,
            "load",
            return_value=incompatible_model,
        ):
            with self.assertRaisesRegex(ModelLoadError, "inny zestaw"):
                load_model(model_path)

    @patch.object(streamlit_app, "show_result")
    @patch.object(streamlit_app.st, "error")
    @patch.object(streamlit_app.LOGGER, "exception")
    def test_prediction_error_does_not_show_fake_result(
        self,
        log_exception,
        show_error,
        show_result,
    ):
        """Sprawdza brak wyniku po wystąpieniu błędu predykcji."""
        failing_model = Mock()
        failing_model.predict.side_effect = ValueError("prediction failed")

        result = handle_prediction(failing_model, self.form_data)

        self.assertFalse(result)
        log_exception.assert_called_once()
        show_error.assert_called_once()
        show_result.assert_not_called()

    @patch.object(streamlit_app, "show_result")
    @patch.object(streamlit_app.st, "error")
    def test_missing_model_does_not_show_fake_result(
        self,
        show_error,
        show_result,
    ):
        """Sprawdza brak fikcyjnego wyniku przy braku modelu."""
        result = handle_prediction(None, self.form_data)

        self.assertFalse(result)
        show_error.assert_called_once()
        show_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
