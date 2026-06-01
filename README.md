# CreditCheck – predykcja decyzji kredytowej

## Uruchomienie aplikacji

Aby uruchomić aplikację należy uruchomić plik `run.bat` znajdujący się w głównym katalogu projektu. Plik automatycznie instaluje wymagane biblioteki oraz uruchamia aplikację Streamlit.

## Opis projektu

CreditCheck to aplikacja webowa wspierająca wstępną ocenę szans na uzyskanie kredytu. Użytkownik uzupełnia formularz dotyczący swojej sytuacji finansowej oraz parametrów kredytu, a aplikacja zwraca przewidywaną decyzję: akceptacja lub odrzucenie wniosku.

Aplikacja wykorzystuje model uczenia maszynowego wytrenowany na publicznym datasetcie Loan Prediction Dataset z Kaggle.

## Rodzaj aplikacji

Aplikacja webowa uruchamiana lokalnie z wykorzystaniem Streamlit.

## Dataset

W projekcie wykorzystano dataset Loan Prediction Dataset dostępny na Kaggle:

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Dataset zawiera dane dotyczące historycznych wniosków kredytowych, m.in. dochód wnioskodawcy, dochód współwnioskodawcy, kwotę kredytu, okres kredytowania, historię kredytową, wykształcenie, liczbę osób na utrzymaniu oraz decyzję kredytową.

Dane zostały oczyszczone i przygotowane na potrzeby trenowania modelu.

## Model uczenia maszynowego

Na potrzeby projektu zostanie wytrenowany model klasyfikacyjny przewidujący decyzję kredytową. Rozważane modele obejmują Logistic Regression, Random Forest oraz Gradient Boosting.

Po zakończeniu procesu trenowania wybrany zostanie model osiągający najlepsze wyniki.

## Predykcja

Aplikacja przewiduje, czy wniosek kredytowy użytkownika prawdopodobnie zostałby zaakceptowany czy odrzucony.

Wynik aplikacji:
- decyzja: zaakceptowany / odrzucony,
- prawdopodobieństwo akceptacji,
- krótka interpretacja możliwych czynników wpływających na wynik.

## Wektor wejściowy modelu

Model przyjmuje następujące cechy:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

## Wektor wyjściowy modelu

Model zwraca przewidywaną decyzję kredytową:

- Y / 1 – wniosek zaakceptowany,
- N / 0 – wniosek odrzucony.

Jeżeli model obsługuje `predict_proba`, aplikacja pokazuje również prawdopodobieństwo akceptacji.

## Technologie

- Python
- Streamlit
- pandas
- scikit-learn
- joblib
- GitHub

## System testowy

Aplikacja była testowana na:

- Windows 11
- Python 3.14
- Google Chrome
- Opera GX


Aplikacja powinna działać również na innych wersjach systemu Windows, np. Windows 10. Aplikacja nie jest przewidziana jako natywna aplikacja mobilna.

## Podział pracy

### Piotr Balcerzak
- przygotowanie i analiza datasetu,
- preprocessing danych,
- trenowanie modeli uczenia maszynowego,
- porównanie skuteczności modeli,
- wybór najlepszego modelu,
- zapis modelu do pliku .pkl.

### Patrycja Ługiewicz
- utworzenie i konfiguracja repozytorium GitHub,
- przygotowanie struktury projektu,
- implementacja aplikacji webowej w Streamlit,
- stworzenie formularza użytkownika,
- integracja aplikacji z modelem,
- przygotowanie README,
- przygotowanie dokumentacji PDF,
- przygotowanie prezentacji projektu.

## Struktura projektu

```txt
creditcheck-ml-app/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── model/
│   ├── credit_model.pkl
│   ├── train_model.py
│   └── predict.py
├── docs/
│   └── documentation.pdf
├── requirements.txt
├── run.bat
├── README.md
└── .gitignore