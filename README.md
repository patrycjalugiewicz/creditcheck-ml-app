# CreditCheck - predykcja decyzji kredytowej

CreditCheck to lokalna aplikacja webowa wspierająca wstępną ocenę szans na
uzyskanie kredytu. Użytkownik uzupełnia formularz, a aplikacja przedstawia
przewidywaną decyzję modelu, prawdopodobieństwo akceptacji oraz komentarz do
podanych danych.

## Uruchomienie

### Wymagania

- Windows 10 lub Windows 11,
- Python od 3.10 do 3.13 (zalecany Python 3.13),
- połączenie z internetem podczas pierwszego uruchomienia,
- plik modelu `model/credit_model.pkl`.

Python można pobrać ze strony
[python.org](https://www.python.org/downloads/). Podczas instalacji należy
zaznaczyć opcję `Add Python to PATH`.

### Automatyczne uruchomienie

1. Pobierz lub sklonuj repozytorium.
2. Uruchom plik `run.bat`.
3. Przy pierwszym uruchomieniu skrypt:
   - sprawdzi wersję Pythona,
   - utworzy lokalne środowisko `.venv`,
   - zainstaluje wersje bibliotek z `requirements.txt`,
   - uruchomi aplikację Streamlit.
4. Aplikacja otworzy się w przeglądarce pod adresem
   `http://localhost:8501`.

Kolejne uruchomienia wykorzystują istniejące środowisko `.venv`. Jeżeli
środowisko zostało utworzone przez niezgodną wersję Pythona, należy usunąć
katalog `.venv` i ponownie uruchomić `run.bat`.

### Uruchomienie ręczne

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

## Działanie aplikacji

Formularz pobiera informacje o wnioskodawcy i parametrach kredytu:

- płeć,
- stan cywilny,
- liczbę osób na utrzymaniu,
- wykształcenie,
- formę zatrudnienia,
- miesięczny dochód wnioskodawcy,
- miesięczny dochód współwnioskodawcy,
- kwotę i okres kredytowania,
- historię spłaty zobowiązań,
- lokalizację nieruchomości.

Przed predykcją aplikacja sprawdza, czy:

- dochód wnioskodawcy jest większy od zera,
- podano dochód zadeklarowanego współwnioskodawcy,
- kwota kredytu jest większa od zera,
- model istnieje i wykorzystuje oczekiwany zestaw cech.

Brak, uszkodzenie lub niezgodność modelu powoduje wyświetlenie komunikatu
błędu. Aplikacja nie generuje w takim przypadku wyniku demonstracyjnego.

## Model i dane

Model został wytrenowany na zbiorze Loan Prediction Dataset zawierającym 614 rekordów. 
Celem modelu jest przewidywanie decyzji kredytowej na podstawie danych dotyczących 
wnioskodawcy, parametrów kredytu oraz historii kredytowej.

W procesie przygotowania danych:

- usunięto kolumnę Loan_ID,
- uzupełniono brakujące wartości za pomocą mediany lub wartości najczęściej występującej,
- zakodowano zmienne kategoryczne przy użyciu LabelEncoder,
- utworzono dodatkowe cechy:
   - TotalIncome = ApplicantIncome + CoapplicantIncome,
   - IncomeToLoanRatio = TotalIncome / LoanAmount.

Przetestowano trzy modele uczenia maszynowego:

- Logistic Regression,
- Random Forest,
- Gradient Boosting.

Najlepsze wyniki uzyskał model GradientBoostingClassifier, osiągając skuteczność (Accuracy) na poziomie około 78.86%.

Najwyższą ważność według `feature_importances_` uzyskały cechy:

- Credit_History,
- TotalIncome,
- IncomeToLoanRatio,
- LoanAmount.

### Wektor wejściowy

Model wykorzystuje 13 cech wejściowych:

- Gender,
- Married,
- Dependents,
- Education,
- Self_Employed,
- ApplicantIncome,
- CoapplicantIncome,
- LoanAmount,
- Loan_Amount_Term,
- Credit_History,
- Property_Area,
- TotalIncome,
- IncomeToLoanRatio.

Dwie ostatnie cechy są generowane podczas preprocessingu danych i nie występują bezpośrednio w oryginalnym zbiorze danych.

### Wektor wyjściowy

Model zwraca binarną decyzję kredytową:

- 0 – odrzucenie wniosku kredytowego,
- 1 – akceptacja wniosku kredytowego.

Dodatkowo aplikacja wykorzystuje metodę predict_proba(), dzięki której możliwe jest wyznaczenie prawdopodobieństwa 
przyznania kredytu i przedstawienie użytkownikowi bardziej szczegółowej informacji o wyniku predykcji.

## Architektura

Projekt jest podzielony na trzy obszary:

- `data` - dane surowe i przetworzone,
- `model` - preprocessing, trening i zapis modelu,
- `app` - formularz Streamlit, integracja z modelem i prezentacja wyniku.

```text
creditcheck-ml-app/
|-- app/
|   |-- __init__.py
|   `-- streamlit_app.py
|-- data/
|   |-- __init__.py
|   |-- preprocess.py
|   `-- raw/
|       |-- train_u6lujuX_CVtuZ9i.csv
|       `-- test_Y3wMUE5_7gLdaTN.csv
|-- docs/
|   `-- Dokumentacja.docx
|-- model/
|   |-- credit_model.pkl
|   `-- train_model.py
|-- tests/
|   `-- test_model_integration.py
|-- .gitignore
|-- README.md
|-- requirements.txt
`-- run.bat
```

## Testowanie i jakość kodu

Testy integracyjne sprawdzają między innymi:

- zgodność kodowania formularza z modelem,
- wykonanie predykcji na rzeczywistym pliku `.pkl`,
- brak, uszkodzenie i niezgodność modelu,
- błędy predykcji,
- walidację danych formularza.

Uruchomienie testów:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sprawdzenie kodu:

```powershell
.\.venv\Scripts\python.exe -m pylint app\streamlit_app.py data\preprocess.py model\train_model.py tests\test_model_integration.py
```

W ostatniej lokalnej weryfikacji projekt uzyskał wynik pylint powyżej
wymaganego progu 8 punktów.

## System testowy i ograniczenia

Aplikację przetestowano lokalnie na:

- Windows 11,
- Python 3.13,
- Streamlit 1.41.1,
- scikit-learn 1.6.1.

Aplikacja jest przeznaczona do uruchamiania lokalnego na Windows. Nie była
testowana na Linuxie, macOS etc. Pierwsze uruchomienie wymaga internetu w celu pobrania bibliotek.


## Technologie

- Python,
- Streamlit,
- pandas,
- NumPy,
- scikit-learn,
- joblib,
- Git i GitHub.

## Ograniczenia

- model wytrenowany na zagranicznym zbiorze danych, nieprzystosowanym do polskiego rynku.

## Podział pracy

### Piotr Balcerzak

- przygotowanie i preprocessing datasetu,
- analiza danych i wizualizacja wybranych zależności,
- implementacja i trenowanie modeli uczenia maszynowego,
- porównanie skuteczności modeli i wybór najlepszego rozwiązania,
- zapis wytrenowanego modelu do pliku `.pkl`,
- przygotowanie dokumentacji modelu i analizy danych.

### Patrycja Ługiewicz

- utworzenie i konfiguracja repozytorium Git,
- przygotowanie struktury projektu,
- implementacja aplikacji webowej w Streamlit,
- przygotowanie formularza i integracja aplikacji z modelem,
- prezentacja wyniku i komunikatów interpretacyjnych,
- obsługa błędów i testy integracyjne,
- przygotowanie README, dokumentacji aplikacji oraz prezentacji projektu.
