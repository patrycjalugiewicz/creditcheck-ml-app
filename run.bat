@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="

where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.13 3.12 3.11 3.10) do (
        if not defined PYTHON_CMD (
            py -%%V -c "import sys" >nul 2>&1
            if not errorlevel 1 set "PYTHON_CMD=py -%%V"
        )
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo.
    echo Nie znaleziono zgodnej wersji Pythona.
    echo Zainstaluj Python 3.13 i zaznacz opcje "Add Python to PATH".
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Tworzenie srodowiska wirtualnego...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo.
        echo Nie udalo sie utworzyc srodowiska .venv.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Istniejace srodowisko .venv korzysta z niezgodnej wersji Pythona.
    echo Usun katalog .venv i uruchom ten plik ponownie.
    pause
    exit /b 1
)

echo Instalowanie wymaganych bibliotek...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo Nie udalo sie zainstalowac wymaganych bibliotek.
    echo Sprawdz polaczenie z internetem i sprobuj ponownie.
    pause
    exit /b 1
)

echo Uruchamianie aplikacji CreditCheck...
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py

if errorlevel 1 (
    echo.
    echo Aplikacja zakonczyla dzialanie z bledem.
)

pause
endlocal
