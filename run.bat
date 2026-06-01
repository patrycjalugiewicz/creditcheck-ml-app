@echo off

python -m pip install -r requirements.txt --quiet

python -m streamlit run app/streamlit_app.py

pause