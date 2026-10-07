@echo off
REM InboxIQ - start the Streamlit app (Demo Mode is ON by default).
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo First run detected - running setup.bat ...
    call setup.bat
)
venv\Scripts\python.exe -m streamlit run app.py
pause
