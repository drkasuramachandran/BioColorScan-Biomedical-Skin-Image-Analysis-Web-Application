@echo off
cd /d "%~dp0"

echo ==========================================
echo   Skin Cancer Imaging Research Platform
echo ==========================================
echo.

set "PY=C:\Users\Ramachandran\AppData\Local\Programs\Python\Python313\python.exe"

if not exist "venv\Scripts\python.exe" (
    echo First run: creating environment and installing packages...
    echo This can take several minutes.
    echo.
    "%PY%" -m venv venv
    if errorlevel 1 goto :error
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
    echo.
)

echo Starting Streamlit...
echo.

"venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:error
echo.
echo Setup failed. Check the messages above.

:end
pause
