@echo off
REM Quick fix script to install missing dependencies and test the application

echo ========================================
echo DXF Quotation Application - Quick Fix
echo ========================================
echo.

echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Step 2: Installing missing dependencies...
echo Installing psycopg2-binary...
pip install psycopg2-binary
if %errorLevel% neq 0 (
    echo ERROR: Failed to install psycopg2-binary
    pause
    exit /b 1
)

echo.
echo Installing Playwright for PDF generation...
pip install playwright
if %errorLevel% neq 0 (
    echo ERROR: Failed to install playwright
    pause
    exit /b 1
)

echo.
echo Installing Playwright Chromium browser...
playwright install chromium
if %errorLevel% neq 0 (
    echo ERROR: Failed to install Chromium browser
    pause
    exit /b 1
)

echo.
echo Step 3: Testing application startup...
echo Starting application in test mode...
echo Press Ctrl+C to stop the test
echo.
python startup_script.py

echo.
echo Test completed!
pause
