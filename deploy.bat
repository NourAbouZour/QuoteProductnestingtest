@echo off
REM Windows Server 2022 Deployment Script for DXF Quotation Application
REM Run this script as Administrator for best results

echo ========================================
echo DXF Quotation Application Deployment
echo Windows Server 2022
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator - Good!
) else (
    echo WARNING: Not running as Administrator
    echo Some operations may fail
    pause
)

echo.
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorLevel% == 0 (
    echo Python is installed
    python --version
) else (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo.
echo Step 2: Checking pip installation...
pip --version >nul 2>&1
if %errorLevel% == 0 (
    echo pip is available
) else (
    echo ERROR: pip is not available
    pause
    exit /b 1
)

echo.
echo Step 3: Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    echo Creating virtual environment...
    python -m venv venv
    if %errorLevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo Step 4: Activating virtual environment...
call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Step 5: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 6: Installing dependencies...
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 7: Creating required directories...
if not exist uploads mkdir uploads
if not exist temp mkdir temp
if not exist logs mkdir logs
if not exist cache mkdir cache

echo.
echo Step 8: Checking database configuration...
echo Using existing DatabaseConfig.py for database settings
echo Database credentials are configured in DatabaseConfig.py

echo.
echo Step 9: Testing application startup...
echo Starting application in test mode...
echo Press Ctrl+C to stop the test
echo.
python startup_script.py

echo.
echo Deployment completed!
echo.
echo To start the application:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run: python startup_script.py
echo.
echo For production deployment, see windows_deployment_guide.md
echo.
pause
