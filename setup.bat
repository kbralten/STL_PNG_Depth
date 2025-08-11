@echo off
echo Setting up STL to Depth Map converter...
echo.

REM Check if Python 3.12 is available
echo Checking for Python 3.12...
set PYTHON312_PATH="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist %PYTHON312_PATH% (
    echo Found Python 3.12 at %PYTHON312_PATH%
) else (
    echo Python 3.12 not found. Installing via winget...
    winget install Python.Python.3.12
    if errorlevel 1 (
        echo Failed to install Python 3.12. Please install manually.
        pause
        exit /b 1
    )
)

REM Create virtual environment if it doesn't exist
if not exist ".venv312" (
    echo Creating virtual environment...
    %PYTHON312_PATH% -m venv .venv312
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM Install dependencies
echo Installing dependencies...
.venv312\Scripts\python.exe -m pip install --upgrade pip
.venv312\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Setup complete! You can now run:
echo run_stl_to_depthmap.bat your_file.stl
echo.
echo Or directly:
echo .venv312\Scripts\python.exe stl_to_depthmap.py your_file.stl
echo.
pause
