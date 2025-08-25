@echo off
REM Batch script to run stl_to_depthmap.py with the correct Python environment
REM Usage: run_stl_to_depthmap.bat <stl_file> [additional_options]

if %1=="" (
    echo Usage: %0 ^<stl_file^> [additional_options]
    echo Example: %0 foam.stl --verbose
    exit /b 1
)

"%~dp0.venv312\Scripts\python.exe" "%~dp0stl_to_depthmap.py" %*
