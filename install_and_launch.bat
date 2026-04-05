@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Qwen TG Installer and Launcher

set "PYTHON_CMD="
py -3 --version >nul 2>nul && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    python --version >nul 2>nul && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    cls
    echo Привет^!
    echo Я установщик, простой и удобный^! ^:)
    echo.
    echo Но есть нюанс: Python не найден.
    echo Поставь Python 3.11+ и перезапусти меня.
    echo.
    pause
    exit /b 1
)

%PYTHON_CMD% "%~dp0launcher_cli.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Что-то пошло не так. Код выхода: %EXIT_CODE%
    pause
)

endlocal & exit /b %EXIT_CODE%
