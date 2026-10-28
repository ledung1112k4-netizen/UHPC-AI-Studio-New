@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title UHPC AI Studio - Du an moi
where py >nul 2>&1
if errorlevel 1 (
 echo Chua cai Python. Hay cai Python 3.11 hoac 3.12, chon Add to PATH.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
 py -m venv .venv
 if errorlevel 1 (echo Loi tao moi truong Python.& pause & exit /b 1)
)
echo Dang cai dat / kiem tra cac thu vien...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Loi cai thu vien. Kiem tra Internet.& pause & exit /b 1)
if not exist "assets\uhpc_model.pkl" (
 echo Chua co model AI. Thu tai tu GitHub cua ban...
 ".venv\Scripts\python.exe" scripts\setup_assets.py
 if errorlevel 1 echo Chua tai du tai san. App se mo nhung AI co the chua san sang.
)
echo Mo trinh duyet toi http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
pause
