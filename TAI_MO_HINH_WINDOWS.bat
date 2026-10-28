@echo off
chcp 65001 >nul
cd /d "%~dp0"
py scripts\setup_assets.py
if errorlevel 1 echo Xem loi phia tren; tai thu cong neu can.
pause
