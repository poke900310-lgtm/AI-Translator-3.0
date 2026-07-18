@echo off
setlocal
call .venv\Scripts\activate
python main.py
set _rc=%errorlevel%
endlocal & exit /b %_rc%
