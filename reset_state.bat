@echo off
setlocal
call "%~dp0.venv\Scripts\activate"
python "%~dp0scripts/reset_state.py"
set _rc=%errorlevel%
endlocal & exit /b %_rc%
