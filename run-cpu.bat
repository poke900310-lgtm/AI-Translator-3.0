@echo off
setlocal
REM CPU-only mode: keep all model layers off the GPU.
REM LLAMA_SERVER_NGL=0 -> llama-server runs the whole model on the CPU.
set LLAMA_SERVER_NGL=0
REM Flash attention is a GPU optimization; disable so CPU-only builds don't reject it.
set LLAMA_SERVER_FA=off
call .venv\Scripts\activate
python main.py
set _rc=%errorlevel%
endlocal & exit /b %_rc%
