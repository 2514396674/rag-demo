@echo off
setlocal
cd /d "%~dp0"

if not exist "key.txt" (
  echo [ERROR] key.txt not found. Put your DeepSeek API key in key.txt ^(one line^).
  pause
  exit /b 1
)

set "DEEPSEEK_API_KEY="
for /f "usebackq delims=" %%a in ("key.txt") do (
  if not defined DEEPSEEK_API_KEY set "DEEPSEEK_API_KEY=%%a"
)

echo Starting web UI at http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
pause
