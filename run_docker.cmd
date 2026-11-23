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

echo DEEPSEEK_API_KEY=%DEEPSEEK_API_KEY%> .env

echo Building and starting container...
docker compose up -d --build
if errorlevel 1 (
  echo [ERROR] docker compose failed.
  pause
  exit /b 1
)

echo.
echo Waiting for the app to be ready...
timeout /t 8 /nobreak >nul
start "" http://localhost:8501
echo Opened http://localhost:8501
echo Stop with: docker compose down
pause
