@echo off
REM CircuitCheck Project Startup Helper
REM This script starts both backend and frontend servers
REM Run from the project root folder

echo === Starting CircuitCheck ===
echo.

REM Start backend Flask server
echo Starting backend server...
start "CircuitCheck Backend" powershell -NoProfile -ExecutionPolicy Bypass -Command "cd 'D:\CircuitCheck\backend'; python app.py"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak > nul

REM Start frontend React server
echo Starting frontend server...
start "CircuitCheck Frontend" powershell -NoProfile -ExecutionPolicy Bypass -Command "cd 'D:\CircuitCheck\frontend'; npm start"

echo.
echo === Services starting ===
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to close this window...
pause > nul