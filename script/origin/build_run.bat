@echo off
setlocal

echo [BUILD] Compiling CUDA knapsack...

nvcc -O2 -std=c++17 ^
    -arch=sm_75 ^
    -Xcompiler "/utf-8" ^
    -allow-unsupported-compiler ^
    src/main.cu ^
    -o main.exe

if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo [RUN] Executing program...
echo.

main.exe

echo.
echo [DONE]
pause
