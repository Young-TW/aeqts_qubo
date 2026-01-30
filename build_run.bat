@echo off
setlocal

echo [BUILD] Compiling CUDA knapsack...

nvcc -O2 -std=c++17 ^
    -arch=sm_75 ^
    -Xcompiler "/utf-8" ^
    -allow-unsupported-compiler ^
    aeqts_knapsack_cuda.cu ^
    -o aeqts_knapsack_cuda.exe

if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo [RUN] Executing program...
echo.

aeqts_knapsack_cuda.exe

echo.
echo [DONE]
pause
