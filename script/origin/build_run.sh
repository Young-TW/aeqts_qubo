#!/usr/bin/env bash

echo "[BUILD] Compiling CUDA knapsack..."

# nvcc compilation
# Note: Removed -Xcompiler "/utf-8" as that is specific to the Windows MSVC compiler.
nvcc -O2 -std=c++17 \
    -arch=sm_75 \
    -allow-unsupported-compiler \
    src/main.cu \
    -o main

# Check if the compilation command failed (return code not 0)
if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[RUN] Executing program..."
echo ""

# Execute the binary
./main

echo ""
echo "[DONE]"
