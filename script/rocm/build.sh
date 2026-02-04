#!/usr/bin/env bash

apt-get update && apt-get install -y \
  cmake \
  hiprand-dev \
  rocrand-dev \
  rocthrust-dev

cmake -B build -DCMAKE_BUILD_TYPE=Release -DBACKEND=HIP
cmake --build build --config Release
