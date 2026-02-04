cmake_minimum_required(VERSION 3.21)
project(aeqts_qubo_cuda LANGUAGES CXX)

# Define backend option: CUDA or HIP
set(BACKEND "CUDA" CACHE STRING "Choose the backend (CUDA or HIP)")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA" "HIP")

if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(SRC_FILES src/aeqts_qubo.cu)

    # CUDA specific settings
    set(CMAKE_CUDA_ARCHITECTURES "native")

    add_executable(aeqts_qubo ${SRC_FILES})

    set_target_properties(aeqts_qubo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )

elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(SRC_FILES src/aeqts_qubo.hip)

    add_executable(aeqts_qubo ${SRC_FILES})

    # Set C++ standard for HIP
    set_target_properties(aeqts_qubo PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
else()
    message(FATAL_ERROR "BACKEND must be either CUDA or HIP")
endif()
