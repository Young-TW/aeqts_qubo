cmake_minimum_required(VERSION 3.21)
project(aeqts_qubo_cuda LANGUAGES CXX)

set(BACKEND "CUDA" CACHE STRING "Choose the backend (CUDA or HIP)")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA" "HIP")

if(BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # if the user hasn't specified architectures, use "native" to target the current GPU
        # set(CMAKE_CUDA_ARCHITECTURES "native")
        # this repo is for h100/h200, so we can set the architecture to "90" for better performance
        set(CMAKE_CUDA_ARCHITECTURES "90")
    endif()

    enable_language(CUDA)

    include_directories(include)
    set(SRC_FILES src/main.cu src/kernels.cu src/qubo_matrix.cpp)

    add_executable(aeqts_qubo ${SRC_FILES})

    target_link_libraries(aeqts_qubo PRIVATE curand)

    set_target_properties(aeqts_qubo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )

elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)

    include_directories(include)

    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    find_package(hiprand REQUIRED)
    find_package(rocthrust REQUIRED)

    set(SRC_FILES src/main.hip)

    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()

    add_executable(aeqts_qubo ${SRC_FILES})

    target_link_libraries(aeqts_qubo PRIVATE
        hip::hiprand
        roc::rocthrust
    )

    set_target_properties(aeqts_qubo PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )

else()
    message(FATAL_ERROR "BACKEND must be either CUDA or HIP")
endif()
