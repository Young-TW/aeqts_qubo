cmake_minimum_required(VERSION 3.21)
project(aeqts_qubo LANGUAGES CXX)

set(BACKEND "CUDA" CACHE STRING "Choose the backend (CUDA or HIP)")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA" "HIP")

# add MPI option, default is OFF (Standalone)
option(ENABLE_MPI "Enable MPI support (CUDA-aware or HIP-aware)" OFF)

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

include_directories(include)

if(BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "90")
    endif()

    enable_language(CUDA)

    set(SRC_FILES src/main.cu src/kernels.cu src/qubo_matrix.cpp)
    add_executable(aeqts_qubo ${SRC_FILES})

    target_link_libraries(aeqts_qubo PRIVATE curand)

    if(ENABLE_MPI)
        target_link_libraries(aeqts_qubo PRIVATE MPI::MPI_CXX)
        target_compile_definitions(aeqts_qubo PRIVATE USE_MPI)
    endif()

    set_target_properties(aeqts_qubo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )

elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)

    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    find_package(hiprand REQUIRED)
    find_package(rocthrust REQUIRED)

    set(SRC_FILES src/origin/aeqts_qubo.hip)

    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()

    add_executable(aeqts_qubo ${SRC_FILES})

    target_link_libraries(aeqts_qubo PRIVATE
        hip::hiprand
        roc::rocthrust
    )

    if(ENABLE_MPI)
        target_link_libraries(aeqts_qubo PRIVATE MPI::MPI_CXX)
        target_compile_definitions(aeqts_qubo PRIVATE USE_MPI)
    endif()

    set_target_properties(aeqts_qubo PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )

else()
    message(FATAL_ERROR "BACKEND must be either CUDA or HIP")
endif()
