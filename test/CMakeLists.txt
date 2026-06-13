# 純 host C++ 單元測試,無需 CUDA/HIP 工具鏈即可建置。
#
# 既可由根 CMakeLists 以 -DBUILD_TESTS=ON 一併納入,
# 也可單獨建置(方便在沒有 GPU 工具鏈的機器上跑):
#   cmake -S test -B build-test
#   cmake --build build-test
#   ctest --test-dir build-test --output-on-failure
cmake_minimum_required(VERSION 3.21)

# 以 STANDALONE 判斷是否被當作獨立專案配置。
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(aeqts_qubo_tests LANGUAGES CXX)
    enable_testing()
    set(AEQTS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    set(AEQTS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
else()
    set(AEQTS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
    set(AEQTS_SRC_DIR "${CMAKE_SOURCE_DIR}/src")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QUBO 矩陣建構(qubo_matrix.cpp 為純 host 程式碼)
add_executable(test_qubo_matrix
    test_qubo_matrix.cpp
    "${AEQTS_SRC_DIR}/qubo_matrix.cpp")
target_include_directories(test_qubo_matrix PRIVATE
    "${AEQTS_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

# 設定檔解析(config.h 為 header-only)
add_executable(test_config test_config.cpp)
target_include_directories(test_config PRIVATE
    "${AEQTS_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME qubo_matrix COMMAND test_qubo_matrix)
add_test(NAME config COMMAND test_config)
