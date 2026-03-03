#pragma once

#include <vector>

static std::vector<float> build_teacher_qubo_matrix_host(
    const std::vector<float>& values,
    const std::vector<float>& weights,
    float capacity,
    float P);
