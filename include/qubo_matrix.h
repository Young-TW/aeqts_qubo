#pragma once

#include <vector>

std::vector<double> build_teacher_qubo_matrix_host(
    const std::vector<double>& values, const std::vector<double>& weights,
    double capacity, double P);
