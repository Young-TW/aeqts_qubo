#include "qubo_matrix.h"

#include <cstdio>

// ------------------------------ Host: build QUBO matrix
// ------------------------------
std::vector<double> build_teacher_qubo_matrix_host(
    const std::vector<double>& values, const std::vector<double>& weights,
    double capacity, double P) {
    const int n = (int)values.size();
    std::vector<double> Q((size_t)n * (size_t)n, 0.0f);

    for (int i = 0; i < n; ++i) {
        double coeff_linear = -values[i] + 2.0 * P * (weights[i] * weights[i]) -
                             2.0 * P * capacity * weights[i];
        Q[(size_t)i * n + i] = coeff_linear;

        for (int j = i + 1; j < n; ++j) {
            double coeff_quad = 2.0 * P * weights[i] * weights[j];
            double v = coeff_quad * 0.5;
            Q[(size_t)i * n + j] = v;
            Q[(size_t)j * n + i] = v;
        }
    }
    return Q;
}
