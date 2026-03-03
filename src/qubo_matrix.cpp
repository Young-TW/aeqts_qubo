#include "qubo_matrix.h"

// ------------------------------ Host: build QUBO matrix ------------------------------
static std::vector<float> build_teacher_qubo_matrix_host(
    const std::vector<float>& values,
    const std::vector<float>& weights,
    float capacity,
    float P)
{
    const int n = (int)values.size();
    std::vector<float> Q((size_t)n * (size_t)n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float coeff_linear = -values[i] + 2.0f * P * (weights[i] * weights[i]) - 2.0f * P * capacity * weights[i];
        Q[(size_t)i * n + i] = coeff_linear;

        for (int j = i + 1; j < n; ++j) {
            float coeff_quad = 2.0f * P * weights[i] * weights[j];
            float v = coeff_quad * 0.5f;
            Q[(size_t)i * n + j] = v;
            Q[(size_t)j * n + i] = v;
        }
    }
    return Q;
}
