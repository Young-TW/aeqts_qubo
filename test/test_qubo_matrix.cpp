#include "qubo_matrix.h"

#include <vector>

#include "test_framework.h"

namespace {

// 以矩陣計算二次型 x^T Q x(獨立於被測程式的實作)。
// Q 為 FP32,累加用 double 以免遮蔽矩陣本身的數值。
double quadratic_form(const std::vector<float>& Q,
                      const std::vector<unsigned char>& x) {
    const size_t n = x.size();
    double e = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!x[i]) continue;
        for (size_t j = 0; j < n; ++j) {
            if (x[j]) e += Q[i * n + j];
        }
    }
    return e;
}

}  // namespace

TEST(qubo_matrix, dimensions_are_n_by_n) {
    std::vector<float> values{1.0f, 2.0f, 3.0f};
    std::vector<float> weights{4.0f, 5.0f, 6.0f};
    auto Q = build_teacher_qubo_matrix_host(values, weights, 7.0f, 2.0f);
    EXPECT_EQ(Q.size(), (size_t)9);
}

TEST(qubo_matrix, empty_input_yields_empty_matrix) {
    std::vector<float> values;
    std::vector<float> weights;
    auto Q = build_teacher_qubo_matrix_host(values, weights, 1.0f, 1.0f);
    EXPECT_EQ(Q.size(), (size_t)0);
}

TEST(qubo_matrix, matrix_is_symmetric) {
    std::vector<float> values{3.0f, 1.5f, 9.0f, 2.0f};
    std::vector<float> weights{2.0f, 4.0f, 1.0f, 3.0f};
    const float C = 5.0f, P = 1.7f;
    const int n = 4;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(Q[i * n + j], Q[j * n + i], 1e-6);
        }
    }
}

TEST(qubo_matrix, diagonal_matches_formula) {
    std::vector<float> values{3.0f, 1.5f, 9.0f};
    std::vector<float> weights{2.0f, 4.0f, 1.0f};
    const float C = 5.0f, P = 1.7f;
    const int n = 3;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);
    for (int i = 0; i < n; ++i) {
        float expected = -values[i] + 2.0f * P * weights[i] * weights[i] -
                         2.0f * P * C * weights[i];
        EXPECT_NEAR(Q[i * n + i], expected, 1e-4);
    }
}

TEST(qubo_matrix, offdiagonal_matches_formula) {
    std::vector<float> values{3.0f, 1.5f, 9.0f};
    std::vector<float> weights{2.0f, 4.0f, 1.0f};
    const float C = 5.0f, P = 1.7f;
    const int n = 3;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // coeff_quad = 2*P*w_i*w_j,平均分配到上下三角各 0.5。
            float expected = P * weights[i] * weights[j];
            EXPECT_NEAR(Q[i * n + j], expected, 1e-4);
        }
    }
}

TEST(qubo_matrix, hand_computed_2x2) {
    // n=2, v=[10,20], w=[1,2], C=2, P=3 — 全為可精確表示的 float。
    // Q00 = -10 + 2*3*1 - 2*3*2*1 = -10 + 6 - 12 = -16
    // Q11 = -20 + 2*3*4 - 2*3*2*2 = -20 + 24 - 24 = -20
    // Q01 = Q10 = 3*1*2 = 6
    std::vector<float> values{10.0f, 20.0f};
    std::vector<float> weights{1.0f, 2.0f};
    auto Q = build_teacher_qubo_matrix_host(values, weights, 2.0f, 3.0f);
    EXPECT_NEAR(Q[0], -16.0, 1e-6);
    EXPECT_NEAR(Q[1], 6.0, 1e-6);
    EXPECT_NEAR(Q[2], 6.0, 1e-6);
    EXPECT_NEAR(Q[3], -20.0, 1e-6);
}

TEST(qubo_matrix, empty_selection_has_zero_energy) {
    std::vector<float> values{3.0f, 1.5f, 9.0f};
    std::vector<float> weights{2.0f, 4.0f, 1.0f};
    auto Q = build_teacher_qubo_matrix_host(values, weights, 5.0f, 1.7f);
    std::vector<unsigned char> x{0, 0, 0};
    EXPECT_NEAR(quadratic_form(Q, x), 0.0, 1e-9);
}

TEST(qubo_matrix, energy_matches_closed_form) {
    // x^T Q x 應等於 -V + P*Sww - 2*P*C*W + P*W^2
    //   其中 W = Σ w_i x_i, Sww = Σ w_i^2 x_i, V = Σ v_i x_i。
    std::vector<float> values{3.0f, 1.5f, 9.0f, 2.0f};
    std::vector<float> weights{2.0f, 4.0f, 1.0f, 3.0f};
    const float C = 5.0f, P = 1.7f;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);

    std::vector<std::vector<unsigned char>> selections = {
        {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 1, 0}, {1, 1, 1, 1}, {1, 0, 1, 1}};

    for (const auto& x : selections) {
        double W = 0, Sww = 0, V = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i]) {
                W += weights[i];
                Sww += (double)weights[i] * weights[i];
                V += values[i];
            }
        }
        double expected = -V + P * Sww - 2.0 * P * C * W + P * W * W;
        // FP32 矩陣 → 較寬鬆容差。
        EXPECT_NEAR(quadratic_form(Q, x), expected, 1e-3);
    }
}

int main() { return aeqts_test::run_all(); }
