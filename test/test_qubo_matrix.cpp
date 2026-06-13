#include "qubo_matrix.h"

#include <vector>

#include "test_framework.h"

namespace {

// 以矩陣計算二次型 x^T Q x(獨立於被測程式的實作)。
double quadratic_form(const std::vector<double>& Q,
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
    std::vector<double> values{1.0, 2.0, 3.0};
    std::vector<double> weights{4.0, 5.0, 6.0};
    auto Q = build_teacher_qubo_matrix_host(values, weights, 7.0, 2.0);
    EXPECT_EQ(Q.size(), (size_t)9);
}

TEST(qubo_matrix, empty_input_yields_empty_matrix) {
    std::vector<double> values;
    std::vector<double> weights;
    auto Q = build_teacher_qubo_matrix_host(values, weights, 1.0, 1.0);
    EXPECT_EQ(Q.size(), (size_t)0);
}

TEST(qubo_matrix, matrix_is_symmetric) {
    std::vector<double> values{3.0, 1.5, 9.0, 2.0};
    std::vector<double> weights{2.0, 4.0, 1.0, 3.0};
    const double C = 5.0, P = 1.7;
    const int n = 4;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(Q[i * n + j], Q[j * n + i], 1e-12);
        }
    }
}

TEST(qubo_matrix, diagonal_matches_formula) {
    std::vector<double> values{3.0, 1.5, 9.0};
    std::vector<double> weights{2.0, 4.0, 1.0};
    const double C = 5.0, P = 1.7;
    const int n = 3;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);
    for (int i = 0; i < n; ++i) {
        double expected = -values[i] + 2.0 * P * weights[i] * weights[i] -
                          2.0 * P * C * weights[i];
        EXPECT_NEAR(Q[i * n + i], expected, 1e-9);
    }
}

TEST(qubo_matrix, offdiagonal_matches_formula) {
    std::vector<double> values{3.0, 1.5, 9.0};
    std::vector<double> weights{2.0, 4.0, 1.0};
    const double C = 5.0, P = 1.7;
    const int n = 3;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // coeff_quad = 2*P*w_i*w_j,平均分配到上下三角各 0.5。
            double expected = P * weights[i] * weights[j];
            EXPECT_NEAR(Q[i * n + j], expected, 1e-9);
        }
    }
}

TEST(qubo_matrix, hand_computed_2x2) {
    // n=2, v=[10,20], w=[1,2], C=2, P=3
    // Q00 = -10 + 2*3*1 - 2*3*2*1 = -10 + 6 - 12 = -16
    // Q11 = -20 + 2*3*4 - 2*3*2*2 = -20 + 24 - 24 = -20
    // Q01 = Q10 = 3*1*2 = 6
    std::vector<double> values{10.0, 20.0};
    std::vector<double> weights{1.0, 2.0};
    auto Q = build_teacher_qubo_matrix_host(values, weights, 2.0, 3.0);
    EXPECT_NEAR(Q[0], -16.0, 1e-9);
    EXPECT_NEAR(Q[1], 6.0, 1e-9);
    EXPECT_NEAR(Q[2], 6.0, 1e-9);
    EXPECT_NEAR(Q[3], -20.0, 1e-9);
}

TEST(qubo_matrix, empty_selection_has_zero_energy) {
    std::vector<double> values{3.0, 1.5, 9.0};
    std::vector<double> weights{2.0, 4.0, 1.0};
    auto Q = build_teacher_qubo_matrix_host(values, weights, 5.0, 1.7);
    std::vector<unsigned char> x{0, 0, 0};
    EXPECT_NEAR(quadratic_form(Q, x), 0.0, 1e-12);
}

TEST(qubo_matrix, energy_matches_closed_form) {
    // x^T Q x 應等於 -V + P*Sww - 2*P*C*W + P*W^2
    //   其中 W = Σ w_i x_i, Sww = Σ w_i^2 x_i, V = Σ v_i x_i。
    std::vector<double> values{3.0, 1.5, 9.0, 2.0};
    std::vector<double> weights{2.0, 4.0, 1.0, 3.0};
    const double C = 5.0, P = 1.7;
    auto Q = build_teacher_qubo_matrix_host(values, weights, C, P);

    std::vector<std::vector<unsigned char>> selections = {
        {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 1, 0}, {1, 1, 1, 1}, {1, 0, 1, 1}};

    for (const auto& x : selections) {
        double W = 0, Sww = 0, V = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i]) {
                W += weights[i];
                Sww += weights[i] * weights[i];
                V += values[i];
            }
        }
        double expected = -V + P * Sww - 2.0 * P * C * W + P * W * W;
        EXPECT_NEAR(quadratic_form(Q, x), expected, 1e-6);
    }
}

int main() { return aeqts_test::run_all(); }
