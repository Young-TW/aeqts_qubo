#pragma once

// 極簡、零相依的單元測試框架。
// 用法:
//   #include "test_framework.h"
//   TEST(group, name) { EXPECT_EQ(1 + 1, 2); }
//   int main() { return aeqts_test::run_all(); }
//
// 提供 EXPECT_TRUE / EXPECT_FALSE / EXPECT_EQ / EXPECT_NEAR。
// 任一斷言失敗會記錄訊息並使該測試標記為失敗,但不中斷其餘測試。

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace aeqts_test {

struct TestCase {
    std::string group;
    std::string name;
    std::function<void(bool&)> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const std::string& group, const std::string& name,
              std::function<void(bool&)> fn) {
        registry().push_back({group, name, std::move(fn)});
    }
};

inline int run_all() {
    int failed = 0;
    for (const auto& t : registry()) {
        bool ok = true;
        t.fn(ok);
        if (ok) {
            std::printf("[ PASS ] %s.%s\n", t.group.c_str(), t.name.c_str());
        } else {
            std::printf("[ FAIL ] %s.%s\n", t.group.c_str(), t.name.c_str());
            ++failed;
        }
    }
    std::printf("\n%zu tests, %d failed\n", registry().size(), failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace aeqts_test

// 將群組與名稱串接成唯一識別字。
#define AEQTS_CONCAT_INNER(a, b) a##b
#define AEQTS_CONCAT(a, b) AEQTS_CONCAT_INNER(a, b)

#define TEST(group, name)                                                    \
    static void AEQTS_CONCAT(aeqts_test_fn_, __LINE__)(bool& _aeqts_ok);     \
    static aeqts_test::Registrar AEQTS_CONCAT(aeqts_test_reg_, __LINE__)(    \
        #group, #name, &AEQTS_CONCAT(aeqts_test_fn_, __LINE__));             \
    static void AEQTS_CONCAT(aeqts_test_fn_, __LINE__)(bool& _aeqts_ok)

#define EXPECT_TRUE(cond)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            _aeqts_ok = false;                                               \
            std::printf("    %s:%d EXPECT_TRUE(%s) failed\n", __FILE__,      \
                        __LINE__, #cond);                                    \
        }                                                                    \
    } while (0)

#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))

#define EXPECT_EQ(a, b)                                                       \
    do {                                                                     \
        auto _va = (a);                                                      \
        auto _vb = (b);                                                      \
        if (!(_va == _vb)) {                                                 \
            _aeqts_ok = false;                                               \
            std::printf("    %s:%d EXPECT_EQ(%s, %s) failed\n", __FILE__,    \
                        __LINE__, #a, #b);                                   \
        }                                                                    \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                                \
    do {                                                                     \
        double _va = (double)(a);                                           \
        double _vb = (double)(b);                                           \
        if (std::fabs(_va - _vb) > (tol)) {                                  \
            _aeqts_ok = false;                                               \
            std::printf(                                                     \
                "    %s:%d EXPECT_NEAR(%s, %s) failed: %.10g vs %.10g\n",    \
                __FILE__, __LINE__, #a, #b, _va, _vb);                       \
        }                                                                    \
    } while (0)
