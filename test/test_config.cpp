#include "config.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "test_framework.h"

namespace {

// 以 mkstemp 安全建立暫存檔,寫入內容後回傳其路徑。
std::string write_temp(const std::string& contents) {
    char tmpl[] = "/tmp/aeqts_test_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd != -1) close(fd);
    std::string path(tmpl);
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

}  // namespace

TEST(config, defaults_are_expected) {
    Config cfg;
    EXPECT_EQ(cfg.iter, 1000);
    EXPECT_EQ(cfg.n_items, 500);
    EXPECT_EQ(cfg.N, 50);
    EXPECT_EQ(cfg.base_seed, 12345ULL);
    EXPECT_EQ(cfg.run_id, 0);
    EXPECT_NEAR(cfg.P_penalty, 10.0, 1e-12);
}

TEST(config, missing_file_returns_false_and_keeps_defaults) {
    Config cfg;
    bool ok = load_config("/no/such/path/definitely_missing.conf", cfg);
    EXPECT_FALSE(ok);
    EXPECT_EQ(cfg.iter, 1000);
    EXPECT_EQ(cfg.n_items, 500);
}

TEST(config, parses_all_keys) {
    std::string path = write_temp(
        "iter = 200\n"
        "n_items = 30\n"
        "N = 8\n"
        "base_seed = 99\n"
        "run_id = 4\n"
        "P_penalty = 2.5\n");
    Config cfg;
    bool ok = load_config(path, cfg);
    std::remove(path.c_str());

    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.iter, 200);
    EXPECT_EQ(cfg.n_items, 30);
    EXPECT_EQ(cfg.N, 8);
    EXPECT_EQ(cfg.base_seed, 99ULL);
    EXPECT_EQ(cfg.run_id, 4);
    EXPECT_NEAR(cfg.P_penalty, 2.5, 1e-12);
}

TEST(config, trailing_semicolon_and_hash_comment) {
    std::string path = write_temp(
        "iter = 7;        # 迭代代數\n"
        "N = 3;           # 種群\n");
    Config cfg;
    load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_EQ(cfg.iter, 7);
    EXPECT_EQ(cfg.N, 3);
}

TEST(config, double_slash_comment) {
    std::string path = write_temp("iter = 42 // inline comment\n");
    Config cfg;
    load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_EQ(cfg.iter, 42);
}

TEST(config, blank_and_comment_only_lines_ignored) {
    std::string path = write_temp(
        "\n"
        "   \n"
        "# full line comment\n"
        "// another comment\n"
        "iter = 11\n");
    Config cfg;
    load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_EQ(cfg.iter, 11);
}

TEST(config, whitespace_around_key_and_value_trimmed) {
    std::string path = write_temp("   N    =     16   \n");
    Config cfg;
    load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_EQ(cfg.N, 16);
}

TEST(config, unknown_key_is_ignored_keeps_other_values) {
    std::string path = write_temp(
        "bogus = 123\n"
        "iter = 55\n");
    Config cfg;
    bool ok = load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.iter, 55);
    EXPECT_EQ(cfg.n_items, 500);  // 未受影響
}

TEST(config, unspecified_keys_keep_defaults) {
    std::string path = write_temp("iter = 1\n");
    Config cfg;
    load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_EQ(cfg.iter, 1);
    EXPECT_EQ(cfg.N, 50);
    EXPECT_EQ(cfg.n_items, 500);
}

TEST(config, malformed_number_leaves_default) {
    // std::stoi 解析失敗會被攔截,該行被忽略。
    std::string path = write_temp("iter = not_a_number\n");
    Config cfg;
    bool ok = load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.iter, 1000);
}

TEST(config, line_without_equals_is_skipped) {
    std::string path = write_temp(
        "this line has no equals sign\n"
        "N = 9\n");
    Config cfg;
    bool ok = load_config(path, cfg);
    std::remove(path.c_str());
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.N, 9);
}

TEST(config_trim, removes_surrounding_whitespace) {
    using config_detail::trim;
    EXPECT_TRUE(trim("  hello  ") == "hello");
    EXPECT_TRUE(trim("\t\r\n x \n") == "x");
    EXPECT_TRUE(trim("nospace") == "nospace");
    EXPECT_TRUE(trim("   ") == "");
    EXPECT_TRUE(trim("") == "");
}

int main() { return aeqts_test::run_all(); }
