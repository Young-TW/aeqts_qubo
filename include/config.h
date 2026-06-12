#ifndef AEQTS_CONFIG_H
#define AEQTS_CONFIG_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// 實驗參數,預設值與原本 main.cu 內建值相同。
struct Config {
    int iter = 1000;
    int n_items = 500;
    int N = 50;
    unsigned long long base_seed = 12345ULL;
    int run_id = 0;
    double P_penalty = 10.0;
};

namespace config_detail {

// 去除字串前後空白
inline std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

}  // namespace config_detail

// 從簡易的 `key = value;` 設定檔載入參數。
// 支援 `#` 與 `//` 註解、行尾分號、空白行。
// 回傳 false 表示檔案開啟失敗(沿用傳入的預設值)。
inline bool load_config(const std::string& path, Config& cfg) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;

        // 去除註解(# 或 //)
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        size_t slashes = line.find("//");
        if (slashes != std::string::npos) line = line.substr(0, slashes);

        // 去除行尾分號
        size_t semi = line.find(';');
        if (semi != std::string::npos) line = line.substr(0, semi);

        line = config_detail::trim(line);
        if (line.empty()) continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            std::cerr << "[config] 第 " << line_no << " 行格式錯誤(缺少 '='): "
                      << line << "\n";
            continue;
        }

        std::string key = config_detail::trim(line.substr(0, eq));
        std::string val = config_detail::trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) continue;

        try {
            if (key == "iter") {
                cfg.iter = std::stoi(val);
            } else if (key == "n_items") {
                cfg.n_items = std::stoi(val);
            } else if (key == "N") {
                cfg.N = std::stoi(val);
            } else if (key == "base_seed") {
                cfg.base_seed = std::stoull(val);
            } else if (key == "run_id") {
                cfg.run_id = std::stoi(val);
            } else if (key == "P_penalty") {
                cfg.P_penalty = std::stod(val);
            } else {
                std::cerr << "[config] 未知參數 '" << key << "' (第 " << line_no
                          << " 行),已忽略\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[config] 第 " << line_no << " 行數值解析失敗: " << key
                      << " = " << val << "\n";
        }
    }

    return true;
}

#endif  // AEQTS_CONFIG_H
