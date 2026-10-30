#include "ac/prompt_file.h"
#include <filesystem>
#include <cstdio>
#include <chrono>

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("ac-prompt-file-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    try {
        const std::string caption = "glitch hop, 92 bpm, D\xe2\x99\xad minor";
        {
            std::ofstream file(path, std::ios::binary);
            file << "\xef\xbb\xbf" << caption;
            if (!file)
                throw std::runtime_error("cannot write test caption");
        }
        if (ac::read_prompt_file(path.string()) != caption)
            throw std::runtime_error("UTF-8/BOM caption changed");
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file << " ";
        }
        if (ac::read_prompt_file(path.string()) != " ")
            throw std::runtime_error("caption whitespace was trimmed");
        std::filesystem::remove(path);
        bool rejected = false;
        try {
            ac::read_prompt_file(path.string());
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("missing caption file accepted");
        return 0;
    } catch (const std::exception &e) {
        std::filesystem::remove(path);
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
