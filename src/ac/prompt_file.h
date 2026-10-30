#pragma once
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace ac {
// UTF-8 caption bytes bypass Windows' ANSI conversion of narrow argv.
inline std::string read_prompt_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot read prompt file: " + path);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad())
        throw std::runtime_error("failed reading prompt file: " + path);
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    return text;
}
} // namespace ac
