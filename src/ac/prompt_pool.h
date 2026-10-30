// Shared Gary smart-dice contract; caption strings are data, kept verbatim.
#pragma once
#include "yyjson.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac {
inline std::string prompt_pool_path(const std::string &adapter) {
    auto p = std::filesystem::path(adapter);
    p.replace_extension(".prompts.json");
    return p.string();
}
inline std::vector<std::string> read_prompt_pool(const std::string &path) {
    yyjson_read_err err{};
    auto doc = yyjson_read_file(path.c_str(), 0, nullptr, &err);
    if (!doc)
        throw std::runtime_error("cannot read prompt pool: " + path);
    struct Release {
        yyjson_doc *d;
        ~Release() { yyjson_doc_free(d); }
    } release{doc};
    auto root = yyjson_doc_get_root(doc);
    auto version = yyjson_obj_get(root, "version");
    auto prompts = yyjson_obj_get(yyjson_obj_get(root, "dice"), "instrumental");
    if (!yyjson_is_uint(version) || yyjson_get_uint(version) != 1 || !yyjson_is_arr(prompts))
        throw std::runtime_error("unsupported prompt pool: " + path);
    std::vector<std::string> result;
    size_t i, n;
    yyjson_val *item;
    yyjson_arr_foreach(prompts, i, n, item) {
        if (!yyjson_is_str(item) || !yyjson_get_len(item))
            throw std::runtime_error("invalid prompt pool caption");
        result.emplace_back(yyjson_get_str(item), yyjson_get_len(item));
    }
    if (result.empty())
        throw std::runtime_error("prompt pool is empty");
    return result;
}
inline std::string dice_prompt(const std::string &path, uint64_t seed) {
    const auto pool = read_prompt_pool(path);
    std::mt19937_64 rng(seed);
    // Stable mapping across standard libraries; independent of the audio RNG.
    return pool[(size_t)(rng() % pool.size())];
}
inline void write_prompt_pool(const std::string &path, const std::string &adapter,
                              const std::vector<std::string> &captions, size_t files) {
    if (std::filesystem::exists(path))
        throw std::runtime_error("refusing to overwrite prompt pool: " + path);
    auto doc = yyjson_mut_doc_new(nullptr);
    if (!doc)
        throw std::runtime_error("prompt pool allocation failed");
    struct Release {
        yyjson_mut_doc *d;
        ~Release() { yyjson_mut_doc_free(d); }
    } release{doc};
    auto root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_uint(doc, root, "version", 1);
    auto source = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "source", source);
    yyjson_mut_obj_add_strcpy(doc, source, "lora",
                              std::filesystem::path(adapter).filename().string().c_str());
    yyjson_mut_obj_add_uint(doc, source, "files", files);
    auto dice = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "dice", dice);
    auto array = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, dice, "instrumental", array);
    std::set<std::string> seen;
    for (const auto &caption : captions)
        if (!caption.empty() && seen.insert(caption).second)
            yyjson_mut_arr_add_strcpy(doc, array, caption.c_str());
    yyjson_mut_obj_add_uint(doc, source, "unique_prompts", seen.size());
    if (seen.empty())
        throw std::runtime_error("training captions contain no nonempty prompts");
    yyjson_write_err err{};
    if (!yyjson_mut_write_file(path.c_str(), doc, YYJSON_WRITE_PRETTY | YYJSON_WRITE_ESCAPE_UNICODE, nullptr,
                               &err))
        throw std::runtime_error("cannot write prompt pool: " + path);
}
} // namespace ac
