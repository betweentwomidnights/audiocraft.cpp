#pragma once
#include "ac/prompt_pool.h"
#include "gguf_model.h"
#include "serve/http.h"
#include "httplib.h"
#include <map>

namespace ac::serve {
// Startup registry: clients select names, never supply arbitrary local paths.
struct AdapterCatalog {
    std::map<std::string, std::string> paths;
    void discover(const std::string &directory, const std::string &architecture) {
        if (!std::filesystem::is_directory(directory))
            return;
        for (const auto &e : std::filesystem::directory_iterator(directory)) {
            if (!e.is_regular_file() || e.path().extension() != ".gguf")
                continue;
            try {
                auto m = load_gguf_metadata(e.path().string().c_str());
                if (m.string("general.architecture") == architecture)
                    paths.emplace(e.path().stem().string(), e.path().string());
            } catch (const std::exception &ex) {
                fprintf(stderr, "[ac] skipped adapter %s: %s\n", e.path().filename().string().c_str(),
                        ex.what());
            }
        }
    }
    std::string path(const std::string &name) const {
        if (name.empty())
            return {};
        auto it = paths.find(name);
        if (it == paths.end())
            throw std::runtime_error("unknown adapter: " + name);
        return it->second;
    }
    void routes(httplib::Server &server) const {
        server.Get("/api/adapters", [this](const httplib::Request &, httplib::Response &res) {
            std::string list = "[";
            bool first = true;
            for (const auto &kv : paths) {
                if (!first)
                    list += ",";
                first = false;
                auto m = load_gguf_metadata(kv.second.c_str());
                list += JsonObject()
                            .str("name", kv.first)
                            .str("base_name", gguf_find_key(m.gguf, "ac.lora.base_name") >= 0
                                                  ? m.string("ac.lora.base_name")
                                                  : "")
                            .str("type", gguf_find_key(m.gguf, "ac.lora.type") >= 0 ? m.string("ac.lora.type")
                                                                                    : "lora")
                            .boolean("prompt_pool", std::filesystem::exists(prompt_pool_path(kv.second)))
                            .str();
            }
            res.set_content(JsonObject().raw("adapters", list + "]").str(), "application/json");
        });
        server.Get("/api/adapters/prompts", [this](const httplib::Request &req, httplib::Response &res) {
            try {
                if (!req.has_param("name"))
                    throw std::runtime_error("adapter name is required");
                auto p = path(req.get_param_value("name"));
                auto pool = read_prompt_pool(prompt_pool_path(p));
                std::string list = "[";
                for (size_t i = 0; i < pool.size(); ++i) {
                    if (i)
                        list += ",";
                    list += json_string(pool[i]);
                }
                res.set_content(JsonObject().raw("prompts", list + "]").str(), "application/json");
            } catch (const std::exception &ex) {
                res.status = 400;
                res.set_content(json_error(ex.what()), "application/json");
            }
        });
    }
};
} // namespace ac::serve
