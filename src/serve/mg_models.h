#pragma once
// gary's model picker, in the shape `g4l_localhost.py`'s `/api/models` returns it.
//
// gary4juce builds its dropdown from that reply and sends the chosen entry's `path` back as
// `model_name`. It only reads objects: `{type: "single", name, path}`, or a `group` whose
// `checkpoints` are `{name, path, epoch}`. A bare string is skipped, so a list of names
// leaves the plugin with an empty dropdown and its hardcoded fallback,
// `thepatch/vanya_ai_dnb_0.1`, which then has to resolve here too.
//
// So each model is listed by the Hugging Face repo it was converted from, which the
// converter records in the gguf, and the server accepts either that or the file's own name.
#include "serve/http.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace ac::serve {

struct MgModel {
    std::string key;   // from the filename: `musicgen-vanya-dnb-0.4B-v1.0-F16.gguf` -> vanya-dnb
    std::string id;    // the source repo, `thepatch/vanya_ai_dnb_0.1`; the key when unrecorded
    std::string size;  // small, medium or large
};

// `https://huggingface.co/thepatch/vanya_ai_dnb_0.1` -> `thepatch/vanya_ai_dnb_0.1`.
// Anything else is not a repo id, and an empty result says so.
inline std::string repo_id(const std::string& url) {
    const std::string prefix = "https://huggingface.co/";
    if (url.rfind(prefix, 0) != 0) return {};
    std::string id = url.substr(prefix.size());
    while (!id.empty() && id.back() == '/') id.pop_back();
    return id.find('/') == std::string::npos ? std::string() : id;
}

// One size's entries, grouped as the Python groups them: a name ending in `-<digits>` is a
// checkpoint of everything sharing the part before it, and anything else stands alone.
inline std::string mg_model_group(const std::vector<MgModel>& models) {
    struct Entry { std::string name, path; bool checkpoint = false; long epoch = 0; };
    std::vector<std::string> order;
    std::map<std::string, std::vector<Entry>> groups;
    for (const auto& m : models) {
        const std::string name = m.id.substr(m.id.rfind('/') + 1);
        Entry e{name, m.id};
        std::string base = name;
        const size_t dash = name.rfind('-');
        if (dash != std::string::npos && dash + 1 < name.size() &&
            name.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
            e.checkpoint = true;
            e.epoch = std::stol(name.substr(dash + 1));
            base = name.substr(0, dash);
        }
        if (!groups.count(base)) order.push_back(base);
        groups[base].push_back(e);
    }
    std::string out = "[";
    for (size_t g = 0; g < order.size(); ++g) {
        if (g) out += ',';
        auto& entries = groups[order[g]];
        if (entries.size() == 1 && !entries[0].checkpoint) {
            out += "{\"name\":" + json_string(entries[0].name) + ",\"path\":" +
                   json_string(entries[0].path) + ",\"type\":\"single\"}";
            continue;
        }
        // A plain name that shares a base with checkpoints is dropped from the group, as
        // the Python drops it.
        std::vector<Entry> checkpoints;
        for (const auto& e : entries) if (e.checkpoint) checkpoints.push_back(e);
        std::sort(checkpoints.begin(), checkpoints.end(),
                  [](const Entry& a, const Entry& b) { return a.epoch < b.epoch; });
        out += "{\"name\":" + json_string(order[g]) + ",\"type\":\"group\",\"checkpoints\":[";
        for (size_t i = 0; i < checkpoints.size(); ++i) {
            if (i) out += ',';
            out += "{\"name\":" + json_string(order[g] + "-" + std::to_string(checkpoints[i].epoch)) +
                   ",\"path\":" + json_string(checkpoints[i].path) +
                   ",\"epoch\":" + std::to_string(checkpoints[i].epoch) + "}";
        }
        out += "]}";
    }
    return out + "]";
}

inline std::string mg_models_json(const std::vector<MgModel>& models) {
    std::string out = "{\"success\":true,\"models\":{";
    const char* sizes[] = {"small", "medium", "large"};
    for (int s = 0; s < 3; ++s) {
        std::vector<MgModel> group;
        for (const auto& m : models) if (m.size == sizes[s]) group.push_back(m);
        if (s) out += ',';
        out += json_string(sizes[s]) + ":" + mg_model_group(group);
    }
    return out + "},\"source\":\"audiocraft.cpp\"}";
}

// The key a request's `model_name` means: the key itself, or the repo id listed for it.
inline std::string mg_model_key(const std::vector<MgModel>& models, const std::string& name) {
    for (const auto& m : models) if (m.key == name) return m.key;
    for (const auto& m : models) if (m.id == name) return m.key;
    return {};
}

} // namespace ac::serve
