// musicgen-server -- MusicGen's `generate_continuation` over HTTP, on :8000.
//
// A drop-in for gary4local's `g4l_localhost.py`, so `service_manager.rs` only changes
// `entryPoint` from `python.exe g4l_localhost.py` to this binary and skips building a venv.
// The routes and reply shapes are the service's, because gary4juce is already reading them:
//
//   GET  /health
//   GET  /api/models                     grouped by size, from what is on disk
//   POST /api/juce/process_audio         continue from the *first* N seconds
//   POST /api/juce/continue_music        continue from the *last* N seconds
//   POST /api/juce/retry_music           the same input again, new parameters
//   GET  /api/juce/poll_status/<id>      -> progress, then {audio_data} on completion
//
// One divergence, and it is the point of the exercise: `/api/models` reports the GGUFs
// actually present rather than a hardcoded list of `thepatch/*` repositories. A native
// service cannot fetch and convert a PyTorch checkpoint on demand, and offering a model the
// user has not converted would fail at generation time instead of at the picker.
#include "ac/codec.h"
#include "ac/conditioner.h"
#include "gguf_model.h"
#include "mg/encodec.h"
#include "mg/pipeline.h"
#include "serve/http.h"

#include "httplib.h"
#include "yyjson.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace ac::serve;

std::string g_models_dir;
std::string g_device;
std::string g_codec, g_t5;
Sessions g_sessions;

// name -> gguf path, discovered at startup.
std::map<std::string, std::string> g_lms;

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

std::string find_model(const std::string& stem) {
    for (const char* suffix : {"-F16.gguf", "-F32.gguf"}) {
        const std::string path = g_models_dir + "/" + stem + suffix;
        if (file_exists(path)) return path;
    }
    return {};
}

// Every `musicgen-*.gguf` in the models directory, keyed by the part between the prefix and
// the version. `musicgen-vanya-dnb-0.4B-v1.0-F16.gguf` becomes `vanya-dnb`.
//
// F16 wins over F32 for the same name: it is half the file for a difference that never
// reaches the tokens gary samples.
void discover_models() {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(g_models_dir, ec)) return;
    std::map<std::string, std::string> f32;
    for (const auto& entry : fs::directory_iterator(g_models_dir, ec)) {
        const std::string file = entry.path().filename().string();
        if (file.rfind("musicgen-", 0) != 0) continue;
        const size_t version = file.find("-v1.0-");
        if (version == std::string::npos) continue;
        std::string name = file.substr(9, version - 9);
        // Trim the size label the artifact convention appends, e.g. "-0.4B".
        const size_t dash = name.rfind('-');
        if (dash != std::string::npos && name.size() - dash >= 3 &&
            (name.back() == 'B' || name.back() == 'M')) {
            name = name.substr(0, dash);
        }
        const std::string path = entry.path().string();
        if (file.find("-F16.gguf") != std::string::npos) g_lms[name] = path;
        else if (file.find("-F32.gguf") != std::string::npos) f32[name] = path;
    }
    for (const auto& kv : f32) g_lms.emplace(kv.first, kv.second);
}

// The picker groups by size; the parameter count is in the gguf, so read it rather than
// guessing from the name.
std::string size_group(const std::string& path) {
    try {
        ac::GgufModel m = ac::load_gguf_metadata(path.c_str());
        const uint64_t params = m.u64("ac.lm.parameter_count");
        if (params < 700ull * 1000 * 1000) return "small";
        if (params < 2000ull * 1000 * 1000) return "medium";
        return "large";
    } catch (...) {
        return "small";
    }
}

struct Json {
    yyjson_doc* doc = nullptr;
    yyjson_val* root = nullptr;
    explicit Json(const std::string& body) {
        doc = yyjson_read(body.data(), body.size(), 0);
        root = doc ? yyjson_doc_get_root(doc) : nullptr;
        if (!root || !yyjson_is_obj(root)) throw std::runtime_error("body is not a JSON object");
    }
    ~Json() { if (doc) yyjson_doc_free(doc); }
    Json(const Json&) = delete;
    Json& operator=(const Json&) = delete;

    std::string str(const char* key, const std::string& fallback = {}) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : fallback;
    }
    double num(const char* key, double fallback) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (!v || yyjson_is_null(v)) return fallback;
        if (yyjson_is_num(v)) return yyjson_get_num(v);
        if (yyjson_is_str(v)) { try { return std::stod(yyjson_get_str(v)); } catch (...) {} }
        return fallback;
    }
    long long seed() const {
        yyjson_val* v = yyjson_obj_get(root, "seed");
        if (!v || yyjson_is_null(v)) return -1;
        if (yyjson_is_int(v)) return yyjson_get_sint(v);
        if (yyjson_is_num(v)) return (long long)yyjson_get_num(v);
        if (yyjson_is_str(v)) {
            const std::string s = yyjson_get_str(v);
            if (s.empty()) return -1;
            try { return std::stoll(s); } catch (...) { return -1; }
        }
        return -1;
    }
};

// --- generation ------------------------------------------------------------------------------

struct GenerateRequest {
    std::string audio_b64;
    std::string model;
    std::string description;
    double prompt_duration = 6.0;
    double duration = 30.0;
    int top_k = 250;
    float temperature = 1.0f;
    float cfg_coef = 3.0f;
    uint64_t seed = 1234;
    // `process_audio` primes from the head of the clip, `continue_music` from the tail.
    bool from_start = false;
};

// gary's transform, staged the way mg-generate stages it: text, codec, LM, codec.
std::string run_generation(const GenerateRequest& req, const std::string& session) {
    const auto it = g_lms.find(req.model);
    if (it == g_lms.end()) throw std::runtime_error("unknown model: " + req.model);

    const ac::TextCondition cond = ac::encode_prompt(g_t5, req.description, g_device);

    std::vector<int32_t> prompt_codes;
    int prompt_len = 0, frame_rate = 50, sample_rate = 32000, channels = 1;
    {
        ac::GgufModel codec = ac::load_gguf(
            g_codec.c_str(), ac::make_backend(0, g_device.empty() ? nullptr : g_device.c_str()));
        const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
        frame_rate = c.frame_rate;
        sample_rate = c.sample_rate;
        channels = c.channels;

        int n_samples = 0, n_ch = 0, rate = 0;
        std::vector<float> audio = decode_audio(req.audio_b64, n_samples, n_ch, rate);
        ac::conform_audio(audio, n_samples, n_ch, rate, c,
                          (int)(req.prompt_duration * c.frame_rate), !req.from_start);
        prompt_len = n_samples / c.hop_length;
        const std::vector<float> latent = ac::codec_encode(codec, c, audio, n_samples, n_ch);
        const ac::RvqCodebooks books =
            ac::rvq_load(codec, c.rvq_n_q, c.rvq_bins, c.latent_dim);
        prompt_codes = ac::rvq_encode(books, latent, prompt_len);
    }

    std::vector<int32_t> codes;
    int timesteps = 0;
    {
        ac::GgufModel lm = ac::load_gguf(
            it->second.c_str(), ac::make_backend(0, g_device.empty() ? nullptr : g_device.c_str()));
        const ac::LmConfig c = ac::LmConfig::from(lm);
        ac::GenerateParams params;
        params.max_gen_len = (int)(req.duration * frame_rate);
        params.use_sampling = true;
        params.temperature = req.temperature;
        params.top_k = req.top_k;
        params.cfg_coef = req.cfg_coef;
        params.seed = req.seed;
        timesteps = params.max_gen_len;

        ac::GenerateReport report;
        auto progress = [&session](int done, int total) {
            g_sessions.set_progress(session, done, total);
        };
        codes = ac::mg_generate(lm, c, cond, prompt_codes, prompt_len, params, progress, report);
    }

    ac::GgufModel codec = ac::load_gguf(
        g_codec.c_str(), ac::make_backend(0, g_device.empty() ? nullptr : g_device.c_str()));
    const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
    const ac::RvqCodebooks books = ac::rvq_load(codec, c.rvq_n_q, c.rvq_bins, c.latent_dim);
    const std::vector<float> latent = ac::rvq_decode(books, codes, timesteps);
    int64_t out_samples = 0;
    std::vector<float> audio = ac::codec_decode(codec, c, latent, timesteps, out_samples);
    // Headroom, not loudness -- see ac::peak_normalize_if_clipping. gary does this in
    // `save_audio_to_base64`, and without it the 16-bit conversion flattens every transient
    // the decoder overshot on.
    ac::peak_normalize_if_clipping(audio);
    return encode_audio(audio, (int)out_samples, channels, sample_rate);
}

GenerateRequest parse_generate(const Json& body, bool from_start) {
    GenerateRequest req;
    req.audio_b64 = body.str("audio_data");
    req.model = body.str("model_name");
    req.description = body.str("description");
    req.prompt_duration = body.num("prompt_duration", 6.0);
    req.duration = body.num("duration", 30.0);
    req.top_k = (int)body.num("top_k", 250);
    req.temperature = (float)body.num("temperature", 1.0);
    req.cfg_coef = (float)body.num("cfg_coef", 3.0);
    req.seed = resolve_seed(body.seed());
    req.from_start = from_start;
    return req;
}

// The shared tail of the three generation routes: validate, open a session, start the job.
void start_generation(GenerateRequest req, const char* started_message,
                      httplib::Response& res) {
    if (req.audio_b64.empty()) {
        res.status = 400;
        res.set_content(json_error("audio_data is required"), "application/json");
        return;
    }
    if (req.model.empty()) {
        if (g_lms.empty()) {
            res.status = 503;
            res.set_content(json_error("no musicgen gguf in " + g_models_dir),
                            "application/json");
            return;
        }
        req.model = g_lms.begin()->first;
    }
    if (!g_lms.count(req.model)) {
        res.status = 400;
        res.set_content(json_error("Unknown model: " + req.model), "application/json");
        return;
    }

    Session initial;
    initial.status = "queued";
    initial.seed = req.seed;
    initial.model = req.model;
    initial.prompt_duration = (int)req.prompt_duration;
    initial.description = req.description;
    // Kept so `retry_music` can run the same input again: the Python service is explicit
    // that a retry reuses the last *input* audio, not the last result.
    initial.original_b64 = req.audio_b64;
    const std::string id = g_sessions.create(initial);

    run_job(g_sessions, id, [req](const std::string& session) {
        const std::string out = run_generation(req, session);
        g_sessions.update(session, [&](Session& s) {
            s.status = "completed";
            s.progress = 100;
            s.audio_b64 = out;
            s.finished = now_s();
        });
    });

    res.set_content(JsonObject()
                        .boolean("success", true)
                        .str("session_id", id)
                        .num("seed", (double)req.seed)
                        .str("message", started_message)
                        .str(),
                    "application/json");
}

void usage() {
    fprintf(stderr,
        "usage: musicgen-server [--port 8000] [--models-dir DIR] [--device NAME]\n"
        "\n"
        "Serves MusicGen continuation on the same routes as gary4local's gary service.\n"
        "Every musicgen-*.gguf in --models-dir (or AC_MODELS_DIR) is offered by\n"
        "/api/models; encodec-32khz-* and t5-base-encoder-* are also required.\n");
}

} // namespace

int main(int argc, char** argv) {
    int port = 8000;
    std::string host = "127.0.0.1";
    g_models_dir = env_or("AC_MODELS_DIR", "models");
    g_device = env_or("AC_DEVICE", "");

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--port") port = atoi(next("--port").c_str());
        else if (a == "--host") host = next("--host");
        else if (a == "--models-dir") g_models_dir = next("--models-dir");
        else if (a == "--device") g_device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }

    g_codec = find_model("encodec-32khz-v1.0");
    g_t5 = find_model("t5-base-encoder-0.1B-v1.0");
    discover_models();
    if (g_codec.empty()) fprintf(stderr, "[ac] warning: no encodec gguf in %s\n",
                                 g_models_dir.c_str());
    if (g_t5.empty()) fprintf(stderr, "[ac] warning: no t5 gguf in %s\n", g_models_dir.c_str());
    fprintf(stderr, "[ac] %zu musicgen model(s) in %s\n", g_lms.size(), g_models_dir.c_str());
    for (const auto& kv : g_lms) fprintf(stderr, "[ac]   %s -> %s\n", kv.first.c_str(),
                                         kv.second.c_str());

    httplib::Server server;
    server.set_payload_max_length(256ull * 1024 * 1024);

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        const bool ready = !g_codec.empty() && !g_t5.empty() && !g_lms.empty();
        JsonObject o;
        o.str("status", ready ? "live" : "degraded")
         .str("service", "gary4juce-localhost")
         .str("session_store", "live")            // in-process; nothing external to be down
         .str("backend", "audiocraft.cpp")
         .str("models_dir", g_models_dir)
         .str("device", g_device.empty() ? "auto" : g_device)
         .num("models", (double)g_lms.size());
        res.status = ready ? 200 : 500;
        res.set_content(o.str(), "application/json");
    });

    server.Get("/api/models", [](const httplib::Request&, httplib::Response& res) {
        std::map<std::string, std::vector<std::string>> groups{
            {"small", {}}, {"medium", {}}, {"large", {}}};
        for (const auto& kv : g_lms) groups[size_group(kv.second)].push_back(kv.first);

        std::string body = "{\"models\":{";
        bool first_group = true;
        for (const auto& g : groups) {
            if (!first_group) body += ',';
            first_group = false;
            body += json_string(g.first) + ":[";
            for (size_t i = 0; i < g.second.size(); ++i) {
                if (i) body += ',';
                body += json_string(g.second[i]);
            }
            body += ']';
        }
        body += "},\"source\":\"audiocraft.cpp\"}";
        res.set_content(body, "application/json");
    });

    auto generation_route = [](bool from_start, const char* message) {
        return [from_start, message](const httplib::Request& req, httplib::Response& res) {
            try {
                Json body(req.body);
                start_generation(parse_generate(body, from_start), message, res);
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json_error(e.what()), "application/json");
            }
        };
    };

    // `process_audio` primes from the head of the clip; `continue_music` from the tail.
    server.Post("/api/juce/process_audio",
                generation_route(true, "Audio processing started"));
    server.Post("/api/juce/continue_music",
                generation_route(false, "Continue processing started"));

    // A retry re-runs the *input* the previous session was given, with whatever parameters
    // the client now sends. Reusing the previous result instead would compound the
    // continuation, which is the bug the Python file's comments call out.
    server.Post("/api/juce/retry_music", [](const httplib::Request& req,
                                            httplib::Response& res) {
        try {
            Json body(req.body);
            const std::string old_id = body.str("session_id");
            Session previous;
            if (old_id.empty() || !g_sessions.get(old_id, previous)) {
                res.status = 404;
                res.set_content(json_error("Original session not found"), "application/json");
                return;
            }
            if (previous.original_b64.empty()) {
                res.status = 404;
                res.set_content(json_error("No last input audio found for retry"),
                                "application/json");
                return;
            }
            GenerateRequest req2 = parse_generate(body, /*from_start=*/false);
            req2.audio_b64 = previous.original_b64;
            if (req2.model.empty()) req2.model = previous.model;
            if (!body.num("prompt_duration", 0.0) && previous.prompt_duration > 0)
                req2.prompt_duration = previous.prompt_duration;
            if (req2.description.empty()) req2.description = previous.description;
            start_generation(req2, "Retry processing started", res);
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    server.Get(R"(/api/juce/poll_status/(\w+))",
               [](const httplib::Request& req, httplib::Response& res) {
        Session s;
        if (!g_sessions.get(req.matches[1], s)) {
            // The Python service answers 200 with status "unknown" for an id it has never
            // seen, and gary4juce treats a 404 as a hard error, so match it.
            res.set_content(JsonObject()
                                .boolean("success", true)
                                .str("status", "unknown")
                                .num("progress", 0)
                                .boolean("generation_in_progress", false)
                                .raw("queue_status", "{}")
                                .str(),
                            "application/json");
            return;
        }
        const bool running = s.status == "warming" || s.status == "processing" ||
                             s.status == "queued";
        JsonObject o;
        o.boolean("success", true)
         .str("status", s.status)
         .num("progress", s.progress)
         .num("generation_seed", (double)s.seed)
         .raw("queue_status", "{}")
         .boolean("generation_in_progress", running);
        if (s.status == "completed" && !s.audio_b64.empty()) o.str("audio_data", s.audio_b64);
        if (s.status == "failed") o.str("error", s.error.empty() ? "Unknown error" : s.error);
        res.set_content(o.str(), "application/json");
    });

    fprintf(stderr, "[ac] musicgen-server listening on http://%s:%d\n", host.c_str(), port);
    if (!server.listen(host.c_str(), port)) {
        fprintf(stderr, "error: could not bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
