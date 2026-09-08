// terry-server -- MelodyFlow's `edit` over HTTP, on :8002.
//
// A drop-in for gary4local's `localhost_melodyflow.py`, so `service_manager.rs` only changes
// `entryPoint` from `python.exe localhost_melodyflow.py` to this binary and skips building a
// venv. The routes and reply shapes are the service's, because gary4juce is already reading
// them:
//
//   GET  /health
//   GET  /variations                     the 34 presets, name -> {prompt, flowstep}
//   POST /transform                      JSON in, WAV bytes out (the control centre's path)
//   POST /api/juce/transform_audio       -> {success, session_id, seed}, runs in background
//   GET  /api/juce/poll_status/<id>      -> progress, then {audio_data} on completion
//   POST /api/juce/undo_transform        -> the audio that was sent in
//
// Two deliberate differences from the Python, both noted in docs/SERVICES.md: `/transform`
// does not accept multipart uploads (nothing in gary4local sends them), and `/variations`
// actually works -- the Python reads `VARIATIONS[name]['flowstep']` where the table defines
// `default_flowstep`, so that endpoint raises today.
#include "ac/conditioner.h"
#include "gguf_model.h"
#include "mf/pipeline.h"
#include "mf/vae.h"
#include "mf/variations.h"
#include "rng.h"
#include "serve/http.h"

#include "httplib.h"
#include "yyjson.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace ac::serve;

std::string g_models_dir;
std::string g_device;
std::string g_dit, g_codec, g_t5;
Sessions g_sessions;

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

// The first matching gguf in the models directory. Precision is a deployment choice, so
// prefer F16 (the DiT is 1.9 GB there against 3.9) unless only F32 is present.
std::string find_model(const std::string& stem) {
    for (const char* suffix : {"-F16.gguf", "-F32.gguf"}) {
        const std::string path = g_models_dir + "/" + stem + suffix;
        if (file_exists(path)) return path;
    }
    return {};
}

// --- request parsing ------------------------------------------------------------------------

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
    bool has(const char* key) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        return v && !yyjson_is_null(v);
    }
    double num(const char* key, double fallback) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (!v) return fallback;
        if (yyjson_is_num(v)) return yyjson_get_num(v);
        // The plugin sometimes sends numbers as strings; the Python side coerces too.
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

// --- the transform ---------------------------------------------------------------------------

struct TransformRequest {
    std::string audio_b64;
    std::string prompt;
    float flowstep = 0.12f;
    std::string solver = "euler";
    uint64_t seed = 1234;
};

// terry's `process_audio`, in the same order and with the same staging as mf-edit: the text
// encoder, the codec, the DiT and the codec again, one at a time. A resident model would
// save about three seconds a request and cost the headroom that keeps a 30 s window inside
// 8 GB, which is the wrong trade for the machines this ships to.
std::string run_transform(const TransformRequest& req, const std::string& session) {
    ac::EditParams params;
    params.solver = ac::parse_flow_solver(req.solver);
    params.steps = params.solver == ac::FlowSolverKind::Midpoint ? 64 : 25;
    params.regularize = params.solver == ac::FlowSolverKind::Euler;
    params.target_flowstep = req.flowstep;

    const ac::TextCondition cond = ac::encode_prompt(g_t5, req.prompt, g_device);

    std::vector<float> encoded;
    int frames = 0, latent_dim = 0, channels = 0, sample_rate = 0;
    {
        ac::GgufModel codec = ac::load_gguf(
            g_codec.c_str(), ac::make_backend(0, g_device.empty() ? nullptr : g_device.c_str()));
        const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
        latent_dim = c.latent_dim;
        channels = c.channels;
        sample_rate = c.sample_rate;

        int n_samples = 0, n_ch = 0, rate = 0;
        std::vector<float> audio = decode_audio(req.audio_b64, n_samples, n_ch, rate);
        // The window is fixed: max_duration at the codec's frame rate, which
        // `_generate_tokens` asserts against on the Python side.
        ac::conform_audio(audio, n_samples, n_ch, rate, c, (int)(c.frame_rate * 30));
        frames = n_samples / c.hop_length;
        encoded = ac::codec_encode(codec, c, audio, n_samples, n_ch);
    }

    std::vector<float> edited;
    {
        ac::GgufModel dit = ac::load_gguf(
            g_dit.c_str(), ac::make_backend(0, g_device.empty() ? nullptr : g_device.c_str()));
        const ac::DitConfig dc = ac::DitConfig::from(dit);
        params.cfg_coef = dc.cfg_coef;
        const ac::LatentStats stats = ac::mf_latent_stats(dit, latent_dim);

        // Every draw comes from one seed, the way `torch.manual_seed` before `edit()` makes
        // the whole transform reproducible on the Python side. The streams are not the same
        // numbers -- see docs/MELODYFLOW_EDIT.md -- so a seed does not carry across.
        ac::Rng rng(req.seed);
        auto noise = [&rng](float* dst, size_t n) { rng.fill_normal(dst, n); };

        std::vector<float> noise0((size_t)frames * latent_dim);
        noise(noise0.data(), noise0.size());
        std::vector<float> latent = ac::vae_sample(encoded, frames, latent_dim, noise0);
        ac::latent_normalize(latent, frames, latent_dim, stats.mean, stats.std);

        ac::DitRunner runner(dit, dc, cond, frames);
        ac::EditReport report;
        auto progress = [&session](int done, int total) {
            g_sessions.set_progress(session, done, total);
        };
        edited = ac::mf_edit_latent(runner, latent, frames, latent_dim, params, noise,
                                    progress, report);
        ac::latent_denormalize(edited, frames, latent_dim, stats.mean, stats.std);
    }

    ac::GgufModel codec = ac::load_gguf(
        g_codec.c_str(), ac::make_backend(0, g_device.empty() ? nullptr : g_device.c_str()));
    const ac::SeanetConfig c = ac::SeanetConfig::from(codec);
    int64_t out_samples = 0;
    const std::vector<float> audio = ac::codec_decode(codec, c, edited, frames, out_samples);
    return encode_audio(audio, (int)out_samples, channels, sample_rate);
}

// Resolve a request against the preset table. A custom prompt replaces the preset's; a
// custom flowstep replaces its default. Both are what `process_audio` does.
bool build_request(const Json& body, TransformRequest& out, std::string& error) {
    const std::string variation = body.str("variation");
    if (variation.empty()) { error = "variation is required"; return false; }
    const ac::Variation* preset = ac::find_variation(variation.c_str());
    if (!preset) { error = "Unknown variation: " + variation; return false; }

    out.prompt = body.str("custom_prompt", body.str("prompt", preset->prompt));
    if (out.prompt.empty()) out.prompt = preset->prompt;
    out.flowstep = (float)body.num("flowstep", preset->flowstep);
    if (!(out.flowstep > 0.0f)) { error = "Flowstep must be positive"; return false; }
    out.solver = body.str("solver", "euler");
    for (char& ch : out.solver) ch = (char)tolower((unsigned char)ch);
    if (out.solver != "euler" && out.solver != "midpoint") {
        error = "Invalid solver. Must be \"euler\" or \"midpoint\"";
        return false;
    }
    out.seed = resolve_seed(body.seed());
    return true;
}

void usage() {
    fprintf(stderr,
        "usage: terry-server [--port 8002] [--models-dir DIR] [--device NAME]\n"
        "\n"
        "Serves MelodyFlow's edit on the same routes as gary4local's melodyflow service.\n"
        "Models are found in --models-dir (or AC_MODELS_DIR): melodyflow-dit-*, \n"
        "melodyflow-vae-*, t5-base-encoder-*.\n");
}

} // namespace

int main(int argc, char** argv) {
    int port = 8002;
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

    g_dit = find_model("melodyflow-dit-t24-1.0B-v1.0");
    g_codec = find_model("melodyflow-vae-48khz-v1.0");
    g_t5 = find_model("t5-base-encoder-0.1B-v1.0");
    // Missing models are reported by /health rather than refused at startup: the Tauri side
    // starts services before the user has necessarily fetched anything, and a process that
    // exits immediately looks like a crash.
    for (const auto& m : {std::make_pair("dit", &g_dit), std::make_pair("codec", &g_codec),
                          std::make_pair("t5", &g_t5)}) {
        if (m.second->empty())
            fprintf(stderr, "[ac] warning: no %s gguf in %s\n", m.first, g_models_dir.c_str());
        else
            fprintf(stderr, "[ac] %s: %s\n", m.first, m.second->c_str());
    }

    httplib::Server server;
    server.set_payload_max_length(256ull * 1024 * 1024);   // a 30 s stereo wav in base64

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        const bool ready = !g_dit.empty() && !g_codec.empty() && !g_t5.empty();
        JsonObject o;
        o.str("status", ready ? "healthy" : "degraded")
         .str("service", "melodyflow-localhost")
         .boolean("model_loaded", ready)
         .str("models_dir", g_models_dir)
         .str("device", g_device.empty() ? "auto" : g_device)
         .str("backend", "audiocraft.cpp");
        res.status = ready ? 200 : 503;
        res.set_content(o.str(), "application/json");
    });

    server.Get("/variations", [](const httplib::Request&, httplib::Response& res) {
        size_t count = 0;
        const ac::Variation* table = ac::variations(count);
        std::string body = "{\"variations\":{";
        for (size_t i = 0; i < count; ++i) {
            if (i) body += ',';
            body += json_string(table[i].name) + ":" +
                    JsonObject().str("prompt", table[i].prompt)
                                .num("flowstep", table[i].flowstep).str();
        }
        body += "}}";
        res.set_content(body, "application/json");
    });

    // The control centre's route: JSON in, the WAV itself out, the way `send_file` replies.
    server.Post("/transform", [](const httplib::Request& req, httplib::Response& res) {
        try {
            Json body(req.body);
            TransformRequest tr;
            std::string error;
            if (!build_request(body, tr, error)) {
                res.status = 400;
                res.set_content(json_error(error), "application/json");
                return;
            }
            tr.audio_b64 = body.str("audio_data");
            if (tr.audio_b64.empty()) {
                res.status = 400;
                res.set_content(json_error("audio_data is required"), "application/json");
                return;
            }
            const std::string session = g_sessions.create({});
            std::lock_guard<std::mutex> lock(g_sessions.gpu());
            res.set_content(b64_decode(run_transform(tr, session)), "audio/wav");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    server.Post("/api/juce/transform_audio",
                [](const httplib::Request& req, httplib::Response& res) {
        try {
            Json body(req.body);
            const std::string audio = body.str("audio_data");
            if (audio.empty()) {
                res.status = 400;
                res.set_content(json_error("audio_data is required"), "application/json");
                return;
            }
            TransformRequest tr;
            std::string error;
            if (!build_request(body, tr, error)) {
                res.status = 400;
                res.set_content(json_error(error), "application/json");
                return;
            }
            tr.audio_b64 = audio;

            Session initial;
            initial.status = "queued";
            initial.seed = tr.seed;
            initial.original_b64 = audio;      // undo_transform hands this straight back
            const std::string id = g_sessions.create(initial);

            run_job(g_sessions, id, [tr](const std::string& session) {
                const std::string out = run_transform(tr, session);
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
                                .num("seed", (double)tr.seed)
                                .str("message", "Audio transform started")
                                .str("note", "Poll /api/juce/poll_status/{session_id} for "
                                             "progress and results")
                                .str(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    server.Get(R"(/api/juce/poll_status/(\w+))",
               [](const httplib::Request& req, httplib::Response& res) {
        Session s;
        if (!g_sessions.get(req.matches[1], s)) {
            res.status = 404;
            res.set_content(json_error("Session not found"), "application/json");
            return;
        }
        const bool running = s.status == "warming" || s.status == "processing" ||
                             s.status == "queued";
        JsonObject o;
        o.boolean("success", true)
         .str("status", s.status)
         .num("progress", s.progress)
         .raw("queue_status", "{}")
         .boolean("transform_in_progress", running)
         .num("seed", (double)s.seed);
        if (s.status == "completed" && !s.audio_b64.empty()) o.str("audio_data", s.audio_b64);
        if (s.status == "failed") o.str("error", s.error.empty() ? "Unknown error" : s.error);
        res.set_content(o.str(), "application/json");
    });

    server.Post("/api/juce/undo_transform",
                [](const httplib::Request& req, httplib::Response& res) {
        try {
            Json body(req.body);
            const std::string id = body.str("session_id");
            if (id.empty()) {
                res.status = 400;
                res.set_content(json_error("session_id is required"), "application/json");
                return;
            }
            Session s;
            if (!g_sessions.get(id, s) || s.original_b64.empty()) {
                res.status = 404;
                res.set_content(json_error("No original audio found for undo"),
                                "application/json");
                return;
            }
            res.set_content(JsonObject()
                                .boolean("success", true)
                                .str("audio_data", s.original_b64)
                                .str("message", "Transform undone successfully")
                                .str(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    fprintf(stderr, "[ac] terry-server listening on http://%s:%d\n", host.c_str(), port);
    if (!server.listen(host.c_str(), port)) {
        fprintf(stderr, "error: could not bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
