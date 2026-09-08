#include "mf/pipeline.h"

#include "ac/transformer.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ac {

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

} // namespace

LatentStats mf_latent_stats(const GgufModel& dit, int latent_dim) {
    auto read = [&](const char* name, std::vector<float>& dst) {
        ggml_tensor* t = dit.get(name);
        if (ggml_nelements(t) != (int64_t)latent_dim || t->type != GGML_TYPE_F32)
            throw std::runtime_error(std::string(name) + " is not [latent_dim] f32");
        dst.resize((size_t)latent_dim);
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };
    LatentStats out;
    read("dit.latent_mean", out.mean);
    read("dit.latent_std", out.std);
    return out;
}

// --- DitRunner ---------------------------------------------------------------------------

struct DitRunner::Impl {
    const GgufModel& model;
    DitConfig config;
    int frames = 0;

    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* latent_t = nullptr;
    ggml_tensor* cond_t = nullptr;
    ggml_tensor* tfeat_t = nullptr;
    ggml_tensor* rescale_t = nullptr;
    ggml_tensor* pos_t = nullptr;
    ggml_tensor* mask_t = nullptr;
    ggml_tensor* velocity_t = nullptr;

    // The two cross-attention masks, prebuilt: one all-visible, one with every text key at
    // -inf. Both cover the zero-attention key at column 0, which is never masked -- without
    // it the unconditional branch's softmax would be over nothing but -inf.
    std::vector<float> mask_cond, mask_null;
    std::vector<float> cond;
    std::vector<int32_t> positions;

    Impl(const GgufModel& m, const DitConfig& c) : model(m), config(c) {}
    ~Impl() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

DitRunner::DitRunner(const GgufModel& dit, const DitConfig& config, const TextCondition& cond,
                     int frames)
    : impl_(new Impl(dit, config)) {
    Impl& s = *impl_;
    s.frames = frames;
    if (cond.tokens <= 0) throw std::runtime_error("the conditioning has no tokens");
    if (cond.hidden.size() != (size_t)config.cond_dim * cond.tokens)
        throw std::runtime_error("conditioning does not hold cond_dim * tokens values");

    ggml_init_params ip = {(size_t)1024 * 1024 * 1024, nullptr, true};
    s.ctx = ggml_init(ip);
    if (!s.ctx) throw std::runtime_error("failed to create the DiT graph context");

    s.latent_t = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, frames, config.latent_dim);
    s.cond_t = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, config.cond_dim, cond.tokens);
    s.tfeat_t = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, config.time_dim);
    s.rescale_t = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, config.latent_dim, 1);
    s.pos_t = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, frames);
    s.mask_t = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, cond.tokens + 1, frames);
    for (ggml_tensor* t : {s.latent_t, s.cond_t, s.tfeat_t, s.rescale_t, s.pos_t, s.mask_t})
        ggml_set_input(t);

    s.velocity_t = ggml_cont(s.ctx, dit_forward(s.ctx, dit, s.latent_t, s.rescale_t, s.tfeat_t,
                                                s.cond_t, s.pos_t, s.mask_t, config));
    ggml_set_output(s.velocity_t);
    s.graph = ggml_new_graph_custom(s.ctx, dit_graph_nodes(config), false);
    ggml_build_forward_expand(s.graph, s.velocity_t);
    s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dit.backend));
    if (!s.alloc || !ggml_gallocr_alloc_graph(s.alloc, s.graph))
        throw std::runtime_error("failed to allocate the DiT graph");

    // Everything the graph reads is re-uploaded before every forward, including the parts
    // that never change. ggml_gallocr hands out one arena for the whole graph and is free
    // to place a later node's scratch over an input it has finished with, so an input
    // written once at construction survives exactly one execution and is quietly garbage
    // from the second forward on. A solve does 125 of them.
    s.positions.resize((size_t)frames);
    for (int i = 0; i < frames; ++i) s.positions[(size_t)i] = i;
    s.cond = cond.hidden;

    const size_t row = (size_t)cond.tokens + 1;
    s.mask_cond.assign(row * frames, 0.0f);
    s.mask_null.assign(row * frames, -INFINITY);
    for (int q = 0; q < frames; ++q) s.mask_null[(size_t)q * row] = 0.0f;
}

DitRunner::~DitRunner() = default;

void DitRunner::predict(const float* latent, float t, bool conditional, float* velocity) {
    Impl& s = *impl_;
    const size_t n = (size_t)s.frames * s.config.latent_dim;
    const std::vector<float> tfeat = timestep_features(t, s.config.time_dim);
    const std::vector<float> rescale((size_t)s.config.latent_dim, dit_input_rescale(t));
    const std::vector<float>& mask = conditional ? s.mask_cond : s.mask_null;

    ggml_backend_tensor_set(s.latent_t, latent, 0, n * sizeof(float));
    ggml_backend_tensor_set(s.cond_t, s.cond.data(), 0, s.cond.size() * sizeof(float));
    ggml_backend_tensor_set(s.pos_t, s.positions.data(), 0,
                            s.positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(s.tfeat_t, tfeat.data(), 0, tfeat.size() * sizeof(float));
    ggml_backend_tensor_set(s.rescale_t, rescale.data(), 0, rescale.size() * sizeof(float));
    ggml_backend_tensor_set(s.mask_t, mask.data(), 0, mask.size() * sizeof(float));

    std::string error;
    if (!graph_compute_checked(s.model.backend, s.graph, "MelodyFlow DiT", error))
        throw std::runtime_error(error);
    ggml_backend_tensor_get(s.velocity_t, velocity, 0, n * sizeof(float));
    ++forwards_;
}

VelocityFn DitRunner::guided(float cfg_coef) {
    const size_t n = (size_t)impl_->frames * impl_->config.latent_dim;
    // audiocraft batches the two branches into one forward; running them separately is the
    // same arithmetic, since nothing in the model couples samples in a batch.
    return [this, cfg_coef, n](const float* z, float t, float* out) {
        if (cfg_coef == 0.0f) {
            predict(z, t, /*conditional=*/false, out);
            return;
        }
        std::vector<float> uncond(n);
        predict(z, t, /*conditional=*/true, out);
        predict(z, t, /*conditional=*/false, uncond.data());
        for (size_t i = 0; i < n; ++i)
            out[i] = (1.0f + cfg_coef) * out[i] - cfg_coef * uncond[i];
    };
}

// --- edit ----------------------------------------------------------------------------------

std::vector<float> mf_edit_latent(DitRunner& dit, const std::vector<float>& prompt_latent,
                                  int frames, int latent_dim, const EditParams& params,
                                  const NoiseFn& noise, const ProgressFn& progress,
                                  EditReport& report) {
    const FlowParams inversion = edit_inversion_params(params);
    const FlowParams generation = edit_generation_params(params);

    // terry reports progress across both passes against one total, so the generation's
    // callback is offset by everything the inversion did.
    const int inversion_total = inversion.steps *
                                (inversion.regularize ? inversion.regularize_iters : 1);
    const int total = inversion_total + generation.steps;
    ProgressFn inversion_progress, generation_progress;
    if (progress) {
        inversion_progress = [&progress, total](int done, int) { progress(done, total); };
        generation_progress = [&progress, total, inversion_total](int done, int) {
            progress(inversion_total + done, total);
        };
    }

    double t0 = now_s();
    int before = dit.forwards();
    const std::vector<float> intermediate =
        flow_solve(inversion, prompt_latent, latent_dim, frames,
                   dit.guided(inversion.effective_cfg_coef()), noise, inversion_progress);
    report.inversion_seconds = now_s() - t0;
    report.inversion_forwards = dit.forwards() - before;

    t0 = now_s();
    before = dit.forwards();
    std::vector<float> edited =
        flow_solve(generation, intermediate, latent_dim, frames,
                   dit.guided(generation.effective_cfg_coef()), {}, generation_progress);
    report.generation_seconds = now_s() - t0;
    report.generation_forwards = dit.forwards() - before;
    report.intermediate = intermediate;
    return edited;
}

} // namespace ac
