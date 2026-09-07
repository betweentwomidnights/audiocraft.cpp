// lstm_test -- pin ac::lstm_graph to torch's nn.LSTM, on every backend this build hosts.
//
// The bottleneck LSTM is the one piece of the codec ggml has no operation for, so it is
// unrolled by hand and nothing upstream covers it. The fixtures below come straight from
// torch (regenerate with gen_lstm_fixture.py, documented in docs/MELODYFLOW_VAE.md):
// a 2-layer nn.LSTM plus StreamableLSTM's residual skip, which is exactly what
// audiocraft/modules/lstm.py computes.
//
// Small on purpose -- gate order (i, f, g, o), layer chaining, zero-initialised state, the
// per-timestep write into the output sequence and the residual are all structural, and a
// 3-wide, 5-step case falsifies each of them independently of any checkpoint.
#include "ac/lstm.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

// A GgufModel holding hand-made tensors. ac::lstm_graph only ever calls W.get(), so this
// is enough to drive the real graph builder without a checkpoint on disk.
struct FakeModel {
    ac::GgufModel model;

    FakeModel(ggml_backend_t backend, const std::vector<std::pair<std::string,
                                                                  std::vector<int64_t>>>& shapes) {
        ggml_init_params ip = {ggml_tensor_overhead() * (shapes.size() + 8), nullptr, true};
        model.ctx = ggml_init(ip);
        model.backend = backend;
        model.owns_backend = false;   // ac_test_backends owns them
        for (const auto& s : shapes) {
            ggml_tensor* t = s.second.size() == 1
                ? ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, s.second[0])
                : ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, s.second[0], s.second[1]);
            ggml_set_name(t, s.first.c_str());
            model.tensors[s.first] = t;
        }
        model.buf = ggml_backend_alloc_ctx_tensors(model.ctx, backend);
        if (!model.buf) {
            std::fprintf(stderr, "FAIL: could not allocate fixture tensors\n");
            std::abort();
        }
    }

    void set(const std::string& name, const float* data, size_t n) {
        ggml_backend_tensor_set(model.get(name), data, 0, n * sizeof(float));
    }
};

} // namespace

int main() {
    constexpr int H = 3, T = 5, L = 2;

    static const float w_ih0[] = {
        0.0403250456f, -0.347792119f, 0.183841825f, 0.181161284f, -0.308580369f,
        -0.0865316391f, -0.338227928f, 0.149806619f, -0.155519515f, 0.405609727f,
        0.409853041f, 0.0588148832f, -0.246137112f, -0.339113146f, -0.0634043813f,
        -0.162482589f, 0.254473388f, -0.492997169f, 0.542616487f, -0.452889085f,
        0.442108989f, -0.100259513f, 0.296982586f, 0.224990547f, 0.0241679549f,
        0.107654095f, 0.438444614f, 0.148481905f, 0.306311667f, -0.446614385f, 0.411007464f,
        0.19870615f, 0.146243453f, 0.0797582269f, 0.281438351f, 0.530257583f
    };
    static const float w_hh0[] = {
        -0.128469706f, -0.321645916f, -0.145258933f, -0.351885587f, 0.277733207f,
        -0.285349697f, -0.308131278f, 0.498153687f, 0.528319478f, 0.0664034486f,
        -0.0999755263f, -0.0745271444f, 0.273505151f, -0.539117575f, -0.471836269f,
        0.461192012f, 0.569983363f, -0.0343090296f, -0.456194788f, 0.0157750249f,
        -0.268594593f, -0.0011099577f, 0.282592058f, 0.255578339f, -0.067702949f,
        0.0635638833f, 0.157117963f, -0.452540696f, -0.195744455f, 0.0226052999f,
        -0.329441369f, -0.252148658f, 0.193919063f, 0.332281411f, 0.0080896616f,
        -0.224580407f
    };
    static const float bias0[] = {
        0.07854864f, -0.4593817f, 0.222297162f, 0.861443698f, 0.138434768f, 0.543661177f,
        -0.213005215f, -0.48506248f, 0.754825473f, -0.501928926f, -0.684948742f, -0.4994241f
    };
    static const float w_ih1[] = {
        -0.351472855f, 0.445836306f, 0.261461973f, 0.446643353f, -0.42983228f,
        -0.562896669f, -0.378609568f, 0.06176579f, -0.114030868f, -0.363831013f,
        0.420296133f, -0.186187118f, -0.33392638f, 0.26668942f, -0.155559987f, 0.368068635f,
        0.137595773f, 0.0142437816f, 0.121201813f, -0.433831394f, -0.308914512f,
        -0.271350712f, -0.326690078f, 0.10237807f, 0.115344703f, -0.239885837f,
        0.376370192f, -0.321235925f, 0.116561711f, -0.16486004f, -0.500809491f,
        -0.375940382f, 0.235835016f, 0.228472829f, -0.357006758f, -0.180036932f
    };
    static const float w_hh1[] = {
        -0.53973341f, 0.310350597f, 0.499229074f, 0.188807607f, -0.357234776f, 0.304529011f,
        0.479134679f, -0.118117303f, -0.0946420431f, 0.199831665f, 0.074275136f,
        -0.190820932f, -0.535992265f, -0.40483737f, 0.0312660336f, -0.0309710503f,
        0.476755261f, 0.0197592974f, -0.148658305f, -0.564125478f, -0.530832171f,
        0.239831805f, 0.51424706f, -0.00876003504f, -0.51603651f, 0.204982162f,
        0.470238328f, -0.382121176f, -0.152007192f, 0.0485572815f, -0.329258442f,
        -0.535343766f, 0.252751112f, -0.473761797f, -0.144919276f, -0.306563288f
    };
    static const float bias1[] = {
        0.922208786f, 0.28775537f, 0.166716099f, -0.210611939f, 0.380176067f,
        -0.0827434957f, -0.699003935f, 0.298642576f, -0.0622166693f, 0.171037197f,
        0.663857639f, 0.317254931f
    };
    static const float input_seq[] = {
        0.227393121f, -2.19340587f, -0.312876463f, 0.386904866f, 0.383115232f,
        -0.715553641f, -1.07646322f, -2.10979986f, -0.800749063f, -0.00947141647f,
        0.870284081f, -0.879723012f, -1.04588366f, -1.10580182f, -1.47461486f
    };
    static const float expected[] = {
        0.0271841139f, -2.07793951f, -0.315336555f, 0.0921051204f, 0.576808929f,
        -0.671393931f, -1.43115973f, -1.86733067f, -0.721914232f, -0.405012786f,
        1.17006516f, -0.760722637f, -1.46936488f, -0.77930212f, -1.33582997f
    };

    // torch stores weight_ih_l* as [4*H, H]; a GGUF tensor written from that numpy array
    // has ne = [H, 4*H], which is what ggml_mul_mat wants as its first argument.
    const std::vector<std::pair<std::string, std::vector<int64_t>>> shapes = {
        {"lstm.0.w_ih", {H, 4 * H}}, {"lstm.0.w_hh", {H, 4 * H}}, {"lstm.0.bias", {4 * H}},
        {"lstm.1.w_ih", {H, 4 * H}}, {"lstm.1.w_hh", {H, 4 * H}}, {"lstm.1.bias", {4 * H}},
    };

    for (const AcTestBackend& backend : ac_test_backends()) {
        FakeModel fixture(backend.backend, shapes);
        fixture.set("lstm.0.w_ih", w_ih0, 4 * H * H);
        fixture.set("lstm.0.w_hh", w_hh0, 4 * H * H);
        fixture.set("lstm.0.bias", bias0, 4 * H);
        fixture.set("lstm.1.w_ih", w_ih1, 4 * H * H);
        fixture.set("lstm.1.w_hh", w_hh1, 4 * H * H);
        fixture.set("lstm.1.bias", bias1, 4 * H);

        const size_t nodes = ac::lstm_graph_nodes(L, T) + 64;
        ggml_init_params ip = {ggml_tensor_overhead() * (nodes + 64)
                                   + ggml_graph_overhead_custom(nodes, false),
                               nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_set_input(x);
        ggml_tensor* y = ggml_cont(ctx, ac::lstm_graph(ctx, fixture.model, "lstm", x, L));
        ggml_set_output(y);
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(graph, y);

        ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
            std::fprintf(stderr, "FAIL [%s]: could not allocate the LSTM graph\n", backend.name);
            ++failures;
            ggml_free(ctx);
            continue;
        }
        ggml_backend_tensor_set(x, input_seq, 0, sizeof(input_seq));
        std::string error;
        if (!ac::graph_compute_checked(backend.backend, graph, "LSTM", error)) {
            std::fprintf(stderr, "FAIL [%s]: %s\n", backend.name, error.c_str());
            ++failures;
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            continue;
        }

        std::vector<float> got((size_t)T * H);
        ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));

        double worst = 0.0;
        for (int i = 0; i < T * H; ++i) {
            const double err = std::fabs((double)got[i] - (double)expected[i]);
            if (err > worst) worst = err;
            if (!(err < 1e-5)) {
                std::fprintf(stderr, "FAIL [%s] element %d: got %.9g, want %.9g (err %.3e)\n",
                             backend.name, i, got[i], expected[i], err);
                ++failures;
            }
        }
        std::printf("  %-12s max abs err %.3e\n", backend.name, worst);

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    // layers == 0 must be a pass-through, which is the codec configuration with no
    // bottleneck LSTM at all.
    {
        ggml_init_params ip = {ggml_tensor_overhead() * 16, nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        ac::GgufModel empty;
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        if (ac::lstm_graph(ctx, empty, "lstm", x, 0) != x) {
            std::fprintf(stderr, "FAIL: a zero-layer LSTM did not pass its input through\n");
            ++failures;
        }
        ggml_free(ctx);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("lstm_test: ok\n");
    return 0;
}
