// Ported from sa3.cpp tests/test_backend.h (sa3_test_* -> ac_test_*).
#pragma once

#include "gguf_model.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

inline ggml_backend_t ac_test_cpu_backend() {
    static ggml_backend_t backend = [] {
        ac::load_dynamic_backends_once();
        auto b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        const int requested = ac::cpu_threads_from_env();
        ac::configure_cpu_threads(b, requested > 0 ? requested : 1);
        return b;
    }();
    if (!backend) {
        std::fprintf(stderr, "FAIL: no dynamically registered CPU backend\n");
        std::abort();
    }
    return backend;
}

inline void ac_test_compute(ggml_cgraph* graph) {
    if (ggml_backend_graph_compute(ac_test_cpu_backend(), graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: CPU backend graph compute failed\n");
        std::abort();
    }
}

struct AcTestBackend {
    const char* name;
    ggml_backend_t backend;
};

// BLAS registers an ACCEL device on macOS but implements little beyond matmul. make_backend()
// only ever takes a GPU/IGPU or falls back to CPU, so a model never lands on one.
inline bool ac_test_is_model_hostable(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return true;
        default:                            return false;
    }
}

// Every backend this build can host a model on: build/ gets CPU, build-metal CPU + Metal.
inline const std::vector<AcTestBackend>& ac_test_backends() {
    static std::vector<AcTestBackend> backends = [] {
        ac::load_dynamic_backends_once();
        std::vector<AcTestBackend> out;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const char* selection = std::getenv("AC_TEST_DEVICE");
            if (selection && std::strcmp(selection, "cpu") == 0 &&
                ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            if (!ac_test_is_model_hostable(dev)) continue;
            if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) {
                // Synthetic graphs are tiny: thread dispatch otherwise dwarfs math,
                // especially for thousands of finite-difference evaluations on macOS.
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    const int requested = ac::cpu_threads_from_env();
                    ac::configure_cpu_threads(b, requested > 0 ? requested : 1);
                }
                out.push_back({ ggml_backend_dev_name(dev), b });
            }
        }
        return out;
    }();
    if (backends.empty()) {
        std::fprintf(stderr, "FAIL: no registered backends\n");
        std::abort();
    }
    return backends;
}
