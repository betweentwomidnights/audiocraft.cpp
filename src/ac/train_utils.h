#pragma once
#include "ac/lora.h"
#include <vector>
namespace ac {
inline std::vector<float> train_read_tensor(ggml_tensor *t) {
    if (t->type != GGML_TYPE_F32)
        throw gguf_error("training cache must be F32");
    std::vector<float> v((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    for (float x : v)
        if (!std::isfinite(x))
            throw gguf_error("nonfinite training tensor");
    return v;
}

struct MfTrainParam {
    std::string name;
    ggml_tensor *tensor = nullptr;
    ggml_tensor *grad = nullptr;
    std::vector<float> value, m, v;
};

} // namespace ac
