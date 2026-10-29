#include "mf/dit.h"
#include <cstring>
#include <stdexcept>

int main() {
    auto ctx = ggml_init({1 << 20, nullptr, true});
    if (!ctx) throw std::runtime_error("context allocation failed");
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto half = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 4);
    auto full_product = ggml_mul_mat(ctx, w, x);
    auto half_product = ggml_mul_mat(ctx, half, x);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_add(ctx, full_product, half_product));
    ac::dit_set_matmul_precision(graph);
    enum ggml_prec full_precision, half_precision;
    std::memcpy(&full_precision, full_product->op_params, sizeof(full_precision));
    std::memcpy(&half_precision, half_product->op_params, sizeof(half_precision));
    if (full_precision != GGML_PREC_F32 || half_precision != GGML_PREC_DEFAULT)
        throw std::runtime_error("DiT precision policy changed a model tier incorrectly");
    ggml_free(ctx);
}
