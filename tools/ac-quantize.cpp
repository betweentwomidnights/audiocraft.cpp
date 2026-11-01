// ac-quantize -- compress a MusicGen LM gguf to a k-quant or Q8_0 mix.
//
// A port of sa3.cpp's sa3-quantize for this repository's tensor names. The LM uses its
// weights only through ggml_mul_mat and ggml_get_rows, which every backend runs quantized,
// so nothing downstream changes: musicgen-server reads the result like the F16 file.
//
// musicgen-large is 6.6 GB at F16, which does not fit an 8 GB GPU beside its KV cache. Q8_0
// and Q4_K_M are what make the large finetunes usable there.
//
// Every quantized tensor is dequantized again and compared with its source, and the worst
// cosine is reported; anything under 0.99 fails the run, the gate sa3.cpp's tiers passed.
//
//   ac-quantize --in musicgen-hoenn-lofi-3.3B-v1.0-F16.gguf \
//               --out musicgen-hoenn-lofi-3.3B-v1.0-Q4_K_M.gguf --mix q4_k_m
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace {

// The tensors a k-quant mix keeps at Q6_K: the FFN down projection and the cross-attention
// K/V (the counterparts of llama.cpp's ffn_down and attn_v), the codebook embeddings, which
// reach ggml_get_rows, and the output heads, which pick every token.
bool critical(const std::string& name) {
    auto ends_with = [&](const char* s) {
        const size_t n = strlen(s);
        return name.size() >= n && name.compare(name.size() - n, n, s) == 0;
    };
    return ends_with(".ff.2.weight") || ends_with(".cross.kv.weight") ||
           name.rfind("lm.emb.", 0) == 0 || name.rfind("lm.head.", 0) == 0;
}

struct QuantMix {
    const char* name;  // --mix value; upper-cased, the -<NAME>.gguf suffix
    ggml_type body;
    ggml_type critical;
};

const QuantMix MIXES[] = {
    {"q4_k_m", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K},
    {"q5_k_m", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K},
    {"q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
};

const QuantMix* find_mix(const char* name) {
    for (const QuantMix& m : MIXES)
        if (!strcmp(m.name, name)) return &m;
    return nullptr;
}

ggml_type target_type(const QuantMix& mix, const std::string& name) {
    return critical(name) ? mix.critical : mix.body;
}

// 2D weights whose rows are whole blocks of the target type. Norms and biases are 1D and
// stay as they are; K-quant blocks are 256 wide, so a row length that is only a multiple of
// 32 must not be handed to ggml_quantize_chunk.
bool should_quantize(const QuantMix& mix, const std::string& name, int n_dims, const int64_t* ne) {
    if (n_dims != 2) return false;
    if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) return false;
    return ne[0] % ggml_blck_size(target_type(mix, name)) == 0;
}

void tensor_meta_init(ggml_tensor* t, ggml_type type, int n_dims, const int64_t* ne, const char* name) {
    memset(t, 0, sizeof(*t));
    t->type = type;
    for (int d = 0; d < GGML_MAX_DIMS; d++) t->ne[d] = d < n_dims ? ne[d] : 1;
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(type));
    for (int d = 2; d < GGML_MAX_DIMS; d++) t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
    strncpy(t->name, name, sizeof(t->name) - 1);
}

double cosine(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) {
        ab += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 1.0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* in_path = nullptr;
    const char* out_path = nullptr;
    const char* mix_name = "q4_k_m";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--mix") && i + 1 < argc) mix_name = argv[++i];
    }
    const QuantMix* mixp = find_mix(mix_name);
    if (!in_path || !out_path || !mixp) {
        if (in_path && out_path && !mixp) fprintf(stderr, "error: unknown --mix '%s'\n", mix_name);
        fprintf(stderr,
                "usage: ac-quantize --in <musicgen-lm.gguf> --out <out.gguf> [--mix q4_k_m|q5_k_m|q8_0]\n"
                "  q4_k_m  ff.2/cross.kv/emb/head -> Q6_K, other weights -> Q4_K   (default)\n"
                "  q5_k_m  ff.2/cross.kv/emb/head -> Q6_K, other weights -> Q5_K\n"
                "  q8_0    every weight -> Q8_0\n");
        return 1;
    }
    const QuantMix& mix = *mixp;

    gguf_init_params params = {/*no_alloc =*/true, /*ctx =*/nullptr};
    gguf_context* in_ctx = gguf_init_from_file(in_path, params);
    if (!in_ctx) {
        fprintf(stderr, "error: cannot read %s\n", in_path);
        return 1;
    }
    {
        const int key = gguf_find_key(in_ctx, "general.architecture");
        const char* arch = key >= 0 ? gguf_get_val_str(in_ctx, key) : "";
        if (strcmp(arch, "ac-musicgen-lm") != 0) {
            fprintf(stderr, "error: %s is '%s'; only the MusicGen LM (ac-musicgen-lm) is quantized\n",
                    in_path, arch);
            return 1;
        }
    }
    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    const size_t data_off_in = gguf_get_data_offset(in_ctx);

    // The whole input in memory. Its size via std::filesystem, not ftell: `long` is 32 bits
    // on MSVC and the large LM is 6.6 GB.
    std::error_code ec;
    const size_t fsize = (size_t)std::filesystem::file_size(in_path, ec);
    if (ec) {
        fprintf(stderr, "error: cannot stat %s (%s)\n", in_path, ec.message().c_str());
        return 1;
    }
    std::vector<char> file_data(fsize);
    FILE* f = fopen(in_path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", in_path);
        return 1;
    }
    size_t got = 0;
    while (got < fsize) {
        const size_t want = fsize - got < (64u << 20) ? fsize - got : (64u << 20);
        const size_t n = fread(file_data.data() + got, 1, want, f);
        if (n == 0) break;
        got += n;
    }
    fclose(f);
    if (got != fsize) {
        fprintf(stderr, "error: short read on %s (%zu of %zu bytes)\n", in_path, got, fsize);
        return 1;
    }

    ggml_quantize_init(mix.body);
    if (mix.critical != mix.body) ggml_quantize_init(mix.critical);

    gguf_context* out_ctx = gguf_init_empty();
    gguf_set_kv(out_ctx, in_ctx);
    std::vector<ggml_type> stype(n_tensors), dtype(n_tensors);
    std::vector<int64_t> ne0(n_tensors), ne1(n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(in_ctx, i);
        const int nd = gguf_get_tensor_ndims(in_ctx, i);
        const int64_t* ne = gguf_get_tensor_ne(in_ctx, i);
        stype[i] = gguf_get_tensor_type(in_ctx, i);
        ne0[i] = ne[0];
        ne1[i] = nd >= 2 ? ne[1] : 1;
        const bool q = (stype[i] == GGML_TYPE_F32 || stype[i] == GGML_TYPE_F16) &&
                       should_quantize(mix, name, nd, ne);
        dtype[i] = q ? target_type(mix, name) : stype[i];
        ggml_tensor t;
        tensor_meta_init(&t, dtype[i], nd, ne, name);
        gguf_add_tensor(out_ctx, &t);
    }

    size_t total = 0;
    for (int64_t i = 0; i < n_tensors; i++)
        total = std::max(total, gguf_get_tensor_offset(out_ctx, i) + gguf_get_tensor_size(out_ctx, i));
    std::vector<char> data_buf(total);

    int n_body = 0, n_crit = 0, n_kept = 0;
    double worst = 1.0;
    std::string worst_name;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(in_ctx, i);
        const char* src = file_data.data() + data_off_in + gguf_get_tensor_offset(in_ctx, i);
        char* dst = data_buf.data() + gguf_get_tensor_offset(out_ctx, i);
        if (dtype[i] == stype[i]) {
            memcpy(dst, src, gguf_get_tensor_size(in_ctx, i));
            n_kept++;
            continue;
        }
        const int64_t n = ne0[i] * ne1[i];
        std::vector<float> f32(n), back(n);
        if (stype[i] == GGML_TYPE_F32) memcpy(f32.data(), src, (size_t)n * sizeof(float));
        else ggml_fp16_to_fp32_row((const ggml_fp16_t*)src, f32.data(), n);
        ggml_quantize_chunk(dtype[i], f32.data(), dst, 0, ne1[i], ne0[i], nullptr);

        const ggml_type_traits* tr = ggml_get_type_traits(dtype[i]);
        const size_t row = ggml_row_size(dtype[i], ne0[i]);
        for (int64_t r = 0; r < ne1[i]; r++) tr->to_float(dst + (size_t)r * row, back.data() + r * ne0[i], ne0[i]);
        const double c = cosine(f32.data(), back.data(), (size_t)n);
        if (c < worst) {
            worst = c;
            worst_name = name;
        }
        (dtype[i] == mix.body ? n_body : n_crit)++;
        fprintf(stderr, "  %-32s %-5s cos %.6f\n", name, ggml_type_name(dtype[i]), c);
    }

    for (int64_t i = 0; i < n_tensors; i++)
        gguf_set_tensor_data(out_ctx, gguf_get_tensor_name(out_ctx, i),
                             data_buf.data() + gguf_get_tensor_offset(out_ctx, i));
    if (worst < 0.99) {
        fprintf(stderr, "error: %s round-trips at cosine %.6f, under the 0.99 gate; nothing written\n",
                worst_name.c_str(), worst);
        return 1;
    }
    if (!gguf_write_to_file(out_ctx, out_path, false)) {
        fprintf(stderr, "error: failed to write %s\n", out_path);
        return 1;
    }
    ggml_quantize_free();
    gguf_free(in_ctx);
    gguf_free(out_ctx);
    fprintf(stderr, "wrote %s: %d %s, %d %s, %d kept, %zu MiB; worst cosine %.6f (%s)\n", out_path,
            n_body, ggml_type_name(mix.body), n_crit, ggml_type_name(mix.critical), n_kept,
            total >> 20, worst, worst_name.c_str());
    return 0;
}
