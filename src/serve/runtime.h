#pragma once
#include "gguf_model.h"
#include "yyjson.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ac::serve {
// Supervisors must be able to inspect an unpacked runtime without model files
// or a listening port. Backend enumeration also proves that staged DLLs load.
inline int runtime_probe(int argc, char** argv, const char* service) {
    if (argc != 2) return -1;
    if (std::strcmp(argv[1], "--version") == 0) {
        std::printf("%s\n", AC_VERSION);
        return 0;
    }
    if (std::strcmp(argv[1], "--props") != 0) return -1;
    ac::load_dynamic_backends_once();
    auto doc = yyjson_mut_doc_new(nullptr);
    auto root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "success", true);
    yyjson_mut_obj_add_str(doc, root, "service", service);
    yyjson_mut_obj_add_str(doc, root, "runtime", "audiocraft");
    yyjson_mut_obj_add_str(doc, root, "version", AC_VERSION);
    auto devices = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "devices", devices);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        auto entry = yyjson_mut_obj(doc);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        const auto type = ggml_backend_dev_type(device);
        yyjson_mut_obj_add_str(doc, entry, "backend",
                              ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
        yyjson_mut_obj_add_str(doc, entry, "description", ggml_backend_dev_description(device));
        yyjson_mut_obj_add_str(doc, entry, "type", type == GGML_BACKEND_DEVICE_TYPE_CPU ? "cpu" :
                              type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "igpu" :
                              type == GGML_BACKEND_DEVICE_TYPE_GPU ? "gpu" : "accel");
        yyjson_mut_obj_add_uint(doc, entry, "memory_total_bytes", total);
        yyjson_mut_arr_add_val(devices, entry);
    }
    char* json = yyjson_mut_write(doc, 0, nullptr);
    if (!json) { yyjson_mut_doc_free(doc); return 1; }
    std::puts(json);
    std::free(json);
    yyjson_mut_doc_free(doc);
    return 0;
}

inline int runtime_port(const char* service_key, int fallback) {
    const char* value = std::getenv(service_key);
    if (!value || !*value) value = std::getenv("AC_PORT");
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long port = std::strtol(value, &end, 10);
    if (!end || *end || port < 1 || port > 65535) {
        std::fprintf(stderr, "error: invalid port environment value: %s\n", value);
        std::exit(2);
    }
    return (int)port;
}
} // namespace ac::serve
