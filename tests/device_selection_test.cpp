#include "gguf_model.h"
#include <stdexcept>
static void require(bool v) { if(!v) throw std::runtime_error("device selection mismatch"); }
int main() {
    require(ac::device_request_matches("", "CUDA", "CUDA0"));
    require(ac::device_request_matches("gpu", "Vulkan", "Vulkan1"));
    require(ac::device_request_matches("CuDa", "CUDA", "CUDA0"));
    require(!ac::device_request_matches("vulkan", "CUDA", "CUDA0"));
    require(!ac::device_request_matches("cuda", "Vulkan", "Vulkan1"));
    require(ac::device_request_matches("VULKAN1", "Vulkan", "Vulkan1"));
    require(!ac::device_request_matches("Vulkan1", "Vulkan", "Vulkan0"));
    require(ac::device_request_matches("metal", "Metal", "MTL0"));
}
