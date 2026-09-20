#pragma once
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#if NV_TENSORRT_MAJOR != 8 || NV_TENSORRT_MINOR != 6
#error "This example targets TensorRT 8.6.x."
#endif

inline void require(bool ok, std::string const& message) {
    if (!ok) throw std::runtime_error(message);
}
inline float parseConstant(std::string const& text) {
    size_t consumed = 0;
    float value = std::stof(text, &consumed);
    require(consumed == text.size() && std::isfinite(value),
            "Constant must be a finite float32 number: " + text);
    return value;
}
inline void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity s, char const* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << '\n';
    }
};
// TensorRT 8.6 supports delete; destroy() is deprecated.
template<class T> using TrtPtr = std::unique_ptr<T>;
inline size_t volume(nvinfer1::Dims const& dims) {
    require(dims.nbDims >= 0, "Invalid dimensions");
    size_t n = 1;
    for (int d = 0; d < dims.nbDims; ++d) {
        require(dims.d[d] > 0, "Unresolved or empty dimensions");
        require(n <= std::numeric_limits<size_t>::max() / static_cast<size_t>(dims.d[d]),
                "Dimension overflow");
        n *= static_cast<size_t>(dims.d[d]);
    }
    require(n <= std::numeric_limits<size_t>::max() / sizeof(float), "Byte size overflow");
    return n;
}
inline std::vector<char> readFile(std::string const& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(f), "Cannot open " + path);
    auto const size = f.tellg();
    require(size > 0, "Empty or unreadable file: " + path);
    std::vector<char> data(static_cast<size_t>(size));
    f.seekg(0);
    require(static_cast<bool>(f.read(data.data(), static_cast<std::streamsize>(data.size()))),
            "Cannot read " + path);
    return data;
}
inline void writeFile(std::string const& path, void const* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    require(static_cast<bool>(f), "Cannot create " + path);
    f.write(static_cast<char const*>(data), static_cast<std::streamsize>(size));
    f.close();
    require(static_cast<bool>(f), "Cannot write " + path);
}
inline std::vector<float> readFloats(std::string const& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(f), "Cannot open " + path);
    require(f.tellg() == static_cast<std::streamoff>(count * sizeof(float)),
            "Float32 file size mismatch: " + path);
    f.seekg(0);
    std::vector<float> v(count);
    require(static_cast<bool>(f.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(count * sizeof(float)))), "Cannot read " + path);
    return v;
}
struct DeviceBuffer {
    void* ptr{};
    explicit DeviceBuffer(size_t bytes) { cudaCheck(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};
struct Stream {
    cudaStream_t value{};
    Stream() { cudaCheck(cudaStreamCreate(&value)); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
};
