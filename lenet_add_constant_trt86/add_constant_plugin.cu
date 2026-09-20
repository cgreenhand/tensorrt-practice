#include "add_constant_plugin.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <cmath>

namespace {
using namespace nvinfer1;
constexpr char kType[] = "LeNetAddConstant";
constexpr char kVersion[] = "1";
constexpr char kNamespace[] = "lenet_demo";
constexpr uint32_t kMagic = 0x41444331U;

__global__ void addConstantKernel(float const* x, float* y, size_t n, float value) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x)
        y[i] = x[i] + value;
}

class AddConstantPlugin final : public IPluginV2DynamicExt {
public:
    explicit AddConstantPlugin(float value) noexcept : mValue(value) {}
    char const* getPluginType() const noexcept override { return kType; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    IPluginV2DynamicExt* clone() const noexcept override {
        return new (std::nothrow) AddConstantPlugin(*this);
    }
    void setPluginNamespace(char const* ns) noexcept override {
        std::strncpy(mNamespace, ns ? ns : "", sizeof(mNamespace) - 1);
        mNamespace[sizeof(mNamespace) - 1] = '\0';
    }
    char const* getPluginNamespace() const noexcept override { return mNamespace; }
    size_t getSerializationSize() const noexcept override { return sizeof(kMagic) + sizeof(mValue); }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &kMagic, sizeof(kMagic));
        std::memcpy(static_cast<char*>(buffer) + sizeof(kMagic), &mValue, sizeof(mValue));
    }
    DataType getOutputDataType(int32_t, DataType const* types,
                              int32_t) const noexcept override { return types[0]; }
    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs,
                                 int32_t, IExprBuilder&) noexcept override {
        return inputs[0]; // Preserve shape, including dynamic batch.
    }
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* io,
                                   int32_t ni, int32_t no) noexcept override {
        return ni == 1 && no == 1 && pos >= 0 && pos < 2
            && io[pos].type == DataType::kFLOAT
            && io[pos].format == TensorFormat::kLINEAR;
    }
    void configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                         DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t,
                            PluginTensorDesc const*, int32_t) const noexcept override { return 0; }
    int32_t enqueue(PluginTensorDesc const* desc, PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs,
                    void*, cudaStream_t stream) noexcept override {
        if (desc[0].type != DataType::kFLOAT || desc[0].dims.nbDims < 0) return 1;
        size_t n = 1;
        for (int d = 0; d < desc[0].dims.nbDims; ++d) {
            int const extent = desc[0].dims.d[d];
            if (extent < 0) return 1;
            if (extent == 0) return 0;
            if (n > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)) return 1;
            n *= static_cast<size_t>(extent);
        }
        if (!inputs[0] || !outputs[0]) return 1;
        unsigned int const blocks = static_cast<unsigned int>(
            std::min<size_t>((n - 1) / 256 + 1, 4096));
        addConstantKernel<<<blocks, 256, 0, stream>>>(
            static_cast<float const*>(inputs[0]), static_cast<float*>(outputs[0]), n, mValue);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }
private:
    float mValue;
    char mNamespace[128]{};
};

class AddConstantCreator final : public IPluginCreator {
public:
    AddConstantCreator() noexcept
        : mValueField("value", nullptr, PluginFieldType::kFLOAT32, 1), mFields{1, &mValueField} {}
    char const* getPluginName() const noexcept override { return kType; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    IPluginV2* createPlugin(char const*, PluginFieldCollection const* fc) noexcept override {
        // A single, required float32 field. Copy the value; never retain its pointer.
        if (!fc || fc->nbFields != 1 || !fc->fields) return nullptr;
        auto const& field = fc->fields[0];
        if (!field.name || std::strcmp(field.name, "value") != 0
            || field.type != PluginFieldType::kFLOAT32 || field.length != 1 || !field.data)
            return nullptr;
        float value{};
        std::memcpy(&value, field.data, sizeof(value));
        if (!std::isfinite(value)) return nullptr;
        auto* p = new (std::nothrow) AddConstantPlugin(value);
        if (p) p->setPluginNamespace(mNamespace);
        return p;
    }
    IPluginV2* deserializePlugin(char const*, void const* data, size_t size) noexcept override {
        if (!data || size != sizeof(kMagic) + sizeof(float)) return nullptr;
        uint32_t magic{};
        std::memcpy(&magic, data, sizeof(magic));
        if (magic != kMagic) return nullptr;
        float value{};
        std::memcpy(&value, static_cast<char const*>(data) + sizeof(kMagic), sizeof(value));
        if (!std::isfinite(value)) return nullptr;
        auto* p = new (std::nothrow) AddConstantPlugin(value);
        if (p) p->setPluginNamespace(mNamespace);
        return p;
    }
    void setPluginNamespace(char const* ns) noexcept override {
        std::strncpy(mNamespace, ns ? ns : "", sizeof(mNamespace) - 1);
        mNamespace[sizeof(mNamespace) - 1] = '\0';
    }
    char const* getPluginNamespace() const noexcept override { return mNamespace; }
private:
    PluginField mValueField;
    PluginFieldCollection mFields;
    char mNamespace[128]{};
};
} // namespace

extern "C" ADD_CONSTANT_API bool registerAddConstantPlugin() noexcept {
    // Explicit call ensures the shared library is loaded even with --as-needed.
    // Creator outlives all engines; keep this library loaded while using them.
    static AddConstantCreator creator;
    static bool const registered = [] {
        creator.setPluginNamespace(kNamespace);
        auto* registry = ::getPluginRegistry();
        return registry && registry->registerCreator(creator, kNamespace);
    }();
    return registered;
}

