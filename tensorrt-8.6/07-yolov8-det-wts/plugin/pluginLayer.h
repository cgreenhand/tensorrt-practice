#pragma once
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <cstdint>
#include <string>
#include <vector>

namespace yolov8
{

class Yolov8DecodePlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    Yolov8DecodePlugin(int32_t numClasses, int32_t inputWidth, int32_t inputHeight, int32_t maxOutputBoxes, float socreThreshold, std::vector<int32_t> strides);

    Yolov8DecodePlugin(void const* serialData, size_t serialLength);

    int32_t getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs
    ) const noexcept override;

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs
    ) noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    
    nvinfer1::DataType getOutputDataType(
        int32_t index,
        nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;


    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs
    ) noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext,
        cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator
    ) noexcept override;

    void detachFromContext() noexcept override;

private:
    int32_t mNumClasses{};
    int32_t mInputWidth{};
    int32_t mInputHeight{};
    int32_t mMaxOutputBoxes{};
    float mScoreThreshold{};
    std::vector<int32_t> mStrides;
    std::string mNamespace;

};



class Yolov8DecodePluginCreator final : public nvinfer1::IPluginCreator
{
public:
    Yolov8DecodePluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fieldCollection) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serialData,
        size_t serialLength
    ) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
    std::string mNamespace;


};

}

