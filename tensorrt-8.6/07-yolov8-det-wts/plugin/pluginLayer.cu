#include "pluginLayer.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <utility>

namespace yolov8
{

namespace
{

constexpr char kPluginName[] = "Yolov8Decode_Plugin";
constexpr char kPluginVersion[] = "1";
constexpr int32_t kDetectionWidth = 6;
constexpr int32_t kMaxStrideCount = 8;


template<typename T>
void writeValue(char*&buffer, T const& value)
{
    std::memcpy(buffer, &value, sizeof(T));
    buffer += sizeof(T);

}

template<typename T>
void readValue(char const*& buffer, T& value)
{

    std::memcpy(&value, buffer, sizeof(T));
    buffer += sizeof(T);

}


struct DecodeParameters
{
    int32_t batchSize;
    int32_t numClasses;
    int32_t numPredictions;
    int32_t inputWidth;
    int32_t inputHeight;
    int32_t maxOutputBoxes;
    int32_t strideCount;
    int32_t strides[kMaxStrideCount];
    float scoreThreshold;
};


template <typename T>
__device__ float loadAsFloat(T const* address)
{
    return static_cast<float>(*address);
}

template<>
__device__ float loadAsFloat<__half>(__half const* address)
{
    return __half2float(*address);
}

__device__ int32_t reserveOutputSlot(int32_t* counter, int32_t maximum)
{
    int32_t oldValue = atomicAdd(counter, 0);

    while(oldValue < maximum){
        int32_t const previous = atomicCAS(counter,oldValue,oldValue+1);
        if(previous == oldValue)
        {
            return oldValue;
        }

        oldValue = previous;

    }

    return -1;
}


template <typename T>
__global__ void decodeKernel(
    T const* distances,
    T const* classProbabilities,
    float* detections,
    int32_t* detectionCounts,
    DecodeParameters parameters
)
{

}


cudaError_t launchDecode(
    nvinfer1::DataType inputType,
    void const* distances,
    void const* classProbabilities,
    float* detections,
    int32_t* detectionCounts,
    DecodeParameters const& parameters,
    cudaStream_t stream
)
{
    constexpr int32_t threads = 256;
    int32_t const total = parameters.batchSize * parameters.numPredictions;
    int32_t const blocks = (total + threads - 1) / threads;

    if(inputType == nvinfer1::DataType::kFLOAT)
    {
        decodeKernel<<<blocks,threads,0,stream>>>(
            static_cast<float const*>(distances),
            static_cast<float const*>(classProbabilities),
            detections,
            detectionCounts,
            parameters
        );
    }else if(inputType == nvinfer1::DataType::kHALF)
    {
        decodeKernel<<<blocks,threads,0,stream>>>(
            static_cast<__half const*>(distances),
            static_cast<__half const*>(classProbabilities),
            detections,
            detectionCounts,
            parameters
        );
    }else{
        return cudaErrorInvalidValue;
    }

    return cudaPeekAtLastError();

}



} // 

// Yolov8DecodePlugin
Yolov8DecodePlugin::Yolov8DecodePlugin(
    int32_t numClasses,
    int32_t inputWidth,
    int32_t inputHeight,
    int32_t maxOutputBoxes,
    float scoreThreshold,
    std::vector<int32_t> strides): 
    mNumClasses(numClasses),
    mInputWidth(inputWidth),
    mInputHeight(inputHeight),
    mMaxOutputBoxes(maxOutputBoxes),
    mScoreThreshold(scoreThreshold),
    mStrides(strides)
{
    assert(!mStrides.empty());
    assert(mStrides.size() <= kMaxStrideCount);

}


Yolov8DecodePlugin::Yolov8DecodePlugin(void const* serialData, size_t serialLength)
{
    char const* cursor = static_cast<char const*>(serialData);
    char const* const end = cursor + serialLength;

    int32_t strideCount{};
    readValue(cursor, mNumClasses);
    readValue(cursor, mInputWidth);
    readValue(cursor, mInputHeight);
    readValue(cursor, mMaxOutputBoxes);
    readValue(cursor, mScoreThreshold);
    readValue(cursor, strideCount);  
    
    mStrides.resize(strideCount);
    for(int32_t& stride : mStrides)
    {
        readValue(cursor,stride);
    }

    assert(cursor == end);

}

int32_t Yolov8DecodePlugin::getNbOutputs() const noexcept
{
    return 2;
}

nvinfer1::DimsExprs Yolov8DecodePlugin::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output{};

    if (nbInputs != 2 || outputIndex < 0 || outputIndex >= 2)
    {
        output.nbDims = -1;
        return output;
    }

    if (outputIndex == 0)
    {
        output.nbDims = 3;
        output.d[0] = inputs[0].d[0];
        output.d[1] = exprBuilder.constant(mMaxOutputBoxes);
        output.d[2] = exprBuilder.constant(kDetectionWidth);
    }
    else
    {
        output.nbDims = 1;
        output.d[0] = inputs[0].d[0];
    }

    return output;
}

int32_t Yolov8DecodePlugin::initialize() noexcept
{
    return 0;
}

void Yolov8DecodePlugin::terminate() noexcept
{
}

size_t Yolov8DecodePlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*,
    int32_t,
    nvinfer1::PluginTensorDesc const*,
    int32_t) const noexcept
{
    return 0;
}


int32_t Yolov8DecodePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const*,
    void const* const* inputs,
    void* const* outputs,
    void*,
    cudaStream_t stream
) noexcept
{

    nvinfer1::Dims const& boxDims = inputDesc[0].dims;
    nvinfer1::Dims const& scoreDims = inputDesc[1].dims;

    if(boxDims.nbDims != 3 || scoreDims.nbDims !=3)
    {
        return 1;
    }

    // boxDims (batch,4,8400)  scoreDims (batch,80,8400)
    int32_t const batchSize = boxDims.d[0];
    int32_t const numPredictions = boxDims.d[2];

    if (batchSize <= 0
        || boxDims.d[1] != 4
        || scoreDims.d[0] != batchSize
        || scoreDims.d[1] != mNumClasses
        || scoreDims.d[2] != numPredictions)
    {
        return 1;
    }

    int32_t expectedPredictions = 0;
    for(int32_t const stride : mStrides)
    {
        expectedPredictions +=
            (mInputHeight / stride) * (mInputWidth / stride);
    }

    if(numPredictions != expectedPredictions) return 1;

    auto* detectionCounts = static_cast<int32_t*>(outputs[1]);
    cudaError_t status = cudaMemsetAsync(
        detectionCounts,
        0,
        static_cast<size_t>(batchSize) * sizeof(int32_t),
        stream
    );

    if(status != cudaSuccess) return 1;

    DecodeParameters parameters{};
    parameters.batchSize = batchSize;
    parameters.numClasses = mNumClasses;
    parameters.numPredictions = numPredictions;
    parameters.inputWidth = mInputWidth;
    parameters.inputHeight = mInputHeight;
    parameters.maxOutputBoxes = mMaxOutputBoxes;
    parameters.strideCount = static_cast<int32_t>(mStrides.size());
    parameters.scoreThreshold = mScoreThreshold;

    for (int32_t index = 0; index < parameters.strideCount; ++index)
    {
        parameters.strides[index] = mStrides[index];
    }

    status = launchDecode(
        inputDesc[0].type,
        inputs[0],
        inputs[1],
        static_cast<float*>(outputs[0]),
        detectionCounts,
        parameters,
        stream);

    return status == cudaSuccess ? 0 : 1;

}

size_t Yolov8DecodePlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * (5 + mStrides.size()) + sizeof(float);
}

void Yolov8DecodePlugin::serialize(void* buffer) const noexcept
{
    char* cursor = static_cast<char*>(buffer);

    writeValue(cursor, mNumClasses);
    writeValue(cursor, mInputWidth);
    writeValue(cursor, mInputHeight);
    writeValue(cursor, mMaxOutputBoxes);
    writeValue(cursor, mScoreThreshold);

    int32_t const strideCount = static_cast<int32_t>(mStrides.size());
    writeValue(cursor, strideCount);

    for (int32_t const stride : mStrides)
    {
        writeValue(cursor, stride);
    }
}

bool Yolov8DecodePlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept
{
  
    if(nbInputs != 2 || nbOutputs != 2 || pos < 0 || pos >=4) return false;
    
    nvinfer1::PluginTensorDesc const& descriptor = inOut[pos];
    
    if(descriptor.format != nvinfer1::TensorFormat::kLINEAR) return false;

    switch(pos){
        case 0:
            return descriptor.type == nvinfer1::DataType::kFLOAT
                || descriptor.type == nvinfer1::DataType::kHALF;
        case 1:
            return descriptor.type == inOut[0].type;
        case 2:
            return descriptor.type == nvinfer1::DataType::kFLOAT;
        case 3:
            return descriptor.type == nvinfer1::DataType::kINT32;
        default:
            return false;

    }

}

char const* Yolov8DecodePlugin::getPluginType() const noexcept
{
    return kPluginName;
}

char const* Yolov8DecodePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}


void Yolov8DecodePlugin::destroy() noexcept
{
    delete this;
}

nvinfer1::IPluginV2DynamicExt* Yolov8DecodePlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new Yolov8DecodePlugin(
            mNumClasses,
            mInputWidth,
            mInputHeight,
            mMaxOutputBoxes,
            mScoreThreshold,
            mStrides);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

void Yolov8DecodePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}


char const* Yolov8DecodePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType Yolov8DecodePlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const*,
    int32_t) const noexcept
{
    return index == 0
        ? nvinfer1::DataType::kFLOAT
        : nvinfer1::DataType::kINT32;
}

void Yolov8DecodePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const*,
    int32_t,
    nvinfer1::DynamicPluginTensorDesc const*,
    int32_t) noexcept
{
}

void Yolov8DecodePlugin::attachToContext(
    cudnnContext*,
    cublasContext*,
    nvinfer1::IGpuAllocator*) noexcept
{
}

void Yolov8DecodePlugin::detachFromContext() noexcept
{
}

// plugincreator
Yolov8DecodePluginCreator::Yolov8DecodePluginCreator()
{
    mFields.emplace_back(
        "num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back(
        "input_width", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back(
        "input_height", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back(
        "max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFields.emplace_back(
        "score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mFields.emplace_back(
        "strides", nullptr, nvinfer1::PluginFieldType::kINT32, 0);

    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

char const* Yolov8DecodePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Yolov8DecodePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::PluginFieldCollection const* Yolov8DecodePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

nvinfer1::IPluginV2* Yolov8DecodePluginCreator::createPlugin(
    char const*,
    nvinfer1::PluginFieldCollection const* fieldCollection) noexcept
{
    try{
        int32_t numClasses = 80;
        int32_t inputWidth = 640;
        int32_t inputHeight = 640;
        int32_t maxOutputBoxes = 1000;
        float scoreThreshold = 0.25F;
        std::vector<int32_t> strides{8,16,32};

        for(int32_t index = 0; index < fieldCollection->nbFields; ++index)
        {
            nvinfer1::PluginField const& field = fieldCollection->fields[index];

            if (std::strcmp(field.name, "num_classes") == 0)
            {
                numClasses = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "input_width") == 0)
            {
                inputWidth = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "input_height") == 0)
            {
                inputHeight = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "max_output_boxes") == 0)
            {
                maxOutputBoxes = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "score_threshold") == 0)
            {
                scoreThreshold = *static_cast<float const*>(field.data);
            }
            else if (std::strcmp(field.name, "strides") == 0)
            {
                auto const* values = static_cast<int32_t const*>(field.data);
                strides.assign(values,values+field.length);
            }

        }

        auto* plugin = new Yolov8DecodePlugin(
            numClasses,
            inputWidth,
            inputHeight,
            maxOutputBoxes,
            scoreThreshold,
            std::move(strides));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;

    }
    catch(...){
        return nullptr;
    }
}

nvinfer1::IPluginV2* Yolov8DecodePluginCreator::deserializePlugin(
    char const*,
    void const* serialData,
    size_t serialLength) noexcept
{
    try
    {
        auto* plugin = new Yolov8DecodePlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

void Yolov8DecodePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

char const* Yolov8DecodePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(Yolov8DecodePluginCreator);


}