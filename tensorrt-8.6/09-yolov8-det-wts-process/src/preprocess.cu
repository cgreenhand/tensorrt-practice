#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>
#include <cuda_fp16.h>


namespace
{

#define CUDA_CHECK(call)                                                                    \
    do                                                                                      \
    {                                                                                       \
        cudaError_t const error__ = (call);                                                  \
        if (error__ != cudaSuccess)                                                          \
        {                                                                                   \
            throw std::runtime_error(                                                        \
                std::string("CUDA error: ") + cudaGetErrorString(error__) +                 \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));                       \
        }                                                                                   \
    } while (false)

constexpr std::uint8_t kPaddingValue = 114;
constexpr float kInverse255 = 1.0F / 255.0F;

LetterboxInfo calculateLetterboxInfo(
    int sourceWidth,
    int sourceHeight,
    int targetWidth,
    int targetHeight)
{
    float const scale = std::min(
        static_cast<float>(targetWidth) / static_cast<float>(sourceWidth),
        static_cast<float>(targetHeight) / static_cast<float>(sourceHeight));

    int const resizedWidth = std::min(
        targetWidth,
        std::max(1, static_cast<int>(std::lround(static_cast<double>(sourceWidth) * scale))));
    int const resizedHeight = std::min(
        targetHeight,
        std::max(1, static_cast<int>(std::lround(static_cast<double>(sourceHeight) * scale))));

    return LetterboxInfo{
        sourceWidth,
        sourceHeight,
        resizedWidth,
        resizedHeight,
        (targetWidth - resizedWidth) / 2,
        (targetHeight - resizedHeight) / 2,
        static_cast<float>(resizedWidth) / static_cast<float>(sourceWidth),
        static_cast<float>(resizedHeight) / static_cast<float>(sourceHeight)};
}

__device__ __forceinline__ float sampleBilinear(
    std::uint8_t const* source,
    std::size_t sourceStep,
    int sourceWidth,
    int sourceHeight,
    float sourceX,
    float sourceY,
    int bgrChannel)
{
    sourceX = fminf(fmaxf(sourceX, 0.0F), static_cast<float>(sourceWidth - 1));
    sourceY = fminf(fmaxf(sourceY, 0.0F), static_cast<float>(sourceHeight - 1));

    int const x0 = static_cast<int>(floorf(sourceX));
    int const y0 = static_cast<int>(floorf(sourceY));
    int const x1 = min(x0 + 1, sourceWidth - 1);
    int const y1 = min(y0 + 1, sourceHeight - 1);

    float const dx = sourceX - static_cast<float>(x0);
    float const dy = sourceY - static_cast<float>(y0);

    std::uint8_t const* row0 = source + static_cast<std::size_t>(y0) * sourceStep;
    std::uint8_t const* row1 = source + static_cast<std::size_t>(y1) * sourceStep;

    float const value00 = static_cast<float>(row0[x0 * 3 + bgrChannel]);
    float const value01 = static_cast<float>(row0[x1 * 3 + bgrChannel]);
    float const value10 = static_cast<float>(row1[x0 * 3 + bgrChannel]);
    float const value11 = static_cast<float>(row1[x1 * 3 + bgrChannel]);

    float const top = value00 + (value01 - value00) * dx;
    float const bottom = value10 + (value11 - value10) * dx;
    return top + (bottom - top) * dy;
}

__device__ __forceinline__ void storeValue(float* address, float value)
{
    *address = value;
}

__device__ __forceinline__ void storeValue(__half* address, float value)
{
    *address = __float2half_rn(value);
}

template <typename OutputType>
__global__ void preprocessKernel(
    std::uint8_t const* source,
    std::size_t sourceStep,
    int sourceWidth,
    int sourceHeight,
    OutputType* destination,
    int targetWidth,
    int targetHeight,
    int resizedWidth,
    int resizedHeight,
    int padLeft,
    int padTop)
{
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= targetWidth || y >= targetHeight)
    {
        return;
    }

    int const resizedX = x - padLeft;
    int const resizedY = y - padTop;
    std::size_t const pixel = static_cast<std::size_t>(y) * targetWidth + x;
    std::size_t const plane = static_cast<std::size_t>(targetWidth) * targetHeight;

    float red = static_cast<float>(kPaddingValue);
    float green = static_cast<float>(kPaddingValue);
    float blue = static_cast<float>(kPaddingValue);

    if (resizedX >= 0 && resizedX < resizedWidth &&
        resizedY >= 0 && resizedY < resizedHeight)
    {
        // Half-pixel mapping agrees closely with OpenCV INTER_LINEAR.
        float const sourceX =
            (static_cast<float>(resizedX) + 0.5F) *
                static_cast<float>(sourceWidth) / static_cast<float>(resizedWidth) -
            0.5F;
        float const sourceY =
            (static_cast<float>(resizedY) + 0.5F) *
                static_cast<float>(sourceHeight) / static_cast<float>(resizedHeight) -
            0.5F;

        // OpenCV input is BGR. TensorRT YOLOv8 input is RGB.
        blue = sampleBilinear(
            source, sourceStep, sourceWidth, sourceHeight, sourceX, sourceY, 0);
        green = sampleBilinear(
            source, sourceStep, sourceWidth, sourceHeight, sourceX, sourceY, 1);
        red = sampleBilinear(
            source, sourceStep, sourceWidth, sourceHeight, sourceX, sourceY, 2);
    }

    // CHW layout, normalized to [0, 1].
    storeValue(destination + 0 * plane + pixel, red * kInverse255);
    storeValue(destination + 1 * plane + pixel, green * kInverse255);
    storeValue(destination + 2 * plane + pixel, blue * kInverse255);
}

template <typename OutputType>
__global__ void fillUnusedBatchKernel(
    OutputType* destination,
    std::size_t elementCount,
    float normalizedPadding)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elementCount)
    {
        storeValue(destination + index, normalizedPadding);
    }
}

template <typename OutputType>
void launchPreprocess(
    std::uint8_t const* source,
    std::size_t sourceStep,
    cv::Mat const& image,
    OutputType* batchDestination,
    int targetWidth,
    int targetHeight,
    LetterboxInfo const& info,
    cudaStream_t stream)
{
    dim3 const block(16, 16);
    dim3 const grid(
        static_cast<unsigned int>((targetWidth + block.x - 1) / block.x),
        static_cast<unsigned int>((targetHeight + block.y - 1) / block.y));

    preprocessKernel<<<grid, block, 0, stream>>>(
        source,
        sourceStep,
        image.cols,
        image.rows,
        batchDestination,
        targetWidth,
        targetHeight,
        info.resizedWidth,
        info.resizedHeight,
        info.padLeft,
        info.padTop);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OutputType>
void launchFill(
    OutputType* batchDestination,
    std::size_t elementCount,
    cudaStream_t stream)
{
    int constexpr threads = 256;
    int const blocks = static_cast<int>((elementCount + threads - 1) / threads);
    fillUnusedBatchKernel<<<blocks, threads, 0, stream>>>(
        batchDestination,
        elementCount,
        static_cast<float>(kPaddingValue) * kInverse255);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OutputType>
void enqueueBatch(
    std::vector<cv::Mat> const& images,
    std::uint8_t* deviceSource,
    OutputType* deviceTensor,
    int batchSize,
    int targetWidth,
    int targetHeight,
    std::vector<LetterboxInfo> const& infos,
    cudaStream_t stream)
{
    std::size_t const elementsPerImage =
        static_cast<std::size_t>(3) * targetWidth * targetHeight;

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
    {
        OutputType* batchDestination =
            deviceTensor + static_cast<std::size_t>(batchIndex) * elementsPerImage;

        if (batchIndex >= static_cast<int>(images.size()))
        {
            launchFill(batchDestination, elementsPerImage, stream);
            continue;
        }

        cv::Mat const& image = images[batchIndex];
        std::size_t const rowBytes = static_cast<std::size_t>(image.cols) * 3U;

        // The staging buffer is reused safely because copy and kernel commands
        // are ordered on the same stream: copy_i -> kernel_i -> copy_i+1.
        CUDA_CHECK(cudaMemcpy2DAsync(
            deviceSource,
            rowBytes,
            image.data,
            image.step,
            rowBytes,
            static_cast<std::size_t>(image.rows),
            cudaMemcpyHostToDevice,
            stream));

        launchPreprocess(
            deviceSource,
            rowBytes,
            image,
            batchDestination,
            targetWidth,
            targetHeight,
            infos[batchIndex],
            stream);
    }
}

} // namespace



CudaYoloPreprocessor::CudaYoloPreprocessor(int targetWidth, int targetHeight, int fixedBatchSize)
: mTargetWidth(targetWidth)
, mTargetHeight(targetHeight)
, mBatchSize(fixedBatchSize)
{
    if(mBatchSize <= 0 || mTargetWidth <= 0 || mTargetHeight <= 0)
        throw std::invalid_argument("invalid argument");
}

CudaYoloPreprocessor::~CudaYoloPreprocessor()
{
    if(mDeviceSource != nullptr)
        CUDA_CHECK(cudaFree(mDeviceSource));
}

void CudaYoloPreprocessor::ensureSourceCapacity(std::size_t bytes)
{
    if(bytes <= mSourceCapacity)
    {
        return ;
    }

    if(mDeviceSource != nullptr)
    {
        CUDA_CHECK(cudaFree(mDeviceSource));
        mDeviceSource = nullptr;
        mSourceCapacity = 0;
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&mDeviceSource),bytes)
    );
    mSourceCapacity = bytes;

}

std::vector<LetterboxInfo> CudaYoloPreprocessor::preprocess(
        std::vector<cv::Mat> const& images,
        void* deviceTensor,
        nvinfer1::DataType tensorType,
        cudaStream_t stream)
{
    if(deviceTensor == nullptr)
        throw std::invalid_argument("deviceTensor is null");
    if(images.size() > static_cast<std::size_t>(mBatchSize))
        throw std::invalid_argument("batch size too large");
    
    std::size_t maximumSourceBytes = 0;
    std::vector<LetterboxInfo> infos;
    infos.reserve(images.size());

    for(cv::Mat const& image: images)
    {
        if(image.empty() || image.type() != CV_8UC3)
            throw std::invalid_argument("invalid image");

        std::size_t const bytes = static_cast<std::size_t>(image.cols) * image.rows * 3U;
        maximumSourceBytes = std::max(maximumSourceBytes, bytes);
        infos.push_back(calculateLetterboxInfo(image.cols, image.rows, mTargetWidth, mTargetHeight));

    }

    if (maximumSourceBytes > 0)
    {
        ensureSourceCapacity(maximumSourceBytes);
    }

    if(tensorType == nvinfer1::DataType::kFLOAT){
        enqueueBatch(
            images,
            mDeviceSource,
            static_cast<float*>(deviceTensor),
            mBatchSize,
            mTargetWidth,
            mTargetHeight,
            infos,
            stream
        );
    }else if(tensorType == nvinfer1::DataType::kHALF){
        enqueueBatch(
            images,
            mDeviceSource,
            static_cast<half*>(deviceTensor),
            mBatchSize,
            mTargetWidth,
            mTargetHeight,
            infos,
            stream
        );
    }else{
        throw std::invalid_argument("unsupported data type");
    }

    return infos;

}