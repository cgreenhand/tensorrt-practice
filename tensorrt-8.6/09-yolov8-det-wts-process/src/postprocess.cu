#include "postprocess.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <utility>
#include <memory>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>


namespace yolov8
{
namespace
{
#define POSTPROCESS_CUDA_CHECK(call)                                                     \
    do                                                                                  \
    {                                                                                   \
        cudaError_t const error__ = (call);                                              \
        if (error__ != cudaSuccess)                                                      \
        {                                                                               \
            throw std::runtime_error(                                                    \
                std::string("CUDA error: ") + cudaGetErrorString(error__) +            \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));                  \
        }                                                                               \
    } while (false)

constexpr int32_t kDecodedWidth = 6;
constexpr int32_t kNmsThreads = 64;

class DeviceBuffer
{

public:
    DeviceBuffer() = default;
    ~DeviceBuffer() noexcept
    {
        if(mPointer != nullptr) cudaFree(mPointer);
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    void reset(std::size_t bytes)
    {
        if(mPointer != nullptr)
        {
            POSTPROCESS_CUDA_CHECK(cudaFree(mPointer));
            mPointer = nullptr;
            mBytes = 0;
        }
        if(bytes != 0)
        {
            POSTPROCESS_CUDA_CHECK(cudaMalloc(&mPointer, bytes));
            mBytes = bytes;
        }
    }

    void* pointer() noexcept
    {
        return mPointer;
    }

    void const* data() const noexcept
    {
        return mPointer;
    }



private:
    void* mPointer{};
    std::size_t mBytes{};

};

__device__ __forceinline__ float clampValue(float value, float low, float high)
{
    return fminf(fmaxf(value,low),high);
}

__global__ void prepareSortInputKernel(
    float const* decodedDetections,
    int32_t const* decodedCounts,
    float* scores,
    int32_t* indices,
    int32_t candidateCount,
    float scoreThreshold    
)
{
    int32_t const candidate = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const batchIndex = blockIdx.y;
    if(candidate >= candidateCount){
        return;
    }

    int32_t validCount = decodedCounts[batchIndex];
    validCount = max(0,min(validCount,candidateCount));

    int32_t const flatIndex = batchIndex * candidateCount + candidate;
    indices[flatIndex] = candidate;

    float score = -INFINITY;

    if(candidate < validCount)
    {
        std::size_t const detectionOffset = flatIndex * kDecodedWidth;
        float const candidateScore = decodedDetections[detectionOffset + 4];
        if(isfinite(candidateScore) && candidateScore > scoreThreshold)
        {
            score = candidateScore;
        }

    }
    scores[flatIndex] = score;

}


__device__ __forceinline__ float intersectionOverUnion(
    float const* first,
    float const* second
)
{
    float const left = fmaxf(first[0], second[0]);
    float const top = fmaxf(first[1], second[1]);
    float const right = fminf(first[2], second[2]);
    float const bottom = fminf(first[3], second[3]);

    float intersectionWidth = fmaxf(0.0F,right - left);
    float intersectionHeight = fmaxf(0.0F, bottom - top);
    float const intersection = intersectionWidth * intersectionHeight;

        float const firstArea =
        fmaxf(0.0F, first[2] - first[0]) *
        fmaxf(0.0F, first[3] - first[1]);
    float const secondArea =
        fmaxf(0.0F, second[2] - second[0]) *
        fmaxf(0.0F, second[3] - second[1]);
    float const unionArea = firstArea + secondArea - intersection;

    return  unionArea>0.0F ?  intersection / unionArea : 0.0F;
}


__global__ void buildNmsMaskKernel(
    float const* decodedDetections,
    float const* sortedScores,
    int32_t const* sortedIndices,
    std::uint64_t* masks,
    int32_t candidateCount,
    int32_t topK,
    int32_t maskColumnBlocks,
    float scoreThreshold,
    float nmsThreshold,
    bool classAwareNms)
{
    int32_t const columnBlock = blockIdx.x;
    int32_t const rowBlock = blockIdx.y;
    int32_t const batchIndex = blockIdx.z;

    if(columnBlock < rowBlock) return;

    int32_t const columnStart = columnBlock * kNmsThreads;
    int32_t const rowStart = rowBlock * kNmsThreads;
    int32_t const columnSize = min(kNmsThreads, topK - columnStart);
    int32_t const rowSize = min(kNmsThreads, topK - rowStart);

    // 将数据复制到共享内存
    __shared__ float columnBoxes[kNmsThreads][4];
    __shared__ int32_t columnClasses[kNmsThreads];
    __shared__ float columnScores[kNmsThreads];

    if(threadIdx.x < columnSize)
    {
        int32_t const sortedPosition = columnStart + threadIdx.x;
        int32_t const candidate = sortedIndices[batchIndex * candidateCount + sortedPosition];
        std::size_t const sourceOffset = (static_cast<std::size_t>(batchIndex) * candidateCount + candidate) * kDecodedWidth;
        columnBoxes[threadIdx.x][0] = decodedDetections[sourceOffset + 0];
        columnBoxes[threadIdx.x][1] = decodedDetections[sourceOffset + 1];
        columnBoxes[threadIdx.x][2] = decodedDetections[sourceOffset + 2];
        columnBoxes[threadIdx.x][3] = decodedDetections[sourceOffset + 3];
        columnClasses[threadIdx.x] = static_cast<int32_t>(decodedDetections[sourceOffset + 5]);
        columnScores[threadIdx.x] = sortedScores[batchIndex * candidateCount + sortedPosition];

    }

    __syncthreads();

    // 计算iou

    if(threadIdx.x >= rowSize) return;
    int32_t const rowPosition = rowStart + threadIdx.x;
    float const rowScore = sortedScores[batchIndex * candidateCount + rowPosition];
    std::uint64_t suppressionMask = 0;

    if(rowScore >= scoreThreshold)
    {
        int32_t const rowCandidate = sortedIndices[batchIndex * candidateCount + rowPosition];
        std::size_t const rowOffset = (static_cast<std::size_t>(batchIndex) * candidateCount + rowCandidate) * kDecodedWidth;
        float const* rowDetection = decodedDetections + rowOffset;
        float const rowClass = static_cast<int32_t>(rowDetection[5]);

        int32_t const firstColumn = rowBlock == columnBlock ? threadIdx.x + 1 : 0;

        for(int32_t column = firstColumn; column < columnSize; column++)
        {
            if(columnScores[column] < scoreThreshold)
            {
                continue;
            }
            if(classAwareNms && rowClass != columnClasses[column])
            {
                continue;
            }
            if(intersectionOverUnion(rowDetection, columnBoxes[column]) > nmsThreshold)
            {
                suppressionMask |= (std::uint64_t(1) << column);
            }

        }
    }

    std::size_t const maskOffset = (static_cast<std::size_t>(batchIndex) * topK + rowPosition) * maskColumnBlocks + columnBlock;
    masks[maskOffset] = suppressionMask;

}

__device__ __forceinline__ float resoreCoordinate(
    float value,
    float padding,
    float scale,
    float maximum
)
{
    return scale > 0.0F ? clampValue((value - padding) / scale, 0.0F, maximum) : 0.0F;
}

__global__ void selectNmsOutputKernel(
    float const* decodedDetections,
    float const* sortedScores,
    int32_t const* sortedIndices,
    std::uint64_t const* masks,
    std::uint64_t* removedMasks,
    LetterboxInfo const* letterboxes,
    Detection* outputDetections,
    int32_t* outputCounts,
    int32_t validBatchSize,
    int32_t candidateCount,
    int32_t topK,
    int32_t maskColumnBlocks,
    int32_t maxDetections,
    float scoreThreashold,
    bool restoreToOriginalImage)
{
    int32_t const batchIndex = blockIdx.x;
    if(threadIdx.x != 0) return ;
    if(batchIndex >= validBatchSize){
        outputCounts[batchIndex] = 0;
        return ;
    }

    std::uint64_t* batchRemoved = removedMasks + static_cast<std::size_t>(batchIndex) * maskColumnBlocks;
    for(int32_t block = 0; block < maskColumnBlocks; block++)
    {
        batchRemoved[block] = 0;
    }

    LetterboxInfo letterbox{};
    if(restoreToOriginalImage)
    {
        letterbox = letterboxes[batchIndex];
    }

    int32_t keptCount = 0;

    for(int32_t sortedPosition = 0; sortedPosition < topK && keptCount < maxDetections; sortedPosition++)
    {
        float const score = sortedScores[batchIndex * candidateCount + sortedPosition];
        if (score < scoreThreashold) break;

        int32_t const block = sortedPosition / kNmsThreads;
        int32_t const bit = sortedPosition % kNmsThreads;
        if((batchRemoved[block]&(std::uint64_t{1} << bit)) != 0) continue;

        int32_t const candidate = sortedIndices[batchIndex * candidateCount + sortedPosition];
        std::size_t const sourceOffset =
            (static_cast<std::size_t>(batchIndex) * candidateCount + candidate) *
            kDecodedWidth;
        float const* source = decodedDetections + sourceOffset;

        Detection result{};
        result.x1 = source[0];
        result.y1 = source[1];
        result.x2 = source[2];
        result.y2 = source[3];
        result.score = score;
        result.classId = static_cast<int32_t>(source[5]);


        if(restoreToOriginalImage){
            float const maximumX = static_cast<float>(max(0,letterbox.originalWidth - 1));
            float const maximumY = static_cast<float>(max(0,letterbox.originalHeight - 1));

            result.x1 = resoreCoordinate(result.x1, static_cast<float>(letterbox.padLeft), letterbox.scaleX, maximumX);
            result.y1 = resoreCoordinate(result.y1, static_cast<float>(letterbox.padTop), letterbox.scaleY, maximumY);
            result.x2 = resoreCoordinate(result.x2, static_cast<float>(letterbox.padLeft), letterbox.scaleX, maximumX);
            result.y2 = resoreCoordinate(result.y2, static_cast<float>(letterbox.padTop), letterbox.scaleY, maximumY);

        }

        outputDetections[static_cast<std::size_t>(batchIndex)* maxDetections + keptCount] = result;
        ++keptCount;

        std::uint64_t const* candidateMask = masks + (static_cast<std::size_t>(batchIndex) * topK + sortedPosition) * maskColumnBlocks;
        for(int32_t columnBlock = block; columnBlock < maskColumnBlocks; ++columnBlock)
        {
            batchRemoved[columnBlock] |= candidateMask[columnBlock];
        }

        outputCounts[batchIndex] = keptCount;


    }


}
}// namespace

class CudaPostprocessor::Workspace
{
public:
    DeviceBuffer scoresIn;
    DeviceBuffer scoresOut;
    DeviceBuffer indicesIn;
    DeviceBuffer indicesOut;
    DeviceBuffer segmentOffsets;
    DeviceBuffer nmsMasks;
    DeviceBuffer removedMasks;
    DeviceBuffer letterboxes;
    DeviceBuffer cubTemporary;
};

CudaPostprocessor::~CudaPostprocessor() noexcept
{
    delete mWorkspace;
}


CudaPostprocessor::CudaPostprocessor(PostprocessConfig config)
    : mConfig(config)
{
    if (mConfig.batchSize <= 0 || mConfig.candidateCount <= 0)
    {
        throw std::invalid_argument("Invalid postprocess batch or candidate count");
    }
    if (mConfig.preNmsTopK <= 0 || mConfig.maxDetections <= 0)
    {
        throw std::invalid_argument("Top-K and max detections must be positive");
    }
    if (mConfig.scoreThreshold < 0.0F || mConfig.scoreThreshold > 1.0F ||
        mConfig.nmsThreshold < 0.0F || mConfig.nmsThreshold > 1.0F)
    {
        throw std::invalid_argument("Score and NMS thresholds must be in [0, 1]");
    }  

    mTopK = std::min(mConfig.preNmsTopK, mConfig.candidateCount);
    mMaskColumnBlocks = (mTopK + kNmsThreads - 1) / kNmsThreads;

    std::size_t const totalCandidates = static_cast<std::size_t>(mConfig.batchSize) * mConfig.candidateCount;
    if(totalCandidates > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::invalid_argument("Too many candidates for CUB segmented sort");
    }

    auto workspace = std::make_unique<Workspace>();
    workspace->scoresIn.reset(totalCandidates * sizeof(float));
    workspace->scoresOut.reset(totalCandidates * sizeof(float));
    workspace->indicesIn.reset(totalCandidates * sizeof(int32_t));
    workspace->indicesOut.reset(totalCandidates * sizeof(int32_t));
    workspace->segmentOffsets.reset(
        static_cast<std::size_t>(mConfig.batchSize + 1) * sizeof(int32_t));
    workspace->nmsMasks.reset(
        static_cast<std::size_t>(mConfig.batchSize) * mTopK *
        mMaskColumnBlocks * sizeof(std::uint64_t));
    workspace->removedMasks.reset(
        static_cast<std::size_t>(mConfig.batchSize) * mMaskColumnBlocks *
        sizeof(std::uint64_t));
    workspace->letterboxes.reset(
        static_cast<std::size_t>(mConfig.batchSize) * sizeof(LetterboxInfo));


    std::vector<int32_t> offsets(
        static_cast<std::size_t>(mConfig.batchSize + 1));
    for (int32_t batchIndex = 0; batchIndex <= mConfig.batchSize; ++batchIndex)
    {
        offsets[batchIndex] = batchIndex * mConfig.candidateCount;
    }

    POSTPROCESS_CUDA_CHECK(cudaMemcpy(
        workspace->segmentOffsets.pointer(),
        offsets.data(),
        offsets.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    POSTPROCESS_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        mCubTemporaryBytes,
        static_cast<float const*>(workspace->scoresIn.data()),
        static_cast<float*>(workspace->scoresOut.pointer()),
        static_cast<int32_t const*>(workspace->indicesIn.data()),
        static_cast<int32_t*>(workspace->indicesOut.pointer()),
        static_cast<int32_t>(totalCandidates),
        mConfig.batchSize,
        static_cast<int32_t const*>(workspace->segmentOffsets.data()),
        static_cast<int32_t const*>(workspace->segmentOffsets.data()) + 1));
    
    workspace->cubTemporary.reset(mCubTemporaryBytes);

    mWorkspace = workspace.release();

}

void CudaPostprocessor::enqueue(
    float const* decodedDetections,
    int32_t const* decodedCounts,
    LetterboxInfo const* hostLetterboxes,
    int32_t validBatchSize,
    Detection* outputDetections,
    int32_t* outputCounts,
    cudaStream_t stream)
{
    if(decodedDetections == nullptr || decodedCounts == nullptr || outputDetections == nullptr || outputCounts == nullptr)
    {
        throw std::invalid_argument("Null device pointer passed to postprocess");
    }
    if(validBatchSize < 0 || validBatchSize > mConfig.batchSize)
    {
        throw std::invalid_argument("Invalid batch size");
    }
    if (mConfig.restoreToOriginalImage &&
        validBatchSize > 0 && hostLetterboxes == nullptr)
    {
        throw std::invalid_argument("Letterbox metadata is required for restoration");
    }

    int32_t constexpr threads = 256;
    dim3 const prepareGrid(
        static_cast<unsigned int>((mConfig.candidateCount + threads - 1) / threads),
        static_cast<unsigned int>(mConfig.batchSize));

    prepareSortInputKernel<<<prepareGrid, threads, 0, stream>>>(
        decodedDetections,
        decodedCounts,
        static_cast<float*>(mWorkspace->scoresIn.pointer()),
        static_cast<int32_t*>(mWorkspace->indicesIn.pointer()),
        mConfig.candidateCount,
        mConfig.scoreThreshold);
    POSTPROCESS_CUDA_CHECK(cudaGetLastError());

    std::size_t const totalCandidates =
        static_cast<std::size_t>(mConfig.batchSize) * mConfig.candidateCount;
    POSTPROCESS_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        mWorkspace->cubTemporary.pointer(),
        mCubTemporaryBytes,
        static_cast<float const*>(mWorkspace->scoresIn.data()),
        static_cast<float*>(mWorkspace->scoresOut.pointer()),
        static_cast<int32_t const*>(mWorkspace->indicesIn.data()),
        static_cast<int32_t*>(mWorkspace->indicesOut.pointer()),
        static_cast<int32_t>(totalCandidates),
        mConfig.batchSize,
        static_cast<int32_t const*>(mWorkspace->segmentOffsets.data()),
        static_cast<int32_t const*>(mWorkspace->segmentOffsets.data()) + 1,
        0,
        sizeof(float) * 8,
        stream));

    dim3 const nmsGrid(
        static_cast<unsigned int>(mMaskColumnBlocks),
        static_cast<unsigned int>(mMaskColumnBlocks),
        static_cast<unsigned int>(mConfig.batchSize));
    buildNmsMaskKernel<<<nmsGrid, kNmsThreads, 0, stream>>>(
        decodedDetections,
        static_cast<float const*>(mWorkspace->scoresOut.data()),
        static_cast<int32_t const*>(mWorkspace->indicesOut.data()),
        static_cast<std::uint64_t*>(mWorkspace->nmsMasks.pointer()),
        mConfig.candidateCount,
        mTopK,
        mMaskColumnBlocks,
        mConfig.scoreThreshold,
        mConfig.nmsThreshold,
        mConfig.classAwareNms);
    POSTPROCESS_CUDA_CHECK(cudaGetLastError());

    if (mConfig.restoreToOriginalImage && validBatchSize > 0)
    {
        POSTPROCESS_CUDA_CHECK(cudaMemcpyAsync(
            mWorkspace->letterboxes.pointer(),
            hostLetterboxes,
            static_cast<std::size_t>(validBatchSize) * sizeof(LetterboxInfo),
            cudaMemcpyHostToDevice,
            stream));
    }

    POSTPROCESS_CUDA_CHECK(cudaMemsetAsync(
        outputDetections,
        0,
        outputDetectionBytes(),
        stream));
    POSTPROCESS_CUDA_CHECK(cudaMemsetAsync(
        outputCounts,
        0,
        outputCountBytes(),
        stream));

    selectNmsOutputKernel<<<mConfig.batchSize, 1, 0, stream>>>(
        decodedDetections,
        static_cast<float const*>(mWorkspace->scoresOut.data()),
        static_cast<int32_t const*>(mWorkspace->indicesOut.data()),
        static_cast<std::uint64_t const*>(mWorkspace->nmsMasks.data()),
        static_cast<std::uint64_t*>(mWorkspace->removedMasks.pointer()),
        static_cast<LetterboxInfo const*>(mWorkspace->letterboxes.data()),
        outputDetections,
        outputCounts,
        validBatchSize,
        mConfig.candidateCount,
        mTopK,
        mMaskColumnBlocks,
        mConfig.maxDetections,
        mConfig.scoreThreshold,
        mConfig.restoreToOriginalImage);
    POSTPROCESS_CUDA_CHECK(cudaGetLastError());
    

}



int32_t CudaPostprocessor::batchSize() const noexcept
{
    return mConfig.batchSize;
}

int32_t CudaPostprocessor::candidateCount() const noexcept
{
    return mConfig.candidateCount;
}

int32_t CudaPostprocessor::maxDetections() const noexcept
{
    return mConfig.maxDetections;
}

std::size_t CudaPostprocessor::outputDetectionBytes() const noexcept
{
    return static_cast<std::size_t>(mConfig.batchSize) *
        mConfig.maxDetections * sizeof(Detection);
}

std::size_t CudaPostprocessor::outputCountBytes() const noexcept
{
    return static_cast<std::size_t>(mConfig.batchSize) * sizeof(int32_t);
}


}

