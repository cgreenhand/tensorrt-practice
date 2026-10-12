#pragma once 
#include <cuda_runtime.h>
#include "config.h"


static_assert(sizeof(Detection) == 6 * sizeof(float));

static int32_t kCandidateCount = (yolov8::det::kInputHeight / 8 ) 
                                    * (yolov8::det::kInputWidth / 8) + (yolov8::det::kInputHeight / 16) 
                                    * (yolov8::det::kInputWidth / 16) + (yolov8::det::kInputHeight / 32) 
                                    * (yolov8::det::kInputWidth / 32);

struct PostprocessConfig
{
    int32_t batchSize{yolov8::det::kBatchSize};

    // Number of rows produced by Yolov8DecodePlugin for each image.
    // At 640x640: 80*80 + 40*40 + 20*20 = 8400.
    int32_t candidateCount{kCandidateCount};

    int32_t preNmsTopK{yolov8::det::kNmsThreshold};
    int32_t maxDetections{yolov8::det::kMaxOutputBoxes};
    float scoreThreshold{yolov8::det::kConfidenceThreshold};
    float nmsThreshold{yolov8::det::kNmsThreshold};
    bool classAwareNms{true};
    bool restoreToOriginalImage{true};
};


class CudaPostprocessor
{
public:
    explicit CudaPostprocessor(PostprocessConfig config = {});
    ~CudaPostprocessor() noexcept;

    
    CudaPostprocessor(CudaPostprocessor const&) = delete;
    CudaPostprocessor& operator=(CudaPostprocessor const&) = delete;

    // decodedDetections: TensorRT plugin output [B, candidateCount, 6], FP32.
    // decodedCounts:     TensorRT plugin output [B], INT32.
    // outputDetections:  caller-owned device buffer [B, maxDetections].
    // outputCounts:      caller-owned device buffer [B].
    //
    // hostLetterboxes must contain validBatchSize entries and must remain alive
    // and unchanged until the stream reaches the queued host-to-device copy.
    // The function only enqueues work; it does not synchronize or copy results
    // back to the CPU.
    void enqueue(
        float const* decodeDetections,
        int32_t const* decodeCounts,
        LetterboxInfo const* hostLetterboxes,
        int32_t validBatchSize,
        Detection* outputDetections,
        int32_t* outputCounts,
        cudaStream_t stream
    );
    


private:
    class Workspace;
    PostprocessConfig mConfig;
    int32_t mTopK{};
    int32_t mMaskColumnBlocks{};
    std::size_t mCubTemporaryBytes{};

    Workspace* mWorkspace{};
};

