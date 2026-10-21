#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "config.h"

class CudaYoloPreprocessor
{

public:
    CudaYoloPreprocessor(int targetWidth, int targetHeight, int fixedBatchSize);
    ~CudaYoloPreprocessor();

    CudaYoloPreprocessor(CudaYoloPreprocessor const&) = delete;
    CudaYoloPreprocessor& operator=(CudaYoloPreprocessor const&) = delete;


    std::vector<LetterboxInfo> preprocess(
        std::vector<cv::Mat> const& images,
        void* deviceTensor,
        nvinfer1::DataType dataType,
        cudaStream_t stream
    );

private:
    void ensureSourceCapacity(std::size_t bytes);

    int mTargetWidth{};
    int mTargetHeight{};
    int mBatchSize{};
    std::uint8_t* mDeviceSource{};
    std::size_t mSourceCapacity{};



};