#pragma once

#include <cstddef>
#include <cstdint>

namespace yolov8::det
{

enum class Precision
{
    FP32,
    FP16,
    INT8
};

inline constexpr Precision kPrecision =
    Precision::FP16;

inline constexpr char kInputTensorName[] =
    "images";

inline constexpr char kOutputTensorName[] =
    "output";

inline constexpr int32_t kNumClasses = 80;

inline constexpr int32_t kBatchSize = 1;

inline constexpr int32_t kInputChannels = 3;
inline constexpr int32_t kInputHeight = 640;
inline constexpr int32_t kInputWidth = 640;

inline constexpr int32_t kMaxOutputBoxes = 1000;

inline constexpr float kConfidenceThreshold = 0.5F;
inline constexpr float kNmsThreshold = 0.45F;

inline constexpr int32_t kGpuId = 0;

inline constexpr std::size_t kWorkspaceBytes =
    4ULL << 30;

inline constexpr char kCalibrationDirectory[] =
    "./coco_calib";

static_assert(kNumClasses > 0);
static_assert(kBatchSize > 0);
static_assert(kInputHeight > 0);
static_assert(kInputWidth > 0);

}  // namespace yolov8::det