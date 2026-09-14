find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# 基础 TensorRT Runtime 依赖
add_library(trt_runtime_dependencies INTERFACE)

target_link_libraries(
    trt_runtime_dependencies
    INTERFACE
        TensorRT::nvinfer
        CUDA::cudart
        ${OpenCV_LIBS}
)

target_include_directories(
    trt_runtime_dependencies
    INTERFACE
        ${OpenCV_INCLUDE_DIRS}
)

add_library(
    Learning::TensorRTRuntime
    ALIAS trt_runtime_dependencies
)

# ONNX Builder 额外需要 nvonnxparser
add_library(trt_onnx_dependencies INTERFACE)

target_link_libraries(
    trt_onnx_dependencies
    INTERFACE
        Learning::TensorRTRuntime
        TensorRT::onnxparser
)

add_library(
    Learning::TensorRTOnnx
    ALIAS trt_onnx_dependencies
)