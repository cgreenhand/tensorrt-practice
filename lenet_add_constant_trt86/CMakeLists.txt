cmake_minimum_required(VERSION 3.18)
project(lenet_add_constant_trt86 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 8.6 SDK root")
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "${TENSORRT_ROOT}/include" /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY NAMES nvinfer nvinfer_8
    HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
          /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_library(ONNX_PARSER_LIBRARY NAMES nvonnxparser nvonnxparser_8
    HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
          /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT ONNX_PARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT-8.6.x.x")
endif()
add_library(add_constant_plugin SHARED add_constant_plugin.cu)
target_compile_definitions(add_constant_plugin PRIVATE ADD_CONSTANT_EXPORTS)
target_include_directories(add_constant_plugin PUBLIC "${TENSORRT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(add_constant_plugin PUBLIC "${TENSORRT_LIBRARY}" CUDA::cudart)
add_executable(lenet_build build.cpp)
target_link_libraries(lenet_build PRIVATE add_constant_plugin "${ONNX_PARSER_LIBRARY}")
add_executable(lenet_runtime runtime.cpp)
target_link_libraries(lenet_runtime PRIVATE add_constant_plugin)
