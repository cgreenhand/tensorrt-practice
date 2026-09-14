cmake_minimum_required(VERSION 3.20)

project(
    TensorRTLearning
    VERSION 1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 可以通过命令行覆盖
set(
    TRT_LEARNING_PROJECT
    "tensorrt-8.6/02-mnist-onnx"
    CACHE STRING
    "Relative path of the learning project to build"
)






set(
    TENSORRT_ROOT
    "/home/gw/TensorRT-8.6.1.6"
    CACHE PATH
    "TensorRT installation directory"
)

set(
    CMAKE_CUDA_ARCHITECTURES
    "86"
    CACHE STRING
    "CUDA GPU architectures"
)

include(cmake/CompilerOptions.cmake)
include(cmake/TensorRT.cmake)
include(cmake/Dependencies.cmake)

set(SELECTED_PROJECT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/${TRT_LEARNING_PROJECT}"
)

# 取子项目目录名（如 01-mnist-wts），用作二进制输出子目录名
get_filename_component(SELECTED_PROJECT_NAME "${TRT_LEARNING_PROJECT}" NAME)

if(NOT EXISTS "${SELECTED_PROJECT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Invalid TRT_LEARNING_PROJECT: ${TRT_LEARNING_PROJECT}\n"
        "CMakeLists.txt not found in: ${SELECTED_PROJECT_DIR}"
    )
endif()

message(STATUS "Selected project: ${TRT_LEARNING_PROJECT}")
message(STATUS "TensorRT root: ${TENSORRT_ROOT}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# 指定单独的二进制目录，避免不同实验互相污染
add_subdirectory(
    "${SELECTED_PROJECT_DIR}"
    "${CMAKE_BINARY_DIR}/${SELECTED_PROJECT_NAME}"
)