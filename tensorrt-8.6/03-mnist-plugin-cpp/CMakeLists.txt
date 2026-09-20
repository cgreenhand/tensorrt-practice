# tensorrt-8.6/03-mnist-plugin-cpp 子项目构建脚本
# 由最外层 CMakeLists.txt 通过 add_subdirectory 自动包含
# 使用方式（在仓库最外层）：
#   cmake -S . -B build -DTRT_LEARNING_PROJECT=tensorrt-8.6/03-mnist-plugin-cpp
#   cmake --build build -j

# ------------------------------------------------------------------
# 1) 自定义 Plugin（含 CUDA kernel）必须用 nvcc 编译，单独打成静态库
# ------------------------------------------------------------------
add_library(add_constant_plugin STATIC add_constant_plugin.cu)

# Plugin 自身需要看到 NvInfer.h 与同目录下的 add_constant_plugin.h
target_include_directories(
    add_constant_plugin
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Plugin 链接 TensorRT::nvinfer 与 CUDA Runtime；
# 这里走 Learning::TensorRTRuntime（已包含 nvinfer + cudart + OpenCV）
target_link_libraries(
    add_constant_plugin
    PUBLIC
        Learning::CompilerOptions
        Learning::TensorRTRuntime
)

# 静态库不参与 RUNTIME_OUTPUT_DIRECTORY，但要确保归档输出不污染顶层
set_target_properties(
    add_constant_plugin
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SELECTED_PROJECT_NAME}"
        POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------------
# 2) 主可执行文件：lenet_wts_plugin（构建期注入 plugin + 运行期反序列化）
# ------------------------------------------------------------------
add_executable(lenet_wts_plugin lenet_wts_plugin.cpp)

target_link_libraries(
    lenet_wts_plugin
    PRIVATE
        Learning::CompilerOptions
        Learning::TensorRTRuntime
        add_constant_plugin
)

# 让可执行文件输出到最外层的 build 目录，
# 这样在仓库根目录运行 ./build/lenet_wts_plugin 时，
# 相对路径 tensorrt-8.6/01-mnist-wts/artifacts/ 才能正确解析
set_target_properties(
    lenet_wts_plugin
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
