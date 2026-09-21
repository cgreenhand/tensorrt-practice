# tensorrt-8.6/04-mnist-plugin-onnx01 子项目构建脚本
# 由最外层 CMakeLists.txt 通过 add_subdirectory 自动包含
# 使用方式（在仓库最外层）：
#   cmake -S . -B build -DTRT_LEARNING_PROJECT=tensorrt-8.6/04-mnist-plugin-onnx01
#   cmake --build build -j

# ------------------------------------------------------------------
# 1) 自定义 Plugin（含 CUDA kernel）必须用 nvcc 编译，单独打成静态库
#    注意：__global__ kernel 不能放在匿名命名空间内，否则 nvcc 在 STATIC
#    库模式下生成 cudafe1.stub.c 时会报 "_GLOBAL__N_xxx is ambiguous"
# ------------------------------------------------------------------
add_library(add_constant_plugin STATIC add_constant_plugin.cu)

# Plugin 自身需要看到 NvInfer.h 与同目录下的 add_constant_plugin.h
target_include_directories(
    add_constant_plugin
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Plugin 链接 TensorRT::nvinfer + CUDA Runtime
target_link_libraries(
    add_constant_plugin
    PUBLIC
        Learning::CompilerOptions
        Learning::TensorRTRuntime
)

# CUDA 源专属的警告抑制：
#   -Wno-pedantic        ：避免 nvcc 生成的 cudafe1.cpp 行号指令在 -Wpedantic 下刷屏
#   --diag_suppress=997  ：IPluginV2DynamicExt 与基类 IPluginV2 同名不同签名的虚函数
#                          被派生类 override 后会触发 name hiding 警告，安全忽略
target_compile_options(
    add_constant_plugin
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Wno-pedantic
            -Xcudafe --diag_suppress=997
        >
)

set_target_properties(
    add_constant_plugin
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${SELECTED_PROJECT_NAME}"
        POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------------
# 2) 主可执行文件：lenet_onnx01_plugin
#    既要 ONNX Parser 解析模型，又要在解析后注入自定义 plugin
# ------------------------------------------------------------------
add_executable(lenet_onnx01_plugin lenet_onnx01_plugin.cpp)

target_link_libraries(
    lenet_onnx01_plugin
    PRIVATE
        Learning::CompilerOptions
        Learning::TensorRTOnnx        # 包含 nvinfer + nvonnxparser + cudart + OpenCV
        add_constant_plugin
)

# 让可执行文件输出到最外层的 build 目录，
# 这样在仓库根目录运行 ./build/lenet_onnx01_plugin 时，
# 相对路径 tensorrt-8.6/04-mnist-plugin-onnx01/artifacts/ 才能正确解析
set_target_properties(
    lenet_onnx01_plugin
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
