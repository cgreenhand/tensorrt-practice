# tensorrt-8.6/01-mnist-wts 子项目构建脚本
# 由最外层 CMakeLists.txt 通过 add_subdirectory 自动包含
# 使用方式（在仓库最外层）：
#   cmake -S . -B build
#   cmake --build build -j

# 可执行目标：把 lenet.cpp 编译为 lenet
add_executable(lenet_onnx lenet_onnx.cpp)

# 链接：
#   Learning::CompilerOptions  -> C++17 + 警告选项
#   Learning::TensorRTRuntime  -> TensorRT::nvinfer + CUDA::cudart + OpenCV
target_link_libraries(
    lenet_onnx
    PRIVATE
        Learning::CompilerOptions
        # Learning::TensorRTRuntime
        Learning::TensorRTOnnx
)

# 让可执行文件输出到最外层的 build 目录（CMAKE_BINARY_DIR 即 -B 指定的顶层构建目录），
# 这样在仓库根目录运行 ./build/lenet 时，相对路径 tensorrt-8.6/01-mnist-wts/artifacts/ 能正确读取
set_target_properties(
    lenet_onnx
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
