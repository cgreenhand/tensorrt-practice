# TensorRT 8.6：LeNet 输出加任意常数

网络：输入 -> 原 LeNet 全部网络 -> AddConstantPlugin -> output_plus_constant。
每个元素执行 output[i] = input[i] + value。

## 支持范围

沿用原工程：TensorRT 8.6.x、FP32、LINEAR，输入 1×1×32×32，输出 10 个元素。
ONNX 动态维会在 build 时按该形状固定 min/opt/max。插件本身支持动态形状。
value 支持可表示为有限 float32 的正数、负数、0 和小数；拒绝 NaN、Inf、超出范围及尾随非法字符。
过大常数即使自身有限，也可能使计算结果溢出；runtime 会检查非有限输出。

## 编译

需要完整 TensorRT 8.6 开发 SDK、CUDA Toolkit/nvcc、CMake >= 3.18，以及 CUDA 支持的 C++ 编译器。
在本源码目录执行。替换 SDK 路径；86 是 RTX 30 系列 GPU 架构示例，不是 TensorRT 版本。

Linux：

```bash
cmake -S . -B build -DTENSORRT_ROOT=/opt/TensorRT-8.6.1.6 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
export LD_LIBRARY_PATH=/opt/TensorRT-8.6.1.6/lib:$PWD/build:${LD_LIBRARY_PATH:-}
```

Windows：在 MSVC 开发者 PowerShell 中执行：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DTENSORRT_ROOT="D:/SDK/TensorRT-8.6.1.6" -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
$env:PATH = "D:\SDK\TensorRT-8.6.1.6\lib;D:\SDK\TensorRT-8.6.1.6\bin;$env:PATH"
```

CUDA bin 目录也要在 PATH 中。Windows 下，下文的 ./build/lenet_build 与 ./build/lenet_runtime 分别替换为 .\build\Release\lenet_build.exe 与 .\build\Release\lenet_runtime.exe。
部署时保留 libadd_constant_plugin.so / add_constant_plugin.dll，并提供匹配的 CUDA/TensorRT 运行库。
设备支持仍须满足 TensorRT、CUDA、驱动和 JetPack 的对应要求。

## 构建与推理

以加 2.5 为例：

```bash
./build/lenet_build lenet.onnx lenet_add_constant.engine --value 2.5
./build/lenet_runtime lenet_add_constant.engine input.bin output_plus_constant.bin
```

input.bin 为原项目的 1024 个 float32（4096 字节），保持与模型相同的输入预处理。
常数由 build 指定并保存到 engine，runtime 自动恢复，无须重复传值。
省略 --value 时默认加 1。负数示例：

```bash
./build/lenet_build lenet.onnx lenet_minus_three.engine --value -3
```

每次 build 都重新构建 engine。输出目录必须已存在。

## 对照验证

先生成原网络结果：

```bash
./build/lenet_build lenet.onnx lenet_baseline.engine --baseline
./build/lenet_runtime lenet_baseline.engine input.bin baseline_output.bin
```

再在独立进程加载带插件的 engine，并检查 output ≈ baseline + 2.5：

```bash
./build/lenet_runtime lenet_add_constant.engine input.bin output_plus_constant.bin baseline_output.bin 2.5
```

注意最后的 2.5 仅用于验证期望值，绝不会修改 engine 内的常数！改变计算常数需要重新 build。
命令格式：

    lenet_build model.onnx output.engine [--value NUMBER | --baseline]
    lenet_runtime model.engine input.bin output.bin [baseline_output.bin expected_value]

验证逐元素误差阈值：1e-5 + 1e-4 × abs(baseline[i] + expected_value)。
通过返回 0，误差超限返回 2，文件/参数/CUDA/TensorRT 错误返回 1。
可分别用 --value 0、--value -3、--value 2.5 重建并验证；未进行的运行不能视为 PASS。

若使用原 output_pytorch.bin 作为参考，必须确认它与 ONNX 原始最终输出的语义一致：原代码说该文件已过 softmax，若 ONNX 没包含 softmax，就不能直接用这个文件比较。
插件接在原 ONNX 输出之后，不自动增删 softmax。概率加任意非零常数后不再是概率分布。

## 从固定 +1 改成任意常数：关键变化

1. AddConstantPlugin(float value) 将值保存在 mValue。
2. CUDA kernel 接收 value 参数，执行 x[i] + value。
3. clone() 拷贝整个实例，连同 mValue 和 namespace 一起保留。
4. serialize() 按固定顺序写入 uint32_t 格式标识和 float 常数，共 8 字节；不直接保存带填充的结构体。
5. deserializePlugin() 检查长度、格式标识、有限性，读取常数并重建插件。
6. Creator 的 getFieldNames() 声明一个 value / kFLOAT32 / length=1 参数。
7. createPlugin() 校验该参数并复制浮点值，不持有调用方参数指针。
8. build 用 PluginField 将 --value 的值传入 Creator。

build.cpp 中的关键代码：

```cpp
PluginField field{"value", &value, PluginFieldType::kFLOAT32, 1};
PluginFieldCollection fields{1, &field};
plugin.reset(creator->createPlugin("lenet_add_constant", &fields));
```

Creator 参数元数据 mValueField 是成员变量，保证 getFieldNames() 返回的指针有效。
这与 createPlugin 接收的实际参数不同：前者只描述参数，后者携带实际数值。

## 注册与旧版区分

新身份：LeNetAddConstant / version=1 / namespace=lenet_demo。
新共享库：add_constant_plugin。build/runtime 都先调用 registerAddConstantPlugin()。
使用新的插件名称区分原来的 LeNetAddOne，不用旧插件身份解释不同的序列化格式。
旧 lenet_add_one.engine 需要旧插件库；本工程请从 ONNX 重新生成 engine。
engine 保存常数和插件身份，不包含本工程的 CUDA 实现，因此 runtime 必须加载新插件库。

## 文件与验证状态

add_constant_plugin.h/.cu：导出函数、插件和 Creator。
build.cpp：ONNX -> 带常数插件的 engine。
runtime.cpp：加载 engine、推理和可选对照验证。
common.h：日志、错误检查、参数解析和资源管理。
CMakeLists.txt：插件共享库和 build/runtime 可执行程序。

已进行源码与参数传递、克隆、序列化恢复路径的检查。当前环境未发现 CUDA/TensorRT 编译工具，未提供 ONNX 和输入数据，因此没有完成编译或 GPU 数值实测。
