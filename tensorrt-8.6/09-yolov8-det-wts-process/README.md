# YOLOv8 TensorRT 8.6 检测部署

## 项目简介

本项目是 YOLOv8 目标检测模型的 TensorRT 8.6 部署实践。模型通过 TensorRT C++ API 从 `.wts` 权重构建，使用自定义 YOLOv8 Decode Plugin 完成检测头解码，并通过 CUDA 实现 GPU 图像预处理和 GPU 后处理（排序、NMS、坐标还原）。

## 项目结构

```text
09-yolov8-det-wts-process/
├── CMakeLists.txt              # 子项目构建配置，生成 yolov8_det
├── main.cpp                    # 主程序：构建 Engine、批量推理和结果保存
├── gen_wts.py                  # 将 PyTorch .pt 权重导出为 .wts
├── yolov8.yaml                 # YOLOv8 模型结构配置
├── best.pt                     # 示例训练权重
├── yolov8s.pt                  # 示例 YOLOv8s 权重
├── include/
│   ├── block.h                 # TensorRT 网络基础模块声明
│   ├── config.h                # 精度、输入尺寸、阈值等全局配置
│   ├── cuda_utils.h            # CUDA 错误检查工具
│   ├── model.h                 # YOLOv8 Engine 构建接口
│   ├── preprocess.h            # GPU 预处理接口
│   └── postprocess.h           # GPU 后处理与 NMS 接口
├── src/
│   ├── block.cpp               # 权重加载及 Conv、C2F、SPPF、DFL 等模块实现
│   ├── model.cpp               # YOLOv8 TensorRT 网络定义与插件接入
│   ├── preprocess.cu           # CUDA resize、letterbox、归一化等预处理
│   └── postprocess.cu          # CUDA 排序、NMS 和坐标还原
├── plugin/
│   ├── pluginLayer.h           # YOLOv8 Decode Plugin 声明
│   └── pluginLayer.cu          # Decode Plugin CUDA 实现与注册
├── yolov8_det.cpp              # 早期推理实现参考，不参与当前目标编译
└── 3.cpp                       # 插件接入代码参考，不参与当前目标编译
```

## 环境要求

- TensorRT 8.6.x
- CUDA 11.x+
- OpenCV 4.x
- CMake 3.18+（当前顶层 `CMakeLists.txt` 声明最低版本为 3.20，建议使用 3.20+）
- 支持 C++17 的编译器
- PyTorch + ultralytics（仅用于从 `.pt` 导出 `.wts` 权重）

## 编译步骤

从项目根目录执行：

```bash
cd /home/gw/workspace/UP/tensorrt_learning/tensorrt-practice
mkdir -p build && cd build
cmake .. -DTRT_LEARNING_PROJECT=tensorrt-8.6/09-yolov8-det-wts-process \
         -DTENSORRT_ROOT=/home/gw/TensorRT-8.6.1.6 \
         -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)
```

请根据本机 TensorRT 安装位置和 GPU Compute Capability 修改 `TENSORRT_ROOT` 与 `CMAKE_CUDA_ARCHITECTURES`。

编译产物位于：

```text
/home/gw/workspace/UP/tensorrt_learning/tensorrt-practice/build/09-yolov8-det-wts-process/yolov8_det
```

## 使用方法

以下命令假定项目根目录为 `/home/gw/workspace/UP/tensorrt_learning/tensorrt-practice`。

### Step 1：导出权重

安装导出依赖后，将 `.pt` 权重转换为 `.wts`：

```bash
cd /home/gw/workspace/UP/tensorrt_learning/tensorrt-practice/tensorrt-8.6/09-yolov8-det-wts-process
python3 gen_wts.py -w yolov8s.pt -o yolov8s.wts -t detect
```

参数说明：

- `-w, --weights`：输入 `.pt` 文件，必填。
- `-o, --output`：输出 `.wts` 文件，可选；省略时输出到输入权重同目录并使用同名文件。
- `-t, --type`：模型任务类型，本项目使用 `detect`。

### Step 2：构建 Engine

进入可执行文件目录：

```bash
cd /home/gw/workspace/UP/tensorrt_learning/tensorrt-practice/build/09-yolov8-det-wts-process
```

命令格式：

```bash
./yolov8_det -s [n/s/m/l/x] [wts文件路径]
```

例如构建 YOLOv8s：

```bash
./yolov8_det -s s /home/gw/workspace/UP/tensorrt_learning/tensorrt-practice/tensorrt-8.6/09-yolov8-det-wts-process/yolov8s.wts
```

第一个参数选择模型变体：`n`、`s`、`m`、`l` 或 `x`，必须与导出的权重匹配。Engine 会保存在 `.wts` 文件所在目录，并将扩展名替换为 `.engine`，例如 `yolov8s.engine`。

### Step 3：批量推理

命令格式：

```bash
./yolov8_det -d [engine文件路径] [图片目录路径]
```

示例：

```bash
./yolov8_det -d /home/gw/workspace/UP/tensorrt_learning/tensorrt-practice/tensorrt-8.6/09-yolov8-det-wts-process/yolov8s.engine /path/to/images
```

程序按文件名排序批量处理图片，并在图片目录的同级 `output` 目录中保存带检测框的同名结果。例如输入目录为 `/path/to/images`，结果保存到 `/path/to/output`。

## 模型变体

| 变体 | depth（gd） | width（gw） | max_channels |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1024 |
| s | 0.33 | 0.50 | 1024 |
| m | 0.67 | 0.75 | 768 |
| l | 1.00 | 1.00 | 512 |
| x | 1.00 | 1.25 | 512 |

## 配置参数

关键参数位于 `include/config.h`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `kPrecision` | `Precision::FP16` | TensorRT 推理精度，可选 FP32、FP16、INT8 |
| `kNumClasses` | `80` | 检测类别数 |
| `kBatchSize` | `1` | 固定批量大小 |
| `kInputChannels` | `3` | 输入通道数 |
| `kInputHeight` / `kInputWidth` | `640` / `640` | 网络输入尺寸 |
| `kMaxOutputBoxes` | `1000` | NMS 前 Top-K 和每张图最大检测框数 |
| `kConfidenceThreshold` | `0.5` | 置信度阈值 |
| `kNmsThreshold` | `0.45` | NMS IoU 阈值 |
| `kGpuId` | `0` | 使用的 CUDA 设备编号 |
| `kWorkspaceBytes` | `4 GiB` | TensorRT Builder Workspace 上限 |
| `kCalibrationDirectory` | `./coco_calib` | INT8 校准图片目录配置 |

直接修改 `include/config.h` 中对应常量。修改后需要重新编译；输入尺寸、批量大小、类别数、精度或置信度等影响 Engine 的参数变更后，还需重新构建 Engine。

## 注意事项

- TensorRT Engine 与 TensorRT 版本、GPU 架构及构建环境相关，更换 GPU 或 CUDA/TensorRT 环境后应重新构建。
- 默认推理精度为 FP16；目标 GPU 应支持高效 FP16 运算。
- 默认批量大小为 1。修改 `kBatchSize` 后必须重新编译并重新构建 Engine。
- 输入目录支持 `.jpg`、`.jpeg`、`.png`、`.bmp`，扩展名大小写均可。
- 模型变体参数必须与 `.wts` 权重匹配，否则网络结构与权重可能不兼容。
