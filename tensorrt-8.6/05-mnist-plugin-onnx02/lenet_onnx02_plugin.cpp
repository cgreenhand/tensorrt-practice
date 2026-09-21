#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <map>
#include <algorithm>
#include "cuda_runtime_api.h"
#include <cmath>
#include "add_constant_plugin.h"

using namespace nvinfer1;

// tensorrt的日志
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT]" << msg << std::endl;
    }
};
static Logger gLogger;

// argmax
static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

// 2. 创建引擎
ICudaEngine* createEngine(const std::string& onnx_file)
{
    // 2.1 builder
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder);

    // 2.2 network（显式 batch）
    const uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);
    assert(network);

    // 2.3 ONNX 解析器：把 onnx 模型解析填充到 network
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser);

    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "[ONNX] 解析失败: " << onnx_file << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::cerr << "  错误 " << i << ": " << parser->getError(i)->desc() << std::endl;
        parser->destroy();
        network->destroy();
        builder->destroy();
        return nullptr;
    }
    std::cout << "[ONNX] 解析成功: " << onnx_file << std::endl;

    // 2.4 builder 配置
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

    // 2.5 若输入含动态维（onnx 动态 batch），需配置 optimization profile（这里固定 batch=1）
    bool needProfile = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        Dims dims = network->getInput(i)->getDimensions();
        for (int d = 0; d < dims.nbDims; ++d)
            if (dims.d[d] == -1) needProfile = true;   // -1 表示动态维
    }
    if (needProfile) {
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (int i = 0; i < network->getNbInputs(); ++i) {
            ITensor* in = network->getInput(i);
            Dims dims = in->getDimensions();
            for (int d = 0; d < dims.nbDims; ++d)
                if (dims.d[d] == -1) dims.d[d] = 1;     // 把动态 batch 固定为 1
            profile->setDimensions(in->getName(), OptProfileSelector::kMIN, dims);
            profile->setDimensions(in->getName(), OptProfileSelector::kOPT, dims);
            profile->setDimensions(in->getName(), OptProfileSelector::kMAX, dims);
        }
        config->addOptimizationProfile(profile);
    }

    // 2.6 构建引擎
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 2.7 释放中间资源（engine 需保留给调用方）
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return engine;
}


// 3. 加载引擎
ICudaEngine* loadEngine(const std::string& enine_file, IRuntime* runtime)
{
    std::ifstream input(enine_file, std::ios::binary | std::ios::ate);
    assert(input.is_open() && "Could not open engine file");

    size_t size = static_cast<size_t>(input.tellg());
    input.seekg(0,std::ios::beg);
    std::vector<char> buffer(size);
    input.read(buffer.data(),size);
    return runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
}


// 4. 序列化引擎
void serializeEngine(const std::string& engine_file, ICudaEngine*engine)
{
    assert(engine != nullptr);
    std::ofstream output(engine_file, std::ios::binary);
    assert(output.is_open() && "Could not open engine file for writing");
    IHostMemory* serializedEngine = engine->serialize();
    output.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());

    output.close();
    serializedEngine->destroy();
}



// 6. 推理
void doInference(ICudaEngine* engine, const float* input, float* output)
{
    assert(engine != nullptr);
    assert(input != nullptr);
    assert(output != nullptr);

    // 执行上下文（本函数内部自管理）
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // binding 名字来自 onnx 的 input_names / output_names
    const int inputIndex = engine->getBindingIndex("input");
    const int outputIndex = engine->getBindingIndex("output");
    assert(inputIndex >= 0 && outputIndex >= 0 && "binding 名字与 onnx 不一致");

    constexpr size_t inputBytes = 1 * 1 * 32 * 32 * sizeof(float);
    constexpr size_t outputBytes = 1 * 10 * sizeof(float);

    std::vector<void*> bindings(engine->getNbBindings(), nullptr);
    cudaMalloc(&bindings[inputIndex], inputBytes);
    cudaMalloc(&bindings[outputIndex], outputBytes);

    // 输入 H2D -> 推理 -> 输出 D2H，全部挂到同一条 stream 上（异步）
    cudaMemcpyAsync(bindings[inputIndex], input, inputBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bindings.data(), stream, nullptr);
    cudaMemcpyAsync(output, bindings[outputIndex], outputBytes, cudaMemcpyDeviceToHost, stream);

    // 等待整条流完成，保证 output 已写回主机
    cudaStreamSynchronize(stream);

    // 释放资源
    cudaFree(bindings[inputIndex]);
    cudaFree(bindings[outputIndex]);
    cudaStreamDestroy(stream);
    context->destroy();
}

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// 读取裸 float32 二进制
static std::vector<float> readBin(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    assert(f.is_open() && "cannot open bin file");
    std::vector<float> buf(count);
    f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
    assert(static_cast<size_t>(f.gcount()) == count * sizeof(float) && "bin size mismatch");
    return buf;
}


int main()
{
    // 1. 根据onnx创建lenet引擎
    // 2. 将engine保存到本地
    // 3. 加载engine进行推理
    // 4. 对比trt和pytorch的结果

    std::string engine_path = "tensorrt-8.6/05-mnist-plugin-onnx02/artifacts/lenet_add_constant.engine";
    std::string input_path = "tensorrt-8.6/05-mnist-plugin-onnx02/artifacts/input.bin";
    // 注意：Python 端保存的 output_pytorch.bin 已过 softmax，直接是概率
    std::string pytorch_output_path = "tensorrt-8.6/05-mnist-plugin-onnx02/artifacts/output_pytorch.bin";
    std::string onnx_file = "tensorrt-8.6/05-mnist-plugin-onnx02/artifacts/lenet_add_constant.onnx";

    if(!registerAddConstantPlugin()){
        std::cerr << "Failed to register plugin" << std::endl;
        return -1;
    }
    // ===== runtime =====
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime);

    // ===== 1&2&3. engine：有则加载，无则构建并保存 =====
    ICudaEngine* engine = nullptr;
    if (fileExists(engine_path)) {
        std::cout << "[Engine] 发现已保存的 engine，直接加载: " << engine_path << std::endl;
        engine = loadEngine(engine_path, runtime);
    } else {
        std::cout << "[Engine] 未找到 engine，从 onnx 构建: " << onnx_file << std::endl;
        // createEngine 内部已自建 builder/config，这里无需重复创建
        engine = createEngine(onnx_file);
        assert(engine);

        serializeEngine(engine_path, engine);
        std::cout << "[Engine] 已保存到: " << engine_path << std::endl;
    }
    assert(engine);

    // ===== 读取输入并推理（doInference 内部自建 stream/context 并同步） =====
    const size_t outputCount = 10;
    std::vector<float> input = readBin(input_path, 1*1*32*32);
    std::vector<float> output(1*10, 0.0f);
    doInference(engine, input.data(), output.data());

    // ===== 读取 pytorch 基准输出（已含 softmax，直接是概率） =====
    std::vector<float> ptProbs = readBin(pytorch_output_path, 1*10);

    // ===== 打印结果 =====
    std::cout << "\n[TensorRT 输出] (softmax 概率):" << std::endl;
    for (size_t i = 0; i < outputCount; ++i)
        std::cout << "  " << i << ": " << output[i] << std::endl;

    std::cout << "\n[PyTorch 输出] (softmax 概率):" << std::endl;
    for (size_t i = 0; i < outputCount; ++i)
        std::cout << "  " << i << ": " << ptProbs[i] << std::endl;

    // ===== 对比 =====
    float maxAbsDiff = 0.0f;
    for (size_t i = 0; i < outputCount; ++i)
        maxAbsDiff = std::max(maxAbsDiff, std::fabs(output[i] - ptProbs[i]));

    int trtClass = argmax(output);
    int ptClass  = argmax(ptProbs);

    std::cout << "\n[对比] 最大绝对误差 = " << maxAbsDiff << std::endl;
    std::cout << "[对比] TensorRT 预测类别 = " << trtClass
              << "，PyTorch 预测类别 = " << ptClass
              << "，是否一致 = " << (trtClass == ptClass ? "true" : "false") << std::endl;

    // ===== 释放资源 =====
    engine->destroy();
    runtime->destroy();

    return 0;
}