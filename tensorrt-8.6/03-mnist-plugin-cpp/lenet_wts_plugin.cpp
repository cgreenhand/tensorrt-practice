//  1 定义lenet网络结构
//  2 定义输入输出

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <map>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "add_constant_plugin.h"

using namespace  nvinfer1;

// TensorRT 日志记录器（创建 builder/runtime 必需）
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};
static Logger gLogger;

// 判断文件是否存在
static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// 从本地反序列化 engine
static ICudaEngine* loadEngine(const std::string& fileName, IRuntime* runtime) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return nullptr;
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    return runtime->deserializeCudaEngine(data.data(), size);
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

// argmax
static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}


//  加载权重
std::map<std::string, Weights>loadWeights(const std::string&file)
{
    std::cout << "Loading weights from " << file << std::endl;
    std::map<std::string,Weights> weightMap;
    //  打开文件
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file");
    assert(input.good() && "Failed to move to beginning of binary file");

    uint32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight count");

    //  逐个读取权重
    while(count--)
    {
        Weights wt{DataType::kFLOAT,nullptr,0};
        std::string name;
        int64_t size;
        input >> name >> std::dec >> size;
  
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));

        for(int i = 0 ; i< size ; i++)
            input >> std::hex >> val[i];
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;

    }

    return weightMap;
    
}

// 2. 定义lenet的网络
ICudaEngine* buildLenetEngine(IBuilder* builder, IBuilderConfig* config, const std::map<std::string, Weights>& weightMap, const float addValue)
{
    // 1. 创建network
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    INetworkDefinition* network = builder->createNetworkV2(flag);
    // 2. 创建详细的lenet
    // 2.1 input tensor
    ITensor* data = network->addInput("data",DataType::kFLOAT,Dims4{1,1,32,32});
    assert(data);
    // 2.2 conv1
    IConvolutionLayer* conv1  =  network->addConvolutionNd(*data,6,Dims2{5,5},weightMap.at("conv1.weight"),weightMap.at("conv1.bias"));
    assert(conv1);
    conv1->setStride(DimsHW{1,1});

    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0),ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0),PoolingType::kMAX,Dims2{2,2});
    assert(pool1);
    pool1->setStride(DimsHW{2,2});

    // 2.3 conv2
    IConvolutionLayer* conv2  =  network->addConvolutionNd(*pool1->getOutput(0),16,Dims2{5,5},weightMap.at("conv2.weight"),weightMap.at("conv2.bias"));
    assert(conv2);
    conv2->setStride(DimsHW{1,1});

    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0),ActivationType::kRELU);
    assert(relu2);

    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0),PoolingType::kMAX,Dims2{2,2});
    assert(pool2);
    pool2->setStride(DimsHW{2,2});

    // 2.4 fc1
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0),120,weightMap.at("fc1.weight"),weightMap.at("fc1.bias"));
    IActivationLayer* relu3 = network->addActivation(*fc1->getOutput(0),ActivationType::kRELU);
    assert(relu3);

    // 2.5 fc2
    IFullyConnectedLayer* fc2  = network->addFullyConnected(*relu3->getOutput(0),84,weightMap.at("fc2.weight"),weightMap.at("fc2.bias"));
    assert(fc2);

    IActivationLayer* relu4 = network->addActivation(*fc2->getOutput(0),ActivationType::kRELU);

    assert(relu4);

    IFullyConnectedLayer* fc3  = network->addFullyConnected(*relu4->getOutput(0),10,weightMap.at("fc3.weight"),weightMap.at("fc3.bias"));
    assert(fc3);

    ISoftMaxLayer* prob = network->addSoftMax(*fc3->getOutput(0));
    assert(prob);
    // 在softmax后面添加plugin
    IPluginCreator* pluginCreator = getPluginRegistry()->getPluginCreator("LeNetAddConstant", "1","lnet_demo");
    assert(pluginCreator);
    PluginField valueField{"value", &addValue, PluginFieldType::kFLOAT32, 1};
    PluginFieldCollection fieldCollection{1,&valueField};
    IPluginV2* plugin = pluginCreator->createPlugin("add_value", &fieldCollection);
    ITensor* pluginInputs[] = {prob->getOutput(0)};
    IPluginV2Layer* addConstantLayer = network->addPluginV2(pluginInputs, 1, *plugin);
    assert(addConstantLayer);
    addConstantLayer->setName("add_constant_layer");
    ITensor* output = addConstantLayer->getOutput(0);
    output->setName("prob");
    network->markOutput(*output);


    // config（输入为静态 batch=1，无需 optimization profile）
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    plugin ->destroy();
    network->destroy();
    return engine;
}


void saveEngine(const ICudaEngine* engine, const std::string& fileName)
{
    assert(engine);

    IHostMemory* serializedEngine = engine->serialize();
    assert(serializedEngine);

    std::ofstream output(fileName,std::ios::binary);
    assert(output.is_open());

    output.write(
        reinterpret_cast<const char*>(serializedEngine->data()),
        serializedEngine->size()
    );

    output.close();

    serializedEngine->destroy();

}


void doInference(const ICudaEngine* engine,IExecutionContext *context,float* input, float* output,cudaStream_t* stream)
{
    if (context == nullptr ||
        input == nullptr ||
        output == nullptr ||
        stream == nullptr)
    {
        std::cerr << "Invalid inference argument" << std::endl;
        return;
    }

    const int inputIndex = engine->getBindingIndex("data");
    const int outputIndex = engine->getBindingIndex("prob");

    constexpr size_t inputBytes =  1 * 1 * 32 * 32 * sizeof(float);
    const size_t outputBytes = 1 * 10 * sizeof(float);


    std::vector<void*> bindings(
        engine->getNbBindings(),
        nullptr
    );

    cudaError_t error =
        cudaMalloc(&bindings[inputIndex], inputBytes);

    error = cudaMalloc(
        &bindings[outputIndex],
        outputBytes
    );

    // copy input data to device
    error = cudaMemcpyAsync(
        bindings[inputIndex],
        input,
        inputBytes,
        cudaMemcpyHostToDevice,
        *stream
    );

    context->enqueueV2(bindings.data(),*stream,nullptr);
    cudaMemcpyAsync(output,bindings[outputIndex],outputBytes,cudaMemcpyDeviceToHost,*stream);

    cudaStreamSynchronize(*stream);





}


int main(int argc, char** argv)
{
    // 1. 根据wts创建lenet
    // 2. 将engine保存到本地
    // 3. 加载engine进行推理
    // 4. 对比trt和pytorch的结果

    // 注册plugin
    constexpr float kAddValue = 2.5F;
    if(!registerAddConstantPlugin()){
        std::cerr << "Failed to register plugin" << std::endl;
        return 1;

    }

    std::string wts_path = "tensorrt-8.6/03-mnist-plugin-cpp/artifacts/lenet.wts";
    std::string engine_path = "tensorrt-8.6/03-mnist-plugin-cpp/artifacts/lenet_wts_plugin.engine";
    std::string input_path = "tensorrt-8.6/03-mnist-plugin-cpp/artifacts/input.bin";
    // 注意：Python 端保存的 output_pytorch.bin 已过 softmax，直接是概率
    std::string pytorch_output_path = "tensorrt-8.6/03-mnist-plugin-cpp/artifacts/output_pytorch.bin";

    const size_t inputCount = 1 * 1 * 32 * 32;
    const size_t outputCount = 10;

    // ===== runtime =====
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime);

    // ===== 1&2&3. engine：有则加载，无则构建并保存 =====
    ICudaEngine* engine = nullptr;
    if (fileExists(engine_path)) {
        std::cout << "[Engine] 发现已保存的 engine，直接加载: " << engine_path << std::endl;
        engine = loadEngine(engine_path, runtime);
    } else {
        std::cout << "[Engine] 未找到 engine，从 wts 构建: " << wts_path << std::endl;
        IBuilder* builder = createInferBuilder(gLogger);
        assert(builder);
        IBuilderConfig* config = builder->createBuilderConfig();
        assert(config);

        std::map<std::string, Weights> weightMap = loadWeights(wts_path);
        engine = buildLenetEngine(builder, config, weightMap,kAddValue);
        assert(engine);

        saveEngine(engine, engine_path);
        std::cout << "[Engine] 已保存到: " << engine_path << std::endl;

        config->destroy();
        builder->destroy();
    }
    assert(engine);

    // ===== 执行上下文 & stream =====
    IExecutionContext* context = engine->createExecutionContext();
    assert(context);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ===== 读取输入并推理 =====
    std::vector<float> input = readBin(input_path, inputCount);
    std::vector<float> trtOutput(outputCount, 0.0f);

    doInference(engine, context, input.data(), trtOutput.data(), &stream);
    cudaStreamSynchronize(stream);

    // ===== 读取 pytorch 基准输出（已含 softmax，直接是概率） =====
    std::vector<float> ptProbs = readBin(pytorch_output_path, outputCount);

    // ===== 打印结果 =====
    std::cout << "\n[TensorRT 输出] (softmax 概率):" << std::endl;
    for (size_t i = 0; i < outputCount; ++i)
        std::cout << "  " << i << ": " << trtOutput[i] << std::endl;

    std::cout << "\n[PyTorch 输出] (softmax 概率):" << std::endl;
    for (size_t i = 0; i < outputCount; ++i)
        std::cout << "  " << i << ": " << ptProbs[i] << std::endl;

    // ===== 对比 =====
    float maxAbsDiff = 0.0f;
    for (size_t i = 0; i < outputCount; ++i)
        maxAbsDiff = std::max(maxAbsDiff, std::fabs(trtOutput[i] - ptProbs[i]));

    int trtClass = argmax(trtOutput);
    int ptClass  = argmax(ptProbs);

    std::cout << "\n[对比] 最大绝对误差 = " << maxAbsDiff << std::endl;
    std::cout << "[对比] TensorRT 预测类别 = " << trtClass
              << "，PyTorch 预测类别 = " << ptClass
              << "，是否一致 = " << (trtClass == ptClass ? "true" : "false") << std::endl;

    // ===== 释放资源 =====
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

