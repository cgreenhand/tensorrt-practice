#include "block.h"
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>

// 奇数核 SAME padding stride = 1
int calculatePadding(int ksize){
    return (ksize-1) / 2;
}

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file_path)
{
    // Load weights from file
    std::map<std::string, nvinfer1::Weights> weightMap;
    std::ifstream input(file_path);
    assert(input.is_open() && "Could not open file");

    int32_t count = 0;
    input >> count;
    assert(count > 0 && "Invalid number of weights");

    while(count --){
        std::string name{};
        uint32_t number_size = 0;
        input >> name >> std::dec >> number_size;
        assert(number_size > 0 && "Invalid number of weight elements");
        nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, number_size};

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * number_size));
        for(uint32_t i = 0; i < number_size; i++){
            input >> std::hex >> val[i];
        }
        weight.values = val;
        weightMap[name] = weight;

    }
    return weightMap;

}

// convBnSiLU
// conv + bn + SiLU
static nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                            std::map<std::string, nvinfer1::Weights>& weightMap,
                                            nvinfer1::ITensor& input, const std::string& lname, float eps)
{
    
    const float* gamma = reinterpret_cast<const float*>(weightMap.at(lname+".weight").values);
    const float* beta = reinterpret_cast<const float*>(weightMap.at(lname+".bias").values);
    const float* mean = reinterpret_cast<const float*>(weightMap.at(lname+".running_mean").values);
    const float* var = reinterpret_cast<const float*>(weightMap.at(lname+".running_var").values);
    int len = weightMap.at(lname+".weight").count;

    // scale
    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0 ; i< len ; i++){
        scale_val[i] = gamma[i] / sqrt(var[i] + eps);
    }

    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scale_val, len};

    // shift

    float* shift_val = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0 ; i< len ; i++){
        shift_val[i] = beta[i] - mean[i] * scale_val[i];
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shift_val, len};

    float* power_val = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0 ; i< len ; i++){
        power_val[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, power_val, len};

    weightMap.emplace(lname+".scale", scale);
    weightMap.emplace(lname+".shift", shift);
    weightMap.emplace(lname+".power", power);

    // 显示修改 1
    return network->addScaleNd(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power,1);
}


nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network, std::map<std::string,nvinfer1::Weights>& weightMap,
                                        nvinfer1::ITensor& input, int ch, int ksize, int stride, int pad, const std::string& lname)
{
    // conv
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT,nullptr,0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input,ch,nvinfer1::DimsHW(ksize,ksize),weightMap.at(lname+".conv.weight"),bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW(stride,stride));
    conv->setPaddingNd(nvinfer1::DimsHW(pad,pad));
    // bn
    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network,weightMap,*conv->getOutput(0),lname+".bn",1e-3);

    // silu
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* silu = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    return silu;
}


static nvinfer1::ILayer* bottleNect(nvinfer1::INetworkDefinition* network, std::map<std::string,nvinfer1::Weights>& weightMap,
                                    nvinfer1::ITensor& input, int ch_in, bool shortcut, const std::string& lname)
{

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network,weightMap,input,ch_in,3,1,1,lname+".cv1");
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network,weightMap,*conv1->getOutput(0),ch_in,3,1,1,lname+".cv2");
    if (shortcut){
        nvinfer1::IElementWiseLayer* add = network->addElementWise(input,*conv2->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
        return add;
    }
    return conv2;

}

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network,  std::map<std::string,nvinfer1::Weights>& weightMap,
                                nvinfer1::ITensor& input,  int ch_out, int n , bool shortcut, float e, const std::string& lname)
{
    auto hiddenChannels = static_cast<int>(ch_out * e);


    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network,weightMap,input,2*hiddenChannels,1,1,0,lname+".cv1");
    nvinfer1::Dims d = conv1->getOutput(0)->getDimensions();
    // split CHW -》 0-C/2   C/2 - C
    nvinfer1::ISliceLayer* split1 = network->addSlice(*conv1->getOutput(0),nvinfer1::Dims4(0,0,0,0),nvinfer1::Dims4(d.d[0],hiddenChannels,d.d[2],d.d[3]),nvinfer1::Dims4(1,1,1,1));
    nvinfer1::ISliceLayer* split2 = network->addSlice(*conv1->getOutput(0),nvinfer1::Dims4(0,hiddenChannels,0,0),nvinfer1::Dims4(d.d[0],hiddenChannels,d.d[2],d.d[3]),nvinfer1::Dims4(1,1,1,1));



    std::vector<nvinfer1::ITensor*> inputTensor0 = {split1->getOutput(0),split2->getOutput(0)};

    nvinfer1::ITensor* y1 = split2->getOutput(0);
    for(int i = 0 ; i < n; i++){
        auto* bottleneck = bottleNect(network,weightMap,*y1,hiddenChannels,shortcut,lname+".m."+std::to_string(i));
        y1 = bottleneck->getOutput(0);

        
        inputTensor0.emplace_back(y1);
    }
    nvinfer1::IConcatenationLayer* concat = network->addConcatenation(inputTensor0.data(), inputTensor0.size());
    concat->setAxis(1);
    nvinfer1::IElementWiseLayer* conv2 = convBnSiLU(network,weightMap,*concat->getOutput(0),ch_out,1,1,0,lname+".cv2");
    return conv2;

}


nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                int ch_in, int ch_out, int ksize, const std::string& lname)
{

    int hidden_channels = ch_in / 2;
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network,weightMap,input,hidden_channels,1,1,0,lname+".cv1");
    nvinfer1::IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0),nvinfer1::PoolingType::kMAX,nvinfer1::DimsHW(ksize,ksize));
    pool1->setStrideNd(nvinfer1::DimsHW(1,1));
    pool1->setPaddingNd(nvinfer1::DimsHW(ksize/2,ksize/2));

    nvinfer1::IPoolingLayer* pool2 = network->addPoolingNd(*pool1->getOutput(0),nvinfer1::PoolingType::kMAX,nvinfer1::DimsHW(ksize,ksize));
    pool2->setStrideNd(nvinfer1::DimsHW(1,1));
    pool2->setPaddingNd(nvinfer1::DimsHW(ksize/2,ksize/2));

    nvinfer1::IPoolingLayer* pool3 = network->addPoolingNd(*pool2->getOutput(0),nvinfer1::PoolingType::kMAX,nvinfer1::DimsHW(ksize,ksize));
    pool3->setStrideNd(nvinfer1::DimsHW(1,1));
    pool3->setPaddingNd(nvinfer1::DimsHW(ksize/2,ksize/2));

    nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0),
                                         pool3->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 4);
    cat->setAxis(1);
    nvinfer1::IElementWiseLayer* conv2 =
            convBnSiLU(network, weightMap, *cat->getOutput(0), ch_out,1,1,0,lname+".cv2");
    return conv2;

}

// input  64,grid
// 显示改造
// n,64,grid
nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network,  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, const std::string& lname)
{
    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(input);
    constexpr int kRegMax = 16;
    shuffle1->setZeroIsPlaceholder(true);
    shuffle1->setReshapeDimensions(nvinfer1::Dims4(0,4,kRegMax,-1));
    shuffle1->setSecondTranspose(nvinfer1::Permutation{0,2,1,3});

    // n,16,4,grid
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuffle1->getOutput(0));
    softmax->setAxes(1U << 1U);
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT,nullptr,0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*softmax->getOutput(0),1,nvinfer1::DimsHW(1,1),weightMap.at(lname),bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW(1,1));
    conv->setPaddingNd(nvinfer1::DimsHW(0,0));
    // n,1,4,grid

    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setZeroIsPlaceholder(true);
    shuffle2->setReshapeDimensions(nvinfer1::Dims3(0,4,-1));

    // n 4 8400
    return shuffle2;
}   
