#include "model.h"
#include "block.h"



nvinfer1::IHostMemory* buildEngineYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels)
{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    auto flag = 1ULL << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flag);

    nvinfer1::ITensor* input = network->addInput("image",dt,nvinfer1::Dims4{1,3,640,640});

}

