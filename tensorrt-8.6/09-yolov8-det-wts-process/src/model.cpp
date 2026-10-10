#include "model.h"
#include "block.h"
#include <cmath>
#include <algorithm>
#include "config.h"
#include "pluginLayer.h"


using namespace yolov8;

// static int get_width(int x, float gw, int max_channelds, int divisor = 8)
// {

//     auto channels = int(ceil((x * gw) / divisor)) * divisor;
//     return std::min(channels, max_channelds);

// }

static int get_width(int x, float gw, int max_channelds, int divisor = 8)
{

    auto channels = static_cast<float>(std::min(x,max_channelds)*gw);

    channels = std::ceil(channels / divisor) * divisor;

    return channels;

}

static int get_depth(int x, float gd)
{
    if(x == 1)
        return 1;
    int r = round(x * gd);
    if(x*gd - int(x*gd) == 0.5 && (int(x*gd)%2) == 0)
        --r;
    return std::max<int>(r, 1);
}




nvinfer1::IHostMemory* buildEngineYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels)
{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    auto flag = 1ULL << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flag);

    nvinfer1::ITensor* input = network->addInput("images",dt,nvinfer1::Dims4{det::kBatchSize,3,640,640});

    nvinfer1::IElementWiseLayer* conv0 = 
                        convBnSiLU(network,weightMap,*input,get_width(64,gw,max_channels),3,2,1,"model.0");

    nvinfer1::IElementWiseLayer* conv1 = 
                        convBnSiLU(network,weightMap,*conv0->getOutput(0),get_width(128,gw,max_channels),3,2,1,"model.1");
        
    nvinfer1::IElementWiseLayer* conv2 = 
                        C2F(network,weightMap,*conv1->getOutput(0),get_width(128,gw,max_channels),get_depth(3,gd),true,0.5,"model.2");

    nvinfer1::IElementWiseLayer* conv3 = 
                        convBnSiLU(network,weightMap,*conv2->getOutput(0),get_width(256,gw,max_channels),3,2,1,"model.3");
        
    nvinfer1::IElementWiseLayer* conv4 = 
                        C2F(network,weightMap,*conv3->getOutput(0),get_width(256,gw,max_channels),get_depth(6,gd),true,0.5,"model.4");

    nvinfer1::IElementWiseLayer* conv5 = 
                        convBnSiLU(network,weightMap,*conv4->getOutput(0),get_width(512,gw,max_channels),3,2,1,"model.5");
        
    nvinfer1::IElementWiseLayer* conv6 = 
                        C2F(network,weightMap,*conv5->getOutput(0),get_width(512,gw,max_channels),get_depth(6,gd),true,0.5,"model.6");

    nvinfer1::IElementWiseLayer* conv7 = 
                        convBnSiLU(network,weightMap,*conv6->getOutput(0),get_width(1024,gw,max_channels),3,2,1,"model.7");
        
    nvinfer1::IElementWiseLayer* conv8 = 
                        C2F(network,weightMap,*conv7->getOutput(0),get_width(1024,gw,max_channels),get_depth(3,gd),true,0.5,"model.8");

    nvinfer1::IElementWiseLayer* conv9 = 
                        SPPF(network,weightMap,*conv8->getOutput(0),get_width(1024,gw,max_channels),get_width(1024,gw,max_channels),5,"model.9");   


    // head
    float scale[] = {1.0,1.0,2.0,2.0};

    nvinfer1::IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale,4);
    nvinfer1::ITensor* cat11_input[] = {upsample10->getOutput(0),conv6->getOutput(0)}; 
    nvinfer1::IConcatenationLayer* concat11 = network->addConcatenation(cat11_input,2);
    concat11->setAxis(1);
    nvinfer1::IElementWiseLayer* conv12 = 
                        C2F(network,weightMap,*concat11->getOutput(0),get_width(512,gw,max_channels),get_depth(3,gd),false,0.5,"model.12");

    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale,4);
    nvinfer1::ITensor* cat14_input[] = {upsample13->getOutput(0),conv4->getOutput(0)}; 
    nvinfer1::IConcatenationLayer* concat14 = network->addConcatenation(cat14_input,2);
    concat14->setAxis(1);
    nvinfer1::IElementWiseLayer* conv15 = 
                        C2F(network,weightMap,*concat14->getOutput(0),get_width(256,gw,max_channels),get_depth(3,gd),false,0.5,"model.15");

    
    nvinfer1::IElementWiseLayer* conv16 = 
                        convBnSiLU(network,weightMap,*conv15->getOutput(0),get_width(256,gw,max_channels),3,2,1,"model.16");

    nvinfer1::ITensor* conv17_input[] = {conv16->getOutput(0),conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat17 = network->addConcatenation(conv17_input,2);
    concat17->setAxis(1);
    nvinfer1::IElementWiseLayer* conv18 = 
                        C2F(network,weightMap,*concat17->getOutput(0),get_width(512,gw,max_channels),get_depth(3,gd),false,0.5,"model.18");


    nvinfer1::IElementWiseLayer* conv19 = 
                        convBnSiLU(network,weightMap,*conv18->getOutput(0),get_width(512,gw,max_channels),3,2,1,"model.19");

    nvinfer1::ITensor* conv20_input[] = {conv19->getOutput(0),conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat20 = network->addConcatenation(conv20_input,2);
    concat20->setAxis(1);
    nvinfer1::IElementWiseLayer* conv21 = 
                        C2F(network,weightMap,*concat20->getOutput(0),get_width(1024,gw,max_channels),get_depth(3,gd),false,0.5,"model.21");

    // detect 检测头  p3-conv15 p4-conv18 p5-conv21
   

    constexpr int kRegMax = 16;
    const nvinfer1::Dims p3Dims = conv15->getOutput(0)->getDimensions();
    const int ch0 = p3Dims.d[1];

    const int regHiddenChannels = std::max({16,ch0/4,4*kRegMax});
    const int clsHiddenChannles = std::max(ch0,std::min(det::kNumClasses,100)); 

    //output1 p3-conv15
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 = 
                        convBnSiLU(network,weightMap,*conv15->getOutput(0),regHiddenChannels,3,1,1,"model.22.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 = 
                        convBnSiLU(network,weightMap,*conv22_cv2_0_0->getOutput(0),regHiddenChannels,3,1,1,"model.22.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 =
                        network->addConvolution(*conv22_cv2_0_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap.at("model.22.cv2.0.2.weight"),weightMap.at("model.22.cv2.0.2.bias"));
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 = 
                        convBnSiLU(network,weightMap,*conv15->getOutput(0),clsHiddenChannles,3,1,1,"model.22.cv3.0.0");                                    
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = 
                        convBnSiLU(network,weightMap,*conv22_cv3_0_0->getOutput(0),clsHiddenChannles,3,1,1,"model.22.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 = 
                        network->addConvolution(*conv22_cv3_0_1->getOutput(0),det::kNumClasses,nvinfer1::DimsHW{1,1},weightMap.at("model.22.cv3.0.2.weight"),weightMap.at("model.22.cv3.0.2.bias"));
    conv22_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0,0});    

    //output2 p4-conv18
    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 = 
                        convBnSiLU(network,weightMap,*conv18->getOutput(0),regHiddenChannels,3,1,1,"model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1 = 
                        convBnSiLU(network,weightMap,*conv22_cv2_1_0->getOutput(0),regHiddenChannels,3,1,1,"model.22.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 =
                        network->addConvolution(*conv22_cv2_1_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap.at("model.22.cv2.1.2.weight"),weightMap.at("model.22.cv2.1.2.bias"));
    conv22_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 = 
                        convBnSiLU(network,weightMap,*conv18->getOutput(0),clsHiddenChannles,3,1,1,"model.22.cv3.1.0");                                    
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1 = 
                        convBnSiLU(network,weightMap,*conv22_cv3_1_0->getOutput(0),clsHiddenChannles,3,1,1,"model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 = 
                        network->addConvolution(*conv22_cv3_1_1->getOutput(0),det::kNumClasses,nvinfer1::DimsHW{1,1},weightMap.at("model.22.cv3.1.2.weight"),weightMap.at("model.22.cv3.1.2.bias"));
    conv22_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0,0});    

    //output1 p5-conv21
    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 = 
                        convBnSiLU(network,weightMap,*conv21->getOutput(0),regHiddenChannels,3,1,1,"model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1 = 
                        convBnSiLU(network,weightMap,*conv22_cv2_2_0->getOutput(0),regHiddenChannels,3,1,1,"model.22.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 =
                        network->addConvolution(*conv22_cv2_2_1->getOutput(0),64,nvinfer1::DimsHW{1,1},weightMap.at("model.22.cv2.2.2.weight"),weightMap.at("model.22.cv2.2.2.bias"));
    conv22_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0,0});

    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 = 
                        convBnSiLU(network,weightMap,*conv21->getOutput(0),clsHiddenChannles,3,1,1,"model.22.cv3.2.0");                                    
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1 = 
                        convBnSiLU(network,weightMap,*conv22_cv3_2_0->getOutput(0),clsHiddenChannles,3,1,1,"model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 = 
                        network->addConvolution(*conv22_cv3_2_1->getOutput(0),det::kNumClasses,nvinfer1::DimsHW{1,1},weightMap.at("model.22.cv3.2.2.weight"),weightMap.at("model.22.cv3.2.2.bias"));
    conv22_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1,1});
    conv22_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0,0});    
            

    // 解析输出
    // p3  box(batch,64,80,80) + cls(batch,kNumberClasses,80,80)
    // p4  box(batch,64,40,40) + cls(batch,kNumberClasses,40,40)
    // p5  box(batch,64,20,20) + cls(batch,kNumberClasses,20,20)

    nvinfer1::IShuffleLayer* shuffer23_p3_reg = network->addShuffle(*conv22_cv2_0_2->getOutput(0));
    shuffer23_p3_reg->setReshapeDimensions(nvinfer1::Dims3{0,0,-1});
    shuffer23_p3_reg->setZeroIsPlaceholder(true);
    nvinfer1::IShuffleLayer* shuffer23_p3_cls = network->addShuffle(*conv22_cv3_0_2->getOutput(0));
    shuffer23_p3_cls->setReshapeDimensions(nvinfer1::Dims3{0,0,-1});
    shuffer23_p3_cls->setZeroIsPlaceholder(true);

    nvinfer1::IShuffleLayer* shuffer23_p4_reg = network->addShuffle(*conv22_cv2_1_2->getOutput(0));
    shuffer23_p4_reg->setReshapeDimensions(nvinfer1::Dims3{0,0,-1});
    shuffer23_p4_reg->setZeroIsPlaceholder(true);
    nvinfer1::IShuffleLayer* shuffer23_p4_cls = network->addShuffle(*conv22_cv3_1_2->getOutput(0));
    shuffer23_p4_cls->setReshapeDimensions(nvinfer1::Dims3{0,0,-1});
    shuffer23_p4_cls->setZeroIsPlaceholder(true);

    nvinfer1::IShuffleLayer* shuffer23_p5_reg = network->addShuffle(*conv22_cv2_2_2->getOutput(0));
    shuffer23_p5_reg->setReshapeDimensions(nvinfer1::Dims3{0,0,-1});
    shuffer23_p5_reg->setZeroIsPlaceholder(true);
    nvinfer1::IShuffleLayer* shuffer23_p5_cls = network->addShuffle(*conv22_cv3_2_2->getOutput(0));
    shuffer23_p5_cls->setReshapeDimensions(nvinfer1::Dims3{0,0,-1});
    shuffer23_p5_cls->setZeroIsPlaceholder(true);

    nvinfer1::ITensor* concat23_reg_input[] = {shuffer23_p3_reg->getOutput(0),shuffer23_p4_reg->getOutput(0),shuffer23_p5_reg->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat23_reg = network->addConcatenation(concat23_reg_input,3);
    concat23_reg->setAxis(2);

    nvinfer1::ITensor* concat23_cls_input[] = {shuffer23_p3_cls->getOutput(0),shuffer23_p4_cls->getOutput(0),shuffer23_p5_cls->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat23_cls = network->addConcatenation(concat23_cls_input,3);
    concat23_cls->setAxis(2);    
    // dfl (n,4,8400)
    nvinfer1::IShuffleLayer* dfl23 = DFL(network,weightMap,*concat23_reg->getOutput(0),"model.22.dfl.conv.weight");
    // score (n,80,8400)
    nvinfer1::IActivationLayer* sigmoid23 = network->addActivation(*concat23_cls->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // insert plugin 
    auto* pluginCreator = ::getPluginRegistry()->getPluginCreator(
        "Yolov8Decode_Plugin",
        "1",
        ""
    );

    int32_t numClasses = det::kNumClasses;
    int32_t inputWidth = det::kInputWidth;
    int32_t inputHeight = det::kInputHeight;
    float scoreThreshold = det::kConfidenceThreshold;

    int32_t maxOutputBoxes = (inputHeight / 8) * (inputWidth / 8)
                            + (inputHeight / 16) * (inputWidth / 16)
                            + (inputHeight / 32) * (inputWidth / 32);
    
    int32_t strides[] = {8,16,32};

    nvinfer1::PluginField fields[] = {
        {
            "num_classes",
            &numClasses,
            nvinfer1::PluginFieldType::kINT32,
            1
        },
        {
            "input_width",
            &inputWidth,
            nvinfer1::PluginFieldType::kINT32,
            1
        },
        {
            "input_height",
            &inputHeight,
            nvinfer1::PluginFieldType::kINT32,
            1
        },
        {
            "max_output_boxes",
            &maxOutputBoxes,
            nvinfer1::PluginFieldType::kINT32,
            1
        },
        {
            "score_threshold",
            &scoreThreshold,
            nvinfer1::PluginFieldType::kFLOAT32,
            1
        },
        {
            "strides",
            strides,
            nvinfer1::PluginFieldType::kINT32,
            3
        }  
    };

    nvinfer1::PluginFieldCollection fieldCollection{
        static_cast<int32_t>(
            sizeof(fields) / sizeof(fields[0])
        ),
        fields
    };

    nvinfer1::IPluginV2* decodePlugin = 
        pluginCreator->createPlugin("yolov8_decode", &fieldCollection);

    if(decodePlugin == nullptr){
        throw std::runtime_error("Failed to create plugin");
    };

    nvinfer1::ITensor* pluginInputs[] = {
        dfl23->getOutput(0),
        sigmoid23 -> getOutput(0)
    };

    nvinfer1::IPluginV2Layer* decodeLayer = network->addPluginV2(pluginInputs,2,*decodePlugin);

    if (decodeLayer == nullptr)
    {
        decodePlugin->destroy();

        throw std::runtime_error(
            "Failed to add YoloV8Decode_TRT plugin layer"
        );
    }

    decodeLayer->setName("yolov8_decode_layer");
    
    nvinfer1::ITensor* output = decodeLayer->getOutput(0);

    nvinfer1::ITensor* outputCount =decodeLayer->getOutput(1);

    output->setName(det::kOutputTensorName);
    outputCount->setName("output_count");

    network->markOutput(*output);
    network->markOutput(*outputCount);

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network,*config);

    decodePlugin->destroy();
    network->destroy();

    if (serializedEngine == nullptr)
    {
        throw std::runtime_error(
            "Failed to build serialized YOLOv8 engine"
        );
    }

    return serializedEngine;

}  

