#pragma once 
#include <map>
#include <string>
#include <vector>
#include "NvInfer.h"

int calculatePadding(int ksize);

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file_path);

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network, std::map<std::string,nvinfer1::Weights>& weightMap,
                                        nvinfer1::ITensor& input, int ch, int ksize, int stride, int pad, const std::string& lname);

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network,  std::map<std::string,nvinfer1::Weights>& weightMap,
                                nvinfer1::ITensor& input, int ch_in, int ch_out, int n, bool shortcut, float e, const std::string& lname);

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                int ch_in, int ch_out, int ksize, const std::string& lname);


nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network,  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int ch, int grid, int ksize, int stride, int pad, const std::string& lname);

nvinfer1::IPluginV2Layer* addYoloLayer(nvinfer1::INetworkDefinition* network, std::vector<nvinfer1::IConcatenationLayer*> dets, const int* px_arry,
                                        int px_arry_num, int num_class, bool is_segmentation, bool is_post, bool is_obb);
