#include "common.h"
#include "add_constant_plugin.h"
#include <NvOnnxParser.h>

using namespace nvinfer1;
struct PluginDeleter {
    void operator()(IPluginV2* p) const { if (p) p->destroy(); }
};

int main(int argc, char** argv) {
    bool const baseline = argc == 4 && std::string(argv[3]) == "--baseline";
    bool const withValue = argc == 5 && std::string(argv[3]) == "--value";
    if (argc != 3 && !baseline && !withValue) {
        std::cerr << "Usage: lenet_build model.onnx output.engine [--value NUMBER | --baseline]\n"
                  << "Default value: 1. The constant is saved inside the engine.\n";
        return 1;
    }
    try {
        float const value = withValue ? parseConstant(argv[4]) : 1.0F;
        Logger logger;
        require(registerAddConstantPlugin(), "Cannot register LeNetAddConstant plugin");
        TrtPtr<IBuilder> builder(createInferBuilder(logger));
        require(bool(builder), "Cannot create builder");
        // Keep the original plugin alive until after network destruction.
        std::unique_ptr<IPluginV2, PluginDeleter> plugin;
        TrtPtr<INetworkDefinition> network(builder->createNetworkV2(
            1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        require(bool(network), "Cannot create network");
        TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        require(bool(parser), "Cannot create ONNX parser");
        if (!parser->parseFromFile(argv[1], static_cast<int>(ILogger::Severity::kWARNING))) {
            for (int i = 0; i < parser->getNbErrors(); ++i)
                std::cerr << parser->getError(i)->desc() << '\n';
            throw std::runtime_error("ONNX parsing failed");
        }
        require(network->getNbInputs() == 1 && network->getNbOutputs() == 1,
                "Expected one LeNet input and one output");
        ITensor* in = network->getInput(0);
        require(!in->isShapeTensor() && in->getType() == DataType::kFLOAT,
                "Expected an FP32 execution input");
        Dims dims = in->getDimensions();
        require(dims.nbDims == 4, "Expected NCHW input");
        int expected[] = {1, 1, 32, 32};
        bool dynamic = false;
        for (int d = 0; d < 4; ++d) {
            require(dims.d[d] == -1 || dims.d[d] == expected[d],
                    "This example expects input 1x1x32x32");
            dynamic |= dims.d[d] == -1;
            dims.d[d] = expected[d];
        }
        TrtPtr<IBuilderConfig> config(builder->createBuilderConfig());
        require(bool(config), "Cannot create builder config");
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, size_t{1} << 30);
        config->clearFlag(BuilderFlag::kTF32); // Easier FP32 reference comparison.
        if (dynamic) {
            // Builder owns profiles returned by createOptimizationProfile().
            auto* profile = builder->createOptimizationProfile();
            require(profile != nullptr, "Cannot create profile");
            for (auto selector : {OptProfileSelector::kMIN, OptProfileSelector::kOPT,
                                  OptProfileSelector::kMAX})
                require(profile->setDimensions(in->getName(), selector, dims), "Invalid profile dims");
            require(profile->isValid() && config->addOptimizationProfile(profile) >= 0,
                    "Cannot add optimization profile");
        }
        ITensor* original = network->getOutput(0);
        require(original->getType() == DataType::kFLOAT, "Expected FP32 LeNet output");
        if (!baseline) {
            auto* creator = getPluginRegistry()->getPluginCreator("LeNetAddConstant", "1", "lenet_demo");
            require(creator != nullptr, "Creator not found");
            PluginField field{"value", &value, PluginFieldType::kFLOAT32, 1};
            PluginFieldCollection fields{1, &field};
            plugin.reset(creator->createPlugin("lenet_add_constant", &fields));
            require(bool(plugin), "Cannot create plugin");
            ITensor* inputs[] = {original};
            auto* layer = network->addPluginV2(inputs, 1, *plugin);
            require(layer != nullptr, "Cannot add plugin layer");
            layer->setName("lenet_add_constant");
            network->unmarkOutput(*original);
            original->setName("lenet_before_add_constant");
            auto* result = layer->getOutput(0);
            result->setName("output_plus_constant");
            network->markOutput(*result);
        }
        TrtPtr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        require(bool(plan), "Engine build failed");
        writeFile(argv[2], plan->data(), plan->size());
        std::cout << "Saved " << argv[2];
        if (baseline) std::cout << " (baseline)\n";
        else std::cout << " (output + " << value << ")\n";
        return 0;
    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
