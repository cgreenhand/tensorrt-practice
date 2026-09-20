#include "common.h"
#include "add_constant_plugin.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace nvinfer1;
int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage: lenet_runtime model.engine input.bin output.bin [baseline_output.bin expected_value]\n"
                  << "expected_value is ONLY for verification, not for changing the engine.\n";
        return 1;
    }
    try {
        float const expectedConstant = argc == 6 ? parseConstant(argv[5]) : 0.0F;
        Logger logger;
        // Must precede deserializeCudaEngine, including in a fresh process.
        require(registerAddConstantPlugin(), "Cannot register LeNetAddConstant plugin");
        TrtPtr<IRuntime> runtime(createInferRuntime(logger));
        require(bool(runtime), "Cannot create runtime");
        auto bytes = readFile(argv[1]);
        TrtPtr<ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(engine), "Deserialization failed; check plugin and TensorRT/GPU versions");
        require(engine->getNbOptimizationProfiles() == 1 && engine->getNbBindings() == 2,
                "Expected one profile and two bindings");
        TrtPtr<IExecutionContext> context(engine->createExecutionContext());
        require(bool(context), "Cannot create execution context");
        int inputIndex = -1, outputIndex = -1;
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            require(engine->getBindingDataType(i) == DataType::kFLOAT
                    && engine->getBindingFormat(i) == TensorFormat::kLINEAR
                    && engine->getLocation(i) == TensorLocation::kDEVICE
                    && engine->isExecutionBinding(i) && !engine->isShapeBinding(i),
                    "Expected linear FP32 device execution bindings");
            if (engine->bindingIsInput(i)) {
                require(inputIndex == -1, "Multiple inputs unsupported"); inputIndex = i;
            } else {
                require(outputIndex == -1, "Multiple outputs unsupported"); outputIndex = i;
            }
        }
        require(inputIndex >= 0 && outputIndex >= 0, "Missing input/output");
        auto dims = engine->getBindingDimensions(inputIndex);
        require(dims.nbDims == 4, "Expected NCHW input");
        int expected[] = {1, 1, 32, 32};
        bool dynamic = false;
        for (int d = 0; d < 4; ++d) {
            require(dims.d[d] == -1 || dims.d[d] == expected[d], "Expected 1x1x32x32 input");
            dynamic |= dims.d[d] == -1;
            dims.d[d] = expected[d];
        }
        if (dynamic) require(context->setBindingDimensions(inputIndex, dims), "Cannot set input shape");
        require(context->allInputDimensionsSpecified(), "Input dimensions unresolved");
        size_t const inputCount = volume(context->getBindingDimensions(inputIndex));
        size_t const outputCount = volume(context->getBindingDimensions(outputIndex));
        require(outputCount == 10, "Expected 10 LeNet output elements");
        auto input = readFloats(argv[2], inputCount);
        std::vector<float> output(outputCount);
        DeviceBuffer deviceInput(inputCount * sizeof(float)), deviceOutput(outputCount * sizeof(float));
        // Stream is destroyed before device buffers/context, also on an exception.
        Stream stream;
        std::vector<void*> bindings(engine->getNbBindings(), nullptr);
        bindings[inputIndex] = deviceInput.ptr;
        bindings[outputIndex] = deviceOutput.ptr;
        cudaCheck(cudaMemcpyAsync(deviceInput.ptr, input.data(), inputCount * sizeof(float),
                                  cudaMemcpyHostToDevice, stream.value));
        require(context->enqueueV2(bindings.data(), stream.value, nullptr), "enqueueV2 failed");
        cudaCheck(cudaMemcpyAsync(output.data(), deviceOutput.ptr, outputCount * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream.value));
        cudaCheck(cudaStreamSynchronize(stream.value));
        writeFile(argv[3], output.data(), output.size() * sizeof(float));
        std::cout << "Output tensor: " << engine->getBindingName(outputIndex) << '\n' << std::setprecision(9);
        for (size_t i = 0; i < output.size(); ++i) {
            require(std::isfinite(output[i]), "Non-finite inference output");
            std::cout << i << ": " << output[i] << '\n';
        }
        std::cout << "argmax = " << std::distance(output.begin(), std::max_element(output.begin(), output.end())) << '\n';
        if (argc == 6) {
            auto baseline = readFloats(argv[4], outputCount);
            bool pass = true;
            float maxError = 0;
            for (size_t i = 0; i < outputCount; ++i) {
                float const expectedValue = baseline[i] + expectedConstant;
                require(std::isfinite(expectedValue), "Non-finite reference");
                float const error = std::fabs(output[i] - expectedValue);
                maxError = std::max(maxError, error);
                pass &= error <= 1e-5F + 1e-4F * std::fabs(expectedValue);
            }
            std::cout << "max |output - (baseline + " << expectedConstant << ")| = " << maxError
                      << " -> " << (pass ? "PASS" : "FAIL") << '\n';
            return pass ? 0 : 2;
        }
        return 0;
    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
