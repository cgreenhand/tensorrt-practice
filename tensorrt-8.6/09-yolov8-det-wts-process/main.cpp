#include "model.h"
#include "block.h"
#include "config.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

using namespace yolov8::det;

class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

namespace
{

TRTLogger gLogger;
constexpr char kOutputCountTensorName[] = "output_count";
constexpr int32_t kDecodedDetectionWidth = 6;

struct ModelScale
{
    float gd{};
    float gw{};
    int maxChannels{};
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

class CudaBuffer
{
public:
    explicit CudaBuffer(std::size_t bytes)
    {
        CUDA_CHECK(cudaMalloc(&mPointer, bytes));
    }

    ~CudaBuffer()
    {
        if (mPointer != nullptr)
        {
            cudaFree(mPointer);
        }
    }

    CudaBuffer(CudaBuffer const&) = delete;
    CudaBuffer& operator=(CudaBuffer const&) = delete;

    void* get() noexcept
    {
        return mPointer;
    }

private:
    void* mPointer{};
};

class CudaStream
{
public:
    CudaStream()
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    ~CudaStream()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;

    cudaStream_t get() const noexcept
    {
        return mStream;
    }

private:
    cudaStream_t mStream{};
};

void printUsage(char const* program)
{
    std::cerr << "Usage:\n"
              << "  " << program << " -s [n/s/m/l/x] [wts_path]\n"
              << "  " << program << " -d [engine_path] [image_dir]" << std::endl;
}

ModelScale parseModelScale(std::string const& variant)
{
    if (variant.size() != 1)
    {
        throw std::invalid_argument("Model variant must be one of n/s/m/l/x");
    }

    switch (variant.front())
    {
    case 'n':
        return {0.33F, 0.25F, 1024};
    case 's':
        return {0.33F, 0.50F, 1024};
    case 'm':
        return {0.67F, 0.75F, 768};
    case 'l':
        return {1.00F, 1.00F, 512};
    case 'x':
        return {1.00F, 1.25F, 512};
    default:
        throw std::invalid_argument("Model variant must be one of n/s/m/l/x");
    }
}

nvinfer1::DataType inputDataType()
{
    return kPrecision == Precision::FP16
        ? nvinfer1::DataType::kHALF
        : nvinfer1::DataType::kFLOAT;
}

void configurePrecision(nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config)
{
    if (kPrecision == Precision::FP16)
    {
        if (!builder.platformHasFastFp16())
        {
            std::cout << "Warning: platform does not report fast FP16 support" << std::endl;
        }
        config.setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (kPrecision == Precision::INT8)
    {
        if (!builder.platformHasFastInt8())
        {
            throw std::runtime_error("Platform does not support fast INT8");
        }
        config.setFlag(nvinfer1::BuilderFlag::kINT8);
    }
}

std::filesystem::path enginePathFor(std::filesystem::path wtsPath)
{
    wtsPath.replace_extension(".engine");
    return wtsPath;
}

void buildEngine(std::string const& variant, std::filesystem::path const& wtsPath)
{
    auto const totalStartTime = std::chrono::high_resolution_clock::now();

    if (!std::filesystem::is_regular_file(wtsPath))
    {
        throw std::runtime_error("Weights file does not exist: " + wtsPath.string());
    }

    ModelScale scale = parseModelScale(variant);
    std::filesystem::path const enginePath = enginePathFor(wtsPath);

    std::cout << "Building YOLOv8" << variant << " engine from " << wtsPath << std::endl;

    TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    if (!builder)
    {
        throw std::runtime_error("Failed to create TensorRT builder");
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        throw std::runtime_error("Failed to create TensorRT builder config");
    }

    configurePrecision(*builder, *config);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, kWorkspaceBytes);

    auto const engineBuildStartTime = std::chrono::high_resolution_clock::now();
    TrtUniquePtr<nvinfer1::IHostMemory> serializedEngine{
        buildEngineYolov8Det(
            builder.get(),
            config.get(),
            inputDataType(),
            wtsPath.string(),
            scale.gd,
            scale.gw,
            scale.maxChannels)};
    auto const engineBuildEndTime = std::chrono::high_resolution_clock::now();
    if (!serializedEngine)
    {
        throw std::runtime_error("Failed to build serialized TensorRT engine");
    }
    double const engineBuildMilliseconds =
        std::chrono::duration<double, std::milli>(
            engineBuildEndTime - engineBuildStartTime).count();
    std::cout << std::fixed << std::setprecision(2)
              << "[Timer] Engine build: " << engineBuildMilliseconds << " ms" << std::endl;

    auto const engineSaveStartTime = std::chrono::high_resolution_clock::now();
    std::ofstream output(enginePath, std::ios::binary);
    if (!output)
    {
        throw std::runtime_error("Failed to open engine output file: " + enginePath.string());
    }
    output.write(
        static_cast<char const*>(serializedEngine->data()),
        static_cast<std::streamsize>(serializedEngine->size()));
    output.flush();
    if (!output)
    {
        throw std::runtime_error("Failed to write engine file: " + enginePath.string());
    }
    auto const engineSaveEndTime = std::chrono::high_resolution_clock::now();
    double const engineSaveMilliseconds =
        std::chrono::duration<double, std::milli>(
            engineSaveEndTime - engineSaveStartTime).count();
    std::cout << "[Timer] Engine save: " << engineSaveMilliseconds << " ms" << std::endl;

    std::cout << "Engine saved to: " << enginePath << std::endl;
    auto const totalEndTime = std::chrono::high_resolution_clock::now();
    double const totalMilliseconds =
        std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();
    std::cout << "[Timer] Build total: " << totalMilliseconds << " ms" << std::endl;
}

std::vector<char> loadEngineFile(std::filesystem::path const& enginePath)
{
    std::ifstream input(enginePath, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("Failed to open engine file: " + enginePath.string());
    }

    std::streamsize const size = input.tellg();
    if (size <= 0)
    {
        throw std::runtime_error("Engine file is empty: " + enginePath.string());
    }

    std::vector<char> data(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    input.read(data.data(), size);
    if (!input)
    {
        throw std::runtime_error("Failed to read engine file: " + enginePath.string());
    }
    return data;
}

bool isSupportedImage(std::filesystem::path const& path)
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".jpg" || extension == ".jpeg" ||
        extension == ".png" || extension == ".bmp";
}

std::vector<std::filesystem::path> collectImages(std::filesystem::path const& imageDirectory)
{
    if (!std::filesystem::is_directory(imageDirectory))
    {
        throw std::runtime_error("Image directory does not exist: " + imageDirectory.string());
    }

    std::vector<std::filesystem::path> imagePaths;
    for (auto const& entry : std::filesystem::directory_iterator(imageDirectory))
    {
        if (entry.is_regular_file() && isSupportedImage(entry.path()))
        {
            imagePaths.push_back(entry.path());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

void drawDetections(cv::Mat& image, Detection const* detections, int32_t count)
{
    for (int32_t index = 0; index < count; ++index)
    {
        Detection const& detection = detections[index];
        int const x1 = std::clamp(cvRound(detection.x1), 0, image.cols - 1);
        int const y1 = std::clamp(cvRound(detection.y1), 0, image.rows - 1);
        int const x2 = std::clamp(cvRound(detection.x2), 0, image.cols - 1);
        int const y2 = std::clamp(cvRound(detection.y2), 0, image.rows - 1);

        cv::rectangle(
            image,
            cv::Point{x1, y1},
            cv::Point{x2, y2},
            cv::Scalar{0, 255, 0},
            2);
        std::string const label = cv::format(
            "class %d %.2f", detection.classId, detection.score);
        cv::putText(
            image,
            label,
            cv::Point{x1, std::max(0, y1 - 5)},
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar{0, 255, 0},
            1,
            cv::LINE_AA);
    }
}

void runInference(
    std::filesystem::path const& enginePath,
    std::filesystem::path const& requestedImageDirectory)
{
    CUDA_CHECK(cudaSetDevice(kGpuId));
    auto const endToEndStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "Loading engine: " << enginePath << std::endl;
    auto const engineLoadStartTime = std::chrono::high_resolution_clock::now();
    std::vector<char> const serializedEngine = loadEngineFile(enginePath);

    TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
    if (!runtime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    TrtUniquePtr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(serializedEngine.data(), serializedEngine.size())};
    auto const engineLoadEndTime = std::chrono::high_resolution_clock::now();
    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    double const engineLoadMilliseconds =
        std::chrono::duration<double, std::milli>(
            engineLoadEndTime - engineLoadStartTime).count();
    std::cout << std::fixed << std::setprecision(2)
              << "[Timer] Engine load: " << engineLoadMilliseconds << " ms" << std::endl;

    TrtUniquePtr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context)
    {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    if (engine->getNbBindings() != 3)
    {
        throw std::runtime_error("Expected exactly three engine bindings");
    }

    int const inputIndex = engine->getBindingIndex(kInputTensorName);
    int const outputIndex = engine->getBindingIndex(kOutputTensorName);
    int const outputCountIndex = engine->getBindingIndex(kOutputCountTensorName);
    if (inputIndex != 0 || outputIndex != 1 || outputCountIndex != 2)
    {
        throw std::runtime_error("Unexpected engine binding indices");
    }
    if (engine->getBindingDataType(inputIndex) != inputDataType() ||
        engine->getBindingDataType(outputIndex) != nvinfer1::DataType::kFLOAT ||
        engine->getBindingDataType(outputCountIndex) != nvinfer1::DataType::kINT32)
    {
        throw std::runtime_error("Engine binding data types do not match the configured precision");
    }

    int32_t constexpr expectedCandidateCount =
        (kInputHeight / 8) * (kInputWidth / 8) +
        (kInputHeight / 16) * (kInputWidth / 16) +
        (kInputHeight / 32) * (kInputWidth / 32);
    nvinfer1::Dims const inputDimensions = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims const outputDimensions = engine->getBindingDimensions(outputIndex);
    nvinfer1::Dims const outputCountDimensions = engine->getBindingDimensions(outputCountIndex);
    if (inputDimensions.nbDims != 4 || inputDimensions.d[0] != kBatchSize ||
        inputDimensions.d[1] != kInputChannels || inputDimensions.d[2] != kInputHeight ||
        inputDimensions.d[3] != kInputWidth || outputDimensions.nbDims != 3 ||
        outputDimensions.d[0] != kBatchSize || outputDimensions.d[1] != expectedCandidateCount ||
        outputDimensions.d[2] != kDecodedDetectionWidth || outputCountDimensions.nbDims != 1 ||
        outputCountDimensions.d[0] != kBatchSize)
    {
        throw std::runtime_error("Engine binding dimensions do not match the configured model");
    }

    std::filesystem::path const imageDirectory =
        std::filesystem::weakly_canonical(requestedImageDirectory);
    std::vector<std::filesystem::path> const imagePaths = collectImages(imageDirectory);
    std::filesystem::path const outputDirectory = imageDirectory.parent_path() / "output";
    std::filesystem::create_directories(outputDirectory);

    std::cout << "Found " << imagePaths.size() << " images" << std::endl;
    if (imagePaths.empty())
    {
        std::cout << "No supported images to process" << std::endl;
        auto const endToEndEndTime = std::chrono::high_resolution_clock::now();
        double const endToEndMilliseconds =
            std::chrono::duration<double, std::milli>(
                endToEndEndTime - endToEndStartTime).count();
        std::cout << "[Timer] Inference total: 0.00 ms\n"
                  << "[Timer] End-to-end total: " << endToEndMilliseconds << " ms\n"
                  << "[Timer] Summary - Images: 0 | Total: 0.00 ms | Average: 0.00 ms/image"
                  << std::endl;
        return;
    }

    CudaYoloPreprocessor preprocessor(kInputWidth, kInputHeight, kBatchSize);
    yolov8::CudaPostprocessor postprocessor;
    CudaStream stream;

    std::size_t const inputElementBytes = inputDataType() == nvinfer1::DataType::kHALF
        ? sizeof(uint16_t)
        : sizeof(float);
    std::size_t const inputBytes = static_cast<std::size_t>(kBatchSize) *
        kInputChannels * kInputHeight * kInputWidth * inputElementBytes;
    std::size_t const pluginOutputBytes = static_cast<std::size_t>(kBatchSize) *
        postprocessor.candidateCount() * kDecodedDetectionWidth * sizeof(float);
    std::size_t const pluginOutputCountBytes =
        static_cast<std::size_t>(kBatchSize) * sizeof(int32_t);

    CudaBuffer deviceInput(inputBytes);
    CudaBuffer pluginOutput(pluginOutputBytes);
    CudaBuffer pluginOutputCount(pluginOutputCountBytes);
    CudaBuffer deviceDetections(postprocessor.outputDetectionBytes());
    CudaBuffer deviceCounts(postprocessor.outputCountBytes());

    std::vector<Detection> hostDetections(
        static_cast<std::size_t>(kBatchSize) * postprocessor.maxDetections());
    std::vector<int32_t> hostCounts(static_cast<std::size_t>(kBatchSize));

    std::size_t processedImages = 0;
    std::size_t totalDetections = 0;
    std::size_t const totalBatches =
        (imagePaths.size() + static_cast<std::size_t>(kBatchSize) - 1) /
        static_cast<std::size_t>(kBatchSize);
    auto const inferenceStartTime = std::chrono::high_resolution_clock::now();

    for (std::size_t offset = 0; offset < imagePaths.size(); offset += kBatchSize)
    {
        std::vector<cv::Mat> images;
        std::vector<std::filesystem::path> batchPaths;
        images.reserve(kBatchSize);
        batchPaths.reserve(kBatchSize);

        std::size_t const batchEnd = std::min(
            imagePaths.size(), offset + static_cast<std::size_t>(kBatchSize));
        for (std::size_t index = offset; index < batchEnd; ++index)
        {
            cv::Mat image = cv::imread(imagePaths[index].string(), cv::IMREAD_COLOR);
            if (image.empty())
            {
                throw std::runtime_error("Failed to read image: " + imagePaths[index].string());
            }
            images.push_back(std::move(image));
            batchPaths.push_back(imagePaths[index]);
        }

        int32_t const validBatch = static_cast<int32_t>(images.size());
        auto const preprocessStartTime = std::chrono::high_resolution_clock::now();
        std::vector<LetterboxInfo> const letterboxInfos = preprocessor.preprocess(
            images, deviceInput.get(), inputDataType(), stream.get());
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        auto const preprocessEndTime = std::chrono::high_resolution_clock::now();

        std::array<void*, 3> bindings{};
        bindings[static_cast<std::size_t>(inputIndex)] = deviceInput.get();
        bindings[static_cast<std::size_t>(outputIndex)] = pluginOutput.get();
        bindings[static_cast<std::size_t>(outputCountIndex)] = pluginOutputCount.get();
        auto const inferenceBatchStartTime = std::chrono::high_resolution_clock::now();
        if (!context->enqueueV2(bindings.data(), stream.get(), nullptr))
        {
            throw std::runtime_error("TensorRT enqueueV2 failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        auto const inferenceBatchEndTime = std::chrono::high_resolution_clock::now();

        auto const postprocessStartTime = std::chrono::high_resolution_clock::now();
        postprocessor.enqueue(
            static_cast<float const*>(pluginOutput.get()),
            static_cast<int32_t const*>(pluginOutputCount.get()),
            letterboxInfos.data(),
            validBatch,
            static_cast<Detection*>(deviceDetections.get()),
            static_cast<int32_t*>(deviceCounts.get()),
            stream.get());

        CUDA_CHECK(cudaMemcpyAsync(
            hostDetections.data(),
            deviceDetections.get(),
            postprocessor.outputDetectionBytes(),
            cudaMemcpyDeviceToHost,
            stream.get()));
        CUDA_CHECK(cudaMemcpyAsync(
            hostCounts.data(),
            deviceCounts.get(),
            postprocessor.outputCountBytes(),
            cudaMemcpyDeviceToHost,
            stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        auto const postprocessEndTime = std::chrono::high_resolution_clock::now();

        auto const drawAndSaveStartTime = std::chrono::high_resolution_clock::now();
        for (int32_t batchIndex = 0; batchIndex < validBatch; ++batchIndex)
        {
            int32_t const detectionCount = std::clamp(
                hostCounts[static_cast<std::size_t>(batchIndex)],
                int32_t{0},
                postprocessor.maxDetections());
            Detection const* detections = hostDetections.data() +
                static_cast<std::size_t>(batchIndex) * postprocessor.maxDetections();
            drawDetections(
                images[static_cast<std::size_t>(batchIndex)],
                detections,
                detectionCount);

            std::filesystem::path const outputPath = outputDirectory /
                batchPaths[static_cast<std::size_t>(batchIndex)].filename();
            if (!cv::imwrite(
                    outputPath.string(),
                    images[static_cast<std::size_t>(batchIndex)]))
            {
                throw std::runtime_error("Failed to save result: " + outputPath.string());
            }

            ++processedImages;
            totalDetections += static_cast<std::size_t>(detectionCount);
            std::cout << batchPaths[static_cast<std::size_t>(batchIndex)].filename()
                      << ": " << detectionCount << " detections" << std::endl;
        }
        auto const drawAndSaveEndTime = std::chrono::high_resolution_clock::now();

        double const preprocessMilliseconds =
            std::chrono::duration<double, std::milli>(
                preprocessEndTime - preprocessStartTime).count();
        double const inferenceBatchMilliseconds =
            std::chrono::duration<double, std::milli>(
                inferenceBatchEndTime - inferenceBatchStartTime).count();
        double const postprocessMilliseconds =
            std::chrono::duration<double, std::milli>(
                postprocessEndTime - postprocessStartTime).count();
        double const drawAndSaveMilliseconds =
            std::chrono::duration<double, std::milli>(
                drawAndSaveEndTime - drawAndSaveStartTime).count();
        std::size_t const batchNumber =
            offset / static_cast<std::size_t>(kBatchSize) + 1;
        std::cout << "[Timer] Batch " << batchNumber << '/' << totalBatches
                  << " - Preprocess: " << preprocessMilliseconds
                  << " ms | Inference: " << inferenceBatchMilliseconds
                  << " ms | Postprocess: " << postprocessMilliseconds
                  << " ms | Draw&Save: " << drawAndSaveMilliseconds << " ms"
                  << std::endl;
    }

    auto const inferenceEndTime = std::chrono::high_resolution_clock::now();
    double const inferenceMilliseconds =
        std::chrono::duration<double, std::milli>(
            inferenceEndTime - inferenceStartTime).count();
    double const endToEndMilliseconds =
        std::chrono::duration<double, std::milli>(
            inferenceEndTime - endToEndStartTime).count();
    double const averageMilliseconds = inferenceMilliseconds /
        static_cast<double>(processedImages);

    std::cout << "Processed images: " << processedImages << '\n'
              << "Total detections: " << totalDetections << '\n'
              << "Total time: " << inferenceMilliseconds << " ms\n"
              << "Results saved to: " << outputDirectory << '\n'
              << "[Timer] Inference total: " << inferenceMilliseconds << " ms\n"
              << "[Timer] End-to-end total: " << endToEndMilliseconds << " ms\n"
              << "[Timer] Summary - Images: " << processedImages
              << " | Total: " << inferenceMilliseconds
              << " ms | Average: " << averageMilliseconds << " ms/image"
              << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    try
    {
        std::string const mode = argv[1];
        if (mode == "-s")
        {
            buildEngine(argv[2], argv[3]);
        }
        else if (mode == "-d")
        {
            runInference(argv[2], argv[3]);
        }
        else
        {
            printUsage(argv[0]);
            return 1;
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
