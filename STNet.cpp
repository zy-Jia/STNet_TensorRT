#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 24;
static const int INPUT_W = 94;
static const int OUTPUT_SIZE = 3*2;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../STNet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, DimsHW{3, 3}, weightMap["localization.0.weight"], weightMap["localization.0.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*pool1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 32, DimsHW{5, 5}, weightMap["localization.3.weight"], weightMap["localization.3.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});

    IPoolingLayer* pool2 = network->addPoolingNd(*conv2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool2);
    pool2->setStrideNd(DimsHW{3, 3});
    IActivationLayer* relu2 = network->addActivation(*pool2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*relu2->getOutput(0), 32, weightMap["fc_loc.0.weight"], weightMap["fc_loc.0.bias"]);
    assert(fc1);

    IActivationLayer* relu3 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    IFullyConnectedLayer* fc2 = network->addFullyConnected(*relu3->getOutput(0), 6, weightMap["fc_loc.2.weight"], weightMap["fc_loc.2.bias"]);
    assert(fc2);

    fc2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc2->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./STNet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./STNet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("STNet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("STNet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    // (204, 138) (267, 164)
    // (218, 124) (281, 149)
    // (193, 125) (255, 148)
    // Subtract mean from image
    auto read_start = std::chrono::system_clock::now();
    // cv::Mat img = cv::imread("/home/nvidia/projects/License_Plate_Detection_Pytorch/test/20230524112133.jpg");
    cv::Mat img = cv::imread("/home/nvidia/projects/License_Plate_Detection_Pytorch/test/20230526091428.jpg");
    // cv::Mat img = cv::imread("/home/nvidia/projects/License_Plate_Detection_Pytorch/test/20230526093151.jpg");
    // (y, y + h), (x x + w)
    // cv::Mat tmp = img(cv::Range(138.0, 164.0 + 1.0), cv::Range(204.0, 267.0 + 1.0));
    cv::Mat tmp = img(cv::Range(124.0, 149.0 + 1.0), cv::Range(218.0, 281.0 + 1.0));
    // cv::Mat tmp = img(cv::Range(125.0, 148.0 + 1.0), cv::Range(193.0, 255.0 + 1.0));
    auto read_end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start).count() << "ms" << std::endl;

    static float data[3 * INPUT_H * INPUT_W];

    cv::Mat pr_img(INPUT_H, INPUT_W, CV_8UC3);
    cv::resize(tmp, pr_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_CUBIC);

    // for (int i = 0; i < INPUT_H * INPUT_W; i++) {
    //     // BGR
    //     data[i] = (pr_img.at<cv::Vec3b>(i)[2] / 255.0 - 0.485) / 0.229;
    //     data[i + INPUT_H * INPUT_W] = (pr_img.at<cv::Vec3b>(i)[1] / 255.0 - 0.456) / 0.224;
    //     data[i + 2 * INPUT_H * INPUT_W] = (pr_img.at<cv::Vec3b>(i)[0] / 255.0 - 0.406) / 0.225;
    // }

    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[2] - 127.5) * 0.0078125;
            data[i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] - 127.5) * 0.0078125;
            data[i] = ((float)uc_pixel[0] - 127.5) * 0.0078125;
            uc_pixel += 3;
            ++i;
        }
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];

    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // [ 1.0470,  0.0234, -0.0016, 0.0041,  1.1044, -0.0060]
    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;

    cv::Mat warp_mat = (cv::Mat_<double>(2,3)<<prob[0], -prob[1], prob[2], -prob[3], prob[4], prob[5]);
	cv::Mat warp_dst;

    /// 设置目标图像的大小和类型与源图像一致
	warp_dst = cv::Mat::zeros(pr_img.rows, pr_img.cols, pr_img.type());
 
	/// 对源图像应用上面求得的仿射变换
	cv::warpAffine(pr_img, warp_dst, warp_mat, warp_dst.size());

    cv::imwrite("./warp_dst.jpg", warp_dst);

    return 0;
}
