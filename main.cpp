#include "MLComputeCpp/CppMLCDevice.h"

#include "MLComputeCpp/CppMLCTensor.h"
#include "MLComputeCpp/CppMLCTensorDescriptor.h"

#include "MLComputeCpp/CppMLCActivationLayer.h"
#include "MLComputeCpp/CppMLCActivationDescriptor.h"

#include "MLComputeCpp/CppMLCFullyConnectedLayer.h"

#include "MLComputeCpp/CppMLCConvolutionDescriptor.h"
#include "MLComputeCpp/CppMLCSoftmaxLayer.h"
#include "MLComputeCpp/CppMLCTrainingGraph.h"
#include "MLComputeCpp/CppMLCInferenceGraph.h"
#include "MLComputeCpp/CppMLCLossDescriptor.h"
#include "MLComputeCpp/CppMLCLossLayer.h"
#include "MLComputeCpp/CppMLCAdamOptimizer.h"
#include "MLComputeCpp/CppMLCOptimizerDescriptor.h"
#include "MLComputeCpp/CppMLCGraph.h"
#include "MLComputeCpp/CppMLCArithmeticLayer.h"

#include <iostream>
#include <fstream>
#include <functional>

class MNISTTrain
{
public:
    constexpr static auto batchSize = 32;
    constexpr static auto imageSize = 28*28;
    constexpr static auto dense1LayerOutputSize = 128;
    constexpr static auto numberOfClasses = 10;

    MNISTTrain()
      : _device{CppMLCDevice::deviceWithType(eMLCDeviceType::CPU)}
      , _inputTensor{CppMLCTensor({batchSize, imageSize, 1, 1}, eMLCDataType::Float32)}
      , _dense1WeightsTensor{CppMLCTensor({1, imageSize*dense1LayerOutputSize, 1, 1},
                                          eMLCRandomInitializerType::GlorotUniform)}
      , _dense1BiasesTensor{CppMLCTensor({1, dense1LayerOutputSize, 1, 1},
                            eMLCRandomInitializerType::GlorotUniform)}
      , _dense2WeightsTensor{CppMLCTensor({1, dense1LayerOutputSize*numberOfClasses, 1, 1},
                             eMLCRandomInitializerType::GlorotUniform)}
      , _dense2BiasesTensor{CppMLCTensor({1, numberOfClasses, 1, 1},
                            eMLCRandomInitializerType::GlorotUniform)}
      , _lossLabelTensor{CppMLCTensor({batchSize, numberOfClasses}, eMLCDataType::Float32)}
      , _graph{CppMLCGraph::graph()}
      , _convolutionDescriptor{CppMLCConvolutionDescriptor::descriptorWithKernelWidth(dense1LayerOutputSize,
                                                                                      imageSize,
                                                                                      imageSize,
                                                                                      dense1LayerOutputSize)}
      , _fullyConnectedLayer1{CppMLCFullyConnectedLayer::layerWithWeights(_dense1WeightsTensor,
                                                                          _dense1BiasesTensor,
                                                                          _convolutionDescriptor)}
      , _dense1{_graph.nodeWithLayer(_fullyConnectedLayer1, _inputTensor)}
      , _activationDescriptor{CppMLCActivationDescriptor::descriptorWithType(eMLCActivationType::ReLU)}
      , _activationLayer{CppMLCActivationLayer::layerWithDescriptor(_activationDescriptor)}
      , _relu1{_graph.nodeWithLayer(_activationLayer, _dense1)}
      , _fullyConnectedLayer2{CppMLCFullyConnectedLayer::layerWithWeights(
                    _dense2WeightsTensor,
                    _dense2BiasesTensor,
                    CppMLCConvolutionDescriptor::descriptorWithKernelWidth(numberOfClasses,
                                                                           dense1LayerOutputSize,
                                                                           dense1LayerOutputSize,
                                                                           numberOfClasses))}
      , _dense2{_graph.nodeWithLayer(_fullyConnectedLayer2, _relu1)}
      , _outputSoftmax{_graph.nodeWithLayer(
                    CppMLCSoftmaxLayer::layerWithOperation(eMLCSoftmaxOperation::Softmax),
                    _dense2)}

      , _trainingGraph{CppMLCTrainingGraph::graphWithGraphObjects({_graph},
                                                                  CppMLCLossLayer::layerWithDescriptor(
                                                                          CppMLCLossDescriptor::descriptorWithType(
                                                                                  eMLCLossType::SoftmaxCrossEntropy,
                                                                                  eMLCReductionType::Mean)),
                                                                  CppMLCAdamOptimizer::optimizerWithDescriptor(
                                                                          CppMLCOptimizerDescriptor::descriptorWithLearningRate(
                                                                                  0.001,
                                                                                  1.0,
                                                                                  eMLCRegularizationType::None,
                                                                                  0.0),
                                                                          0.9,
                                                                          0.999,
                                                                          1e-7,
                                                                          1))}
    {
        _trainingGraph.addInputs({std::make_pair("image", _inputTensor)},
                                 {std::make_pair("label", _lossLabelTensor)});

        _trainingGraph.compileWithOptions(eMLCGraphCompilationOptions::None, _device);
    }

    void trainGraph(std::function<void(std::string const&)>&& log)
    {
        //buildGraph();
        //buildTrainingGraph();
        execTrainingLoop(log);
        evaluateGraph(log);
    }

    auto oneHotEncoding(uint32_t number, uint32_t length = 10) -> std::vector<float>
    {
        auto array = std::vector<float>(0.0, length);
        array[number] = 1.0;
        return array;
    }

    auto oneHotDecoding(float const* begin, float const* end) -> int32_t
    {
        auto foundLabelIt = std::find(begin, end, 1);
        return std::distance(begin, foundLabelIt != end ? foundLabelIt : begin);
    }

    auto argmaxDecoding(float const* begin, float const* end) -> int32_t
    {
        auto foundLabelIt = std::max_element(begin, end);
        return std::distance(begin, foundLabelIt != end ? foundLabelIt : begin);
    }

    void buildGraph()
    {
        _graph = CppMLCGraph::graph();

        _dense1 = _graph.nodeWithLayer(
                CppMLCFullyConnectedLayer::layerWithWeights(
                        _dense1WeightsTensor,
                        _dense1BiasesTensor,
                        CppMLCConvolutionDescriptor::descriptorWithKernelWidth(dense1LayerOutputSize,
                                                                               imageSize,
                                                                               imageSize,
                                                                               dense1LayerOutputSize)),
                {_inputTensor});

        _relu1 = _graph.nodeWithLayer(
                CppMLCActivationLayer::layerWithDescriptor(
                        CppMLCActivationDescriptor::descriptorWithType(eMLCActivationType::ReLU)),
                        _dense1);

        _dense2 = _graph.nodeWithLayer(
                CppMLCFullyConnectedLayer::layerWithWeights(
                        _dense2WeightsTensor,
                        _dense2BiasesTensor,
                        CppMLCConvolutionDescriptor::descriptorWithKernelWidth(numberOfClasses,
                                                                               dense1LayerOutputSize,
                                                                               dense1LayerOutputSize,
                                                                               numberOfClasses)),
                {_relu1});

        _outputSoftmax = _graph.nodeWithLayer(
                CppMLCSoftmaxLayer::layerWithOperation(eMLCSoftmaxOperation::Softmax),
                _dense2);
    }

    void buildTrainingGraph()
    {
        _trainingGraph = CppMLCTrainingGraph::graphWithGraphObjects({_graph},
                                             CppMLCLossLayer::layerWithDescriptor(
                                                     CppMLCLossDescriptor::descriptorWithType(
                                                             eMLCLossType::SoftmaxCrossEntropy,
                                                             eMLCReductionType::Mean)),
                                             CppMLCAdamOptimizer::optimizerWithDescriptor(
                                                      CppMLCOptimizerDescriptor::descriptorWithLearningRate(
                                                              0.001,
                                                              1.0,
                                                              eMLCRegularizationType::None,
                                                              0.0),
                                                      0.9,
                                                      0.999,
                                                      1e-7,
                                                      1));

        _trainingGraph.addInputs({std::make_pair("image", _inputTensor)},
                                 {std::make_pair("label", _lossLabelTensor)});

        _trainingGraph.compileWithOptions(eMLCGraphCompilationOptions::None, _device);
    }

    void execTrainingLoop(std::function<void(std::string)> const& log)
    {
        const auto trainingSample = _trainingDataX.size() / imageSize;
        const auto trainingBatches = trainingSample / batchSize;

        for (auto epoch = 0; epoch < epochs; ++epoch)
        {
            auto epochMatch = 0;
            for (auto batch = 0; batch < trainingBatches; ++batch)
            {
                auto const xData = CppMLCTensorData(&_trainingDataX[batch * imageSize * batchSize],
                                                    batchSize * imageSize * sizeof(float));
                auto const yData = CppMLCTensorData(&_trainingDataY[batch * numberOfClasses * batchSize],
                                                    batchSize * numberOfClasses * sizeof(int));

                _trainingGraph.executeWithInputsData({std::make_pair("image", xData)},
                                                     {std::make_pair("label", yData)},
                                                     {},
                                                     batchSize,
                                                     eMLCExecutionOptions::Synchronous,
                                                     [&](auto& r, auto e, auto time) {
                    // VALIDATE
                    auto bufferOutput = std::vector<float>(batchSize * numberOfClasses);
                    _outputSoftmax.copyDataFromDeviceMemoryToBytes(bufferOutput.data(),
                                                                   batchSize * numberOfClasses * sizeof(float),
                                                                   false);

                    for (auto i = 0; i < batchSize; ++i)
                    {
                        const auto batchStartingPoint = i * numberOfClasses;
                        const auto predictionStartingPoint = (i * numberOfClasses) + (batch * batchSize * numberOfClasses);
                        const auto prediction = argmaxDecoding(&bufferOutput[batchStartingPoint],
                                                               &bufferOutput[batchStartingPoint + numberOfClasses]);
                        const auto label = oneHotDecoding(&_trainingDataY[predictionStartingPoint],
                                                          &_trainingDataY[predictionStartingPoint + numberOfClasses]);
                        if (prediction == label)
                        {
                            epochMatch += 1;
                        }
                        // print("\(i + (batch * batchSize)) -> Prediction: \(prediction) Label: \(label)")
                    }
                });
            }

            const auto epochAccuracy = float(epochMatch)/float(trainingSample);
            log(std::string("Epoch ") + std::to_string(epoch) + " Accuracy = " + std::to_string(epochAccuracy) + "%");
        }
    }

    void evaluateGraph(std::function<void(std::string)> const& log)
    {
        const auto testingSample = _testDataX.size() / imageSize;
        const auto testingBatches = testingSample / batchSize;

        auto _inferenceGraph = CppMLCInferenceGraph::graphWithGraphObjects({_graph});
        _inferenceGraph.addInputs({std::make_pair("image", _inputTensor)});
        _inferenceGraph.compileWithOptions({}, _device);

        // TESTING LOOP FOR A FULL EPOCH ON TESTING DATA
        auto match = 0;
        for (auto batch = 0; batch < testingBatches; ++batch)
        {
            auto const xData = CppMLCTensorData(&_testDataX[batch * imageSize * batchSize],
                                                batchSize * imageSize * sizeof(float));

            _inferenceGraph.executeWithInputsData({std::make_pair("image", xData)},
                                                  batchSize,
                                                  eMLCExecutionOptions::Synchronous,
                                                  [&] (auto& r, auto e, auto time) {
              auto bufferOutput = std::vector<float>(batchSize * numberOfClasses);
              r.copyDataFromDeviceMemoryToBytes(bufferOutput.data(),
                                                batchSize * numberOfClasses * sizeof(float),
                                                false);

                for (auto i = 0; i < batchSize; ++i)
                {
                    const auto batchStartingPoint = i * numberOfClasses;
                    const auto predictionStartingPoint = (i * numberOfClasses) + (batch * batchSize * numberOfClasses);
                    const auto prediction = argmaxDecoding(&bufferOutput[batchStartingPoint], &bufferOutput[batchStartingPoint + numberOfClasses]);
                    const auto label = oneHotDecoding(&_testDataY[predictionStartingPoint], &_testDataY[predictionStartingPoint + numberOfClasses]);

                    if (prediction == label)
                    {
                        match += 1;
                    }
                    // print("\(i + (batch * batchSize)) -> Prediction: \(prediction) Label: \(label)")
                }
            });
        }

        const auto accuracy = (float)match / (float)testingSample;
        log(std::string("Test Accuracy = ") + std::to_string(accuracy) + "%");
    }

#if 0
    auto predict(data: [[Float]]) -> int32_t
    {
        var image: [Float] = Array(data.joined())
        image.append(contentsOf: Array<Float>(repeating: 0.0, count: (batchSize - 1) * imageSize))


        auto const xData = image.withUnsafeBufferPointer { pointer in
            MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                    length: batchSize * imageSize * MemoryLayout<Float>.size)
        }

        auto prediction = -1
        inferenceGraph.execute(inputsData: ["image" : xData],
                               batchSize: batchSize,
                               options: [.synchronous]) { [self] (r, e, time) in
            let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: batchSize * numberOfClasses * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

            r!.copyDataFromDeviceMemory(toBytes: bufferOutput, length: batchSize * numberOfClasses * MemoryLayout<Float>.size, synchronizeWithDevice: false)

            let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: batchSize * numberOfClasses)
            let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: batchSize * numberOfClasses)
            let batchOutputArray = Array(float4Buffer)
            let firstImageOutput = Array(batchOutputArray[0..<numberOfClasses])

            prediction = argmaxDecoding(firstImageOutput)

            print(prediction)
        }

        return prediction
    }
#endif
    void getFileLine(std::string const& filePath, std::function<void(std::string const&)>&& process)
    {
        auto file = std::ifstream(filePath);
        std::string line;
        while (std::getline(file, line))
        {
            process(line);
        }
    }

    auto readDataSet(std::string const& filePath, std::function<void(uint32_t updateStatus)>&& updatingStatusCb) -> std::pair<std::vector<float>, std::vector<float>>
    {
        auto file = std::ifstream(filePath, std::ios::binary);

        auto X = std::vector<float>();
        auto Y = std::vector<float>();

        auto iterations = 20;
        auto iteration = 0;
        auto iterationList = std::vector<std::vector<std::string>>(iterations);

        getFileLine(filePath, [&](auto const& line) {
            iterationList[iteration].push_back(line);
            iteration = (iteration + 1) % iterations;
        });
        auto count = 0;
        for(auto& iterationItem : iterationList)
        {
            for (auto& line : iterationItem)
            {
//              let sample = line.split(separator: ",").compactMap({Int($0)});
//                Y.emplace_back(oneHotEncoding(sample[0]));
//              X.append(contentsOf: sample[1...self.imageSize].map{Float($0) / Float(255.0)});
                updatingStatusCb(++count);
            }
        }
        return std::make_pair(X, Y);
    }

private:
    CppMLCDevice _device;
    CppMLCTensor _inputTensor;
    CppMLCTensor _dense1WeightsTensor;
    CppMLCTensor _dense1BiasesTensor;
    CppMLCTensor _dense2WeightsTensor;
    CppMLCTensor _dense2BiasesTensor;
    CppMLCTensor _lossLabelTensor;
    CppMLCGraph _graph;
    CppMLCConvolutionDescriptor _convolutionDescriptor;
    CppMLCFullyConnectedLayer _fullyConnectedLayer1;
    CppMLCTensor _dense1;
    CppMLCActivationDescriptor _activationDescriptor;
    CppMLCActivationLayer _activationLayer;
    CppMLCTensor _relu1;
    CppMLCFullyConnectedLayer _fullyConnectedLayer2;
    CppMLCTensor _dense2;
    CppMLCTensor _outputSoftmax;
    CppMLCTrainingGraph _trainingGraph;
    std::vector<float> _testDataX;
    std::vector<float> _testDataY;
    std::vector<float> _trainingDataX;
    std::vector<float> _trainingDataY;
    uint32_t epochs = 5;
};

int main()
{
    auto device = CppMLCDevice::deviceWithType(eMLCDeviceType::Any);
    if (device.getType() == eMLCDeviceType::GPU) {
        std::cout << "GPU Enabled!" << std::endl;
    } else {
        std::cout << "CPU Enabled!" << std::endl;
    }

    auto tensorDescriptor1 = CppMLCTensorDescriptor{{6, 1}, eMLCDataType::Float32};
    auto tensor1 = CppMLCTensor(tensorDescriptor1);
    auto buffer1 = std::vector<float>{1, 2, 3, 4, 5, 6};
    auto data1 = CppMLCTensorData(buffer1.data(), buffer1.size() * sizeof(float));
    std::cout << "tensor1: "<< tensor1 << std::endl;

    auto tensorDescriptor2 = CppMLCTensorDescriptor{{6, 1}, eMLCDataType::Float32};
    auto tensor2 = CppMLCTensor(tensorDescriptor2);
    auto buffer2 = std::vector<float>{7, 8, 9, 10, 11, 12};
    auto data2 = CppMLCTensorData(buffer2.data(), buffer2.size() * sizeof(float));
    std::cout << "tensor2: "<< tensor2 << "} \n" << std::endl;

    auto tensorDescriptor3 = CppMLCTensorDescriptor{{6, 1}, eMLCDataType::Float32};
    auto tensor3 = CppMLCTensor(tensorDescriptor3);
    auto buffer3 = std::vector<float>{6, 5, 4, 3, 2, 1};
    auto data3 = CppMLCTensorData(buffer3.data(), buffer3.size() * sizeof(float));
    std::cout << "tensor3: { "<< tensor3 << "} \n" << std::endl;

    auto g = CppMLCGraph::graph();
    auto arithmeticLayer = CppMLCArithmeticLayer::layerWithOperation(eMLCArithmeticOperation::Add);
    auto tensor1_tensor2 = std::vector<CppMLCTensor*>{&tensor1, &tensor2};
    auto tensor1plus2 = g.nodeWithLayerPtr(arithmeticLayer, tensor1_tensor2);
    auto arithmeticLayer1 = CppMLCArithmeticLayer::layerWithOperation(eMLCArithmeticOperation::Add);
    auto tensor1plus2_tensor3 = std::vector<CppMLCTensor*>{ &tensor1plus2, &tensor3 };
    g.nodeWithLayerPtr(arithmeticLayer1, tensor1plus2_tensor3);

    auto i = CppMLCInferenceGraph::graphWithGraphObjects({ g });
    i.addInputs({{"data1", tensor1}, {"data2", tensor2}, {"data3", tensor3}});
    i.compileWithOptions(eMLCGraphCompilationOptions::DebugLayers, device);
    auto inputsData = std::map<std::string, CppMLCTensorData*>{{"data1", &data1}, {"data2", &data2}, {"data3", &data3}};
    i.executeWithInputsData(inputsData,
                            0,
                            eMLCExecutionOptions::None,
                            [](CppMLCTensor const& r, std::string e, std::string time) {
        std::cout << "Error: " << e << std::endl;
        std::cout << "Time: " << time << std::endl;

        auto buffer3 = std::vector<float>(6);
        r.copyDataFromDeviceMemoryToBytes(buffer3.data(), 6 * sizeof(float), false);

        for (auto item : buffer3)
        {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    });

#if 0 
    MNISTTrain mnistTrain;
    mnistTrain.readDataSet("mnist_test.csv",
                           [](uint32_t progress){
        std::cout << "progress: " << progress << std::endl;
    });
    mnistTrain.trainGraph([](auto& str) {
        std::cout << str << std::endl;
    });
#endif
    return 0;
}
