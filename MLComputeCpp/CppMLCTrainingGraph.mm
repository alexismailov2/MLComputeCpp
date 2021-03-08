#include "CppMLCTrainingGraph.h"

#include "CppMLCTensor.h"
#include "CppMLCTypesPrivate.h"
#include "CppMLCLayer.h"
#include "CppMLCOptimizer.h"
#include "CppMLCDevice.h"

#import <MLCompute/MLCTensor.h>
#import <MLCompute/MLCDevice.h>
#import <MLCompute/MLCGraph.h>
#import <MLCompute/MLCTrainingGraph.h>
#import <MLCompute/MLCOptimizer.h>
#import <MLCompute/MLCLayer.h>
#import <MLCompute/MLCOptimizer.h>

auto CppMLCTrainingGraph::optimizer() -> CppMLCOptimizer
{
    return CppMLCOptimizer{((MLCTrainingGraph*)self).optimizer};
}

auto CppMLCTrainingGraph::deviceMemorySize() -> uint32_t
{
    return (uint32_t)((MLCTrainingGraph*)self).deviceMemorySize;
}

auto CppMLCTrainingGraph::graphWithGraphObjects(std::vector<CppMLCGraph> const& graphObjects,
                                                CppMLCLayer& lossLayer,
                                                CppMLCOptimizer& optimizer) -> CppMLCTrainingGraph
{
    return CppMLCTrainingGraph{[MLCTrainingGraph graphWithGraphObjects:CppMLCTypesPrivate::toNSArray(graphObjects)
                                                             lossLayer:(MLCLayer*)lossLayer.self
                                                             optimizer:(MLCOptimizer*)optimizer.self]};

}

bool CppMLCTrainingGraph::addInputs(std::map<std::string, CppMLCTensor> const& inputs,
                                    std::map<std::string, CppMLCTensor> const& lossLabels)
{
    return [(MLCTrainingGraph*)self addInputs:CppMLCTypesPrivate::toNSDictionary(inputs)
                                   lossLabels:CppMLCTypesPrivate::toNSDictionary(lossLabels)] == YES;
}

bool CppMLCTrainingGraph::addInputs(std::map<std::string, CppMLCTensor> const& inputs,
                                    std::map<std::string, CppMLCTensor> const& lossLabels,
                                    std::map<std::string, CppMLCTensor> const& lossLabelWeights)
{
    return [(MLCTrainingGraph*)self addInputs:CppMLCTypesPrivate::toNSDictionary(inputs)
                                   lossLabels:CppMLCTypesPrivate::toNSDictionary(lossLabels)
                             lossLabelWeights:CppMLCTypesPrivate::toNSDictionary(lossLabelWeights)] == YES;
}

bool CppMLCTrainingGraph::addOutputs(std::map<std::string, CppMLCTensor> const& outputs)
{
    return [(MLCTrainingGraph*)self addOutputs:CppMLCTypesPrivate::toNSDictionary(outputs)] == YES;
}

bool CppMLCTrainingGraph::stopGradientForTensors(std::vector<CppMLCTensor> const& tensors)
{
    return [(MLCTrainingGraph*)self stopGradientForTensors:CppMLCTypesPrivate::toNSArray(tensors)] == YES;
}

bool CppMLCTrainingGraph::compileWithOptions(eMLCGraphCompilationOptions options, CppMLCDevice &device)
{
    return [(MLCTrainingGraph*)self compileWithOptions:toNative(options)
                                                device:(MLCDevice*)device.self] == YES;
}

bool CppMLCTrainingGraph::compileOptimizer(CppMLCOptimizer& optimizer)
{
    return [(MLCTrainingGraph*)self compileOptimizer:(MLCOptimizer*)optimizer.self] == YES;
}

bool CppMLCTrainingGraph::linkWithGraphs(std::vector<CppMLCTrainingGraph> const& graphs)
{
    return [(MLCTrainingGraph*)self linkWithGraphs:CppMLCTypesPrivate::toNSArray(graphs)] == YES;
}

auto CppMLCTrainingGraph::gradientTensorForInput(CppMLCTensor& input) -> CppMLCTensor
{
    return CppMLCTensor{[(MLCTrainingGraph*)self gradientTensorForInput:(MLCTensor*)input.self]};
}

auto CppMLCTrainingGraph::sourceGradientTensorsForLayer(CppMLCLayer& layer) -> std::vector<CppMLCTensor>
{
    return CppMLCTypesPrivate::MLCTensorArrayToVector([(MLCTrainingGraph*)self sourceGradientTensorsForLayer:(MLCLayer*)layer.self]);
}

auto CppMLCTrainingGraph::resultGradientTensorsForLayer(CppMLCLayer& layer) -> std::vector<CppMLCTensor>
{
    return CppMLCTypesPrivate::MLCTensorArrayToVector([(MLCTrainingGraph*)self resultGradientTensorsForLayer:(MLCLayer*)layer.self]);
}

auto CppMLCTrainingGraph::gradientDataForParameter(CppMLCTensor& parameter, CppMLCLayer& layer) -> std::vector<float>
{
    return CppMLCTypesPrivate::NSDataToVectorFloat([(MLCTrainingGraph*)self gradientDataForParameter:(MLCTensor*)parameter.self
                                                                                               layer:(MLCLayer*)layer.self]);
}

auto CppMLCTrainingGraph::allocateUserGradientForTensor(CppMLCTensor &tensor) -> CppMLCTensor
{
    return [(MLCTrainingGraph*)self allocateUserGradientForTensor:(MLCTensor*)tensor.self];
}

bool CppMLCTrainingGraph::executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                                std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                                                std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                                                uint32_t batchSize,
                                                eMLCExecutionOptions options,
                                                MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeWithInputsData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                           lossLabelsData:CppMLCTypesPrivate::toNSDictionary(lossLabelsData)
                                     lossLabelWeightsData:CppMLCTypesPrivate::toNSDictionary(lossLabelWeightsData)
                                                batchSize:(NSUInteger)batchSize
                                                  options:toNative(options)
                                        completionHandler:completionHandler] == YES;
}

bool CppMLCTrainingGraph::executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                                std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                                                std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                                                std::map<std::string, CppMLCTensorData> const& outputsData,
                                                uint32_t batchSize,
                                                eMLCExecutionOptions options,
                                                MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeWithInputsData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                           lossLabelsData:CppMLCTypesPrivate::toNSDictionary(lossLabelsData)
                                     lossLabelWeightsData:CppMLCTypesPrivate::toNSDictionary(lossLabelWeightsData)
                                              outputsData:CppMLCTypesPrivate::toNSDictionary(outputsData)
                                                batchSize:(NSUInteger)batchSize
                                                  options:toNative(options)
                                        completionHandler:completionHandler] == YES;
}

bool CppMLCTrainingGraph::executeForwardWithBatchSize(uint32_t batchSize,
                                                      eMLCExecutionOptions options,
                                                      MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeForwardWithBatchSize:(NSUInteger)batchSize
                                                        options:toNative(options)
                                               completionHandler:completionHandler] == YES;

}

bool CppMLCTrainingGraph::executeForwardWithBatchSize(uint32_t batchSize,
                                                      eMLCExecutionOptions options,
                                                      std::map<std::string, CppMLCTensorData> const& outputsData,
                                                      MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeForwardWithBatchSize:(NSUInteger)batchSize
                                                        options:toNative(options)
                                                     outputsData:CppMLCTypesPrivate::toNSDictionary(outputsData)
                                               completionHandler:completionHandler] == YES;

}

bool CppMLCTrainingGraph::executeGradientWithBatchSize(uint32_t batchSize,
                                                       eMLCExecutionOptions options,
                                                       MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeGradientWithBatchSize:(NSUInteger)batchSize
                                                         options:toNative(options)
                                               completionHandler:completionHandler] == YES;
}

bool CppMLCTrainingGraph::executeGradientWithBatchSize(uint32_t batchSize,
                                                       eMLCExecutionOptions options,
                                                       std::map<std::string, CppMLCTensorData> outputsData,
                                                       MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeGradientWithBatchSize:(NSUInteger)batchSize
                                                         options:toNative(options)
                                                     outputsData:CppMLCTypesPrivate::toNSDictionary(outputsData)
                                                    completionHandler:completionHandler] == YES;
}

bool CppMLCTrainingGraph::executeOptimizerUpdateWithOptions(eMLCExecutionOptions options,
                                                            MLCGraphCompletionHandler completionHandler)
{
    return [(MLCTrainingGraph*)self executeOptimizerUpdateWithOptions:toNative(options)
                                                    completionHandler:completionHandler] == YES;
}

void CppMLCTrainingGraph::synchronizeUpdates()
{
    [(MLCTrainingGraph*)self synchronizeUpdates];
}

bool CppMLCTrainingGraph::setTrainingTensorParameters(std::vector<CppMLCTensorParameter> const& parameters)
{
    return [(MLCTrainingGraph*)self setTrainingTensorParameters:CppMLCTypesPrivate::toNSArray(parameters)] == YES;
}

bool CppMLCTrainingGraph::bindOptimizerData(std::vector<CppMLCTensorData> const& data,
                                            std::vector<CppMLCTensorOptimizerDeviceData> const& deviceData,
                                            CppMLCTensor &tensor)
{
    return [(MLCTrainingGraph*)self bindOptimizerData:CppMLCTypesPrivate::toNSArray(data)
                                           deviceData:CppMLCTypesPrivate::toNSArray(deviceData)
                                           withTensor:(MLCTensor*)tensor.self] == YES;
}

CppMLCTrainingGraph::CppMLCTrainingGraph(void *self)
    : CppMLCGraph(self)
{
}
