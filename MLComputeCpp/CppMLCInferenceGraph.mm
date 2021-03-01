#include "CppMLCInferenceGraph.h"

#import <MLCompute/MLCDevice.h>
#import <MLCompute/MLCGraph.h>
#import <MLCompute/MLCInferenceGraph.h>

auto CppMLCInferenceGraph::deviceMemorySize() -> uint32_t {
    return (uint32_t)((MLCInferenceGraph*)self).deviceMemorySize;
}

auto CppMLCInferenceGraph::graphWithGraphObjects(std::vector<CppMLCGraph> const& graphObjects) -> CppMLCInferenceGraph {
    return CppMLCInferenceGraph{[MLCInferenceGraph graphWithGraphObjects:CppMLCTypesPrivate::toNSArray(graphObjects)]};
}

bool CppMLCInferenceGraph::addInputs(std::map<std::string, CppMLCTensor> const& inputs) {
    return [(MLCInferenceGraph*)self addInputs:CppMLCTypesPrivate::toNSDictionary(inputs)] == YES;
}

bool CppMLCInferenceGraph::addInputs(std::map<std::string, CppMLCTensor> const& inputs,
                                     std::map<std::string, CppMLCTensor> const& lossLabels,
                                     std::map<std::string, CppMLCTensor> const& lossLabelWeights) {
    return [(MLCInferenceGraph*)self addInputs:CppMLCTypesPrivate::toNSDictionary(inputs)
                                    lossLabels:CppMLCTypesPrivate::toNSDictionary(lossLabels)
                              lossLabelWeights:CppMLCTypesPrivate::toNSDictionary(lossLabelWeights)] == YES;
}

bool CppMLCInferenceGraph::addOutputs(std::map<std::string, CppMLCTensor> const& outputs) {
    return [(MLCInferenceGraph*)self addOutputs:CppMLCTypesPrivate::toNSDictionary(outputs)] == YES;
}

bool CppMLCInferenceGraph::compileWithOptions(eMLCGraphCompilationOptions options,
                                              CppMLCDevice const& device) {
    return [(MLCInferenceGraph*)self compileWithOptions:toNative(options)
                                                 device:(MLCDevice*)device.self] == YES;
}

bool CppMLCInferenceGraph::linkWithGraphs(std::vector<CppMLCInferenceGraph> const& graphs) {
    return [(MLCInferenceGraph*)self linkWithGraphs:CppMLCTypesPrivate::toNSArray(graphs)] == YES;
}

// TODO: Hack for compilation
void HandlerCb(MLCTensor __autoreleasing * _Nullable resultTensor, NSError * _Nullable error, NSTimeInterval executionTime) {

}

bool CppMLCInferenceGraph::executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                                 uint32_t batchSize,
                                                 eMLCExecutionOptions options,
                                                 CppMLCGraphCompletionHandler completionHandler) {
    return [(MLCInferenceGraph *) self executeWithInputsData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                                   batchSize:(NSUInteger) batchSize
                                                     options:toNative(options)
                                           completionHandler:(MLCGraphCompletionHandler)HandlerCb] == YES;
}

bool CppMLCInferenceGraph::executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                                 std::map<std::string, CppMLCTensorData> const& outputsData,
                                                 uint32_t batchSize,
                                                 eMLCExecutionOptions options,
                                                 CppMLCGraphCompletionHandler completionHandler) {
    return [(MLCInferenceGraph*)self executeWithInputsData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                               outputsData:CppMLCTypesPrivate::toNSDictionary(outputsData)
                                                 batchSize:(NSUInteger)batchSize
                                                   options:toNative(options)
                                         completionHandler:(MLCGraphCompletionHandler)HandlerCb] == YES;
}

bool CppMLCInferenceGraph::executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                                 std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                                                 std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                                                 uint32_t batchSize,
                                                 eMLCExecutionOptions options,
                                                 CppMLCGraphCompletionHandler completionHandler) {
    return [(MLCInferenceGraph*)self executeWithInputsData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                            lossLabelsData:CppMLCTypesPrivate::toNSDictionary(lossLabelsData)
                                      lossLabelWeightsData:CppMLCTypesPrivate::toNSDictionary(lossLabelWeightsData)
                                                 batchSize:(NSUInteger)batchSize
                                                   options:toNative(options)
                                         completionHandler:(MLCGraphCompletionHandler)HandlerCb] == YES;
}

bool CppMLCInferenceGraph::executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                                                 std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                                                 std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                                                 std::map<std::string, CppMLCTensorData> const& outputsData,
                                                 uint32_t batchSize,
                                                 eMLCExecutionOptions options,
                                                 CppMLCGraphCompletionHandler completionHandler) {
    return [(MLCInferenceGraph*)self executeWithInputsData:CppMLCTypesPrivate::toNSDictionary(inputsData)
                                            lossLabelsData:CppMLCTypesPrivate::toNSDictionary(lossLabelsData)
                                      lossLabelWeightsData:CppMLCTypesPrivate::toNSDictionary(lossLabelWeightsData)
                                               outputsData:CppMLCTypesPrivate::toNSDictionary(outputsData)
                                                 batchSize:(NSUInteger)batchSize
                                                   options:toNative(options)
                                         completionHandler:(MLCGraphCompletionHandler)HandlerCb] == YES;
}

CppMLCInferenceGraph::CppMLCInferenceGraph(void *self)
    : CppMLCGraph(self)
    , self{self}
{
}
