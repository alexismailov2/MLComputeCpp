#include "CppMLCFullyConnectedLayer.h"
#include "CppMLCTensor.h"
#include "CppMLCTensorParameter.h"

#include <MLCompute/MLCFullyConnectedLayer.h>

auto CppMLCFullyConnectedLayer::descriptor() -> CppMLCConvolutionDescriptor
{
    return CppMLCConvolutionDescriptor{((MLCFullyConnectedLayer*)getSelf()).descriptor};
}

auto CppMLCFullyConnectedLayer::weights() -> CppMLCTensor
{
    return CppMLCTensor{((MLCFullyConnectedLayer*)getSelf()).weights};
}

auto CppMLCFullyConnectedLayer::biases() -> CppMLCTensor
{
    return CppMLCTensor{((MLCFullyConnectedLayer*)getSelf()).biases};
}

auto CppMLCFullyConnectedLayer::weightsParameter() -> CppMLCTensorParameter
{
    return CppMLCTensorParameter{((MLCFullyConnectedLayer*)getSelf()).weightsParameter};
}

auto CppMLCFullyConnectedLayer::biasesParameter() -> CppMLCTensorParameter
{
    return CppMLCTensorParameter{((MLCFullyConnectedLayer*)getSelf()).biasesParameter};
}

auto CppMLCFullyConnectedLayer::layerWithWeights(CppMLCTensor const& weights,
                                                 CppMLCTensor const& biases,
                                                 CppMLCConvolutionDescriptor const& descriptor) -> CppMLCFullyConnectedLayer
{
    [(id)weights.self retain];
    [(id)biases.self retain];
    [(id)descriptor.self retain];
    return CppMLCFullyConnectedLayer{[MLCFullyConnectedLayer layerWithWeights:(MLCTensor*)weights.self
                                                                       biases:(MLCTensor*)biases.self
                                                                   descriptor:(MLCConvolutionDescriptor*)descriptor.self]};
}

CppMLCFullyConnectedLayer::CppMLCFullyConnectedLayer(void *self)
    : CppMLCLayer(self)
    //, self{self}
{
    //[(id)self retain];
}

CppMLCFullyConnectedLayer::~CppMLCFullyConnectedLayer()
{
    //[(id)self release];
}
