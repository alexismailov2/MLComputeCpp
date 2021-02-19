#include "CppMLCFullyConnectedLayer.h"
#include "CppMLCTensor.h"
#include "CppMLCTensorParameter.h"

#include <MLCompute/MLCFullyConnectedLayer.h>

auto CppMLCFullyConnectedLayer::descriptor() -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{((MLCFullyConnectedLayer*)self).descriptor};
}

auto CppMLCFullyConnectedLayer::weights() -> CppMLCTensor {
    return CppMLCTensor{((MLCFullyConnectedLayer*)self).weights};
}

auto CppMLCFullyConnectedLayer::biases() -> CppMLCTensor {
    return CppMLCTensor{((MLCFullyConnectedLayer*)self).biases};
}

auto CppMLCFullyConnectedLayer::weightsParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCFullyConnectedLayer*)self).weightsParameter};
}

auto CppMLCFullyConnectedLayer::biasesParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCFullyConnectedLayer*)self).biasesParameter};
}

auto CppMLCFullyConnectedLayer::layerWithWeights(const CppMLCTensor &weights, const CppMLCTensor &biases,
                                                 const CppMLCConvolutionDescriptor &descriptor) -> CppMLCFullyConnectedLayer {
    return CppMLCFullyConnectedLayer{[MLCFullyConnectedLayer layerWithWeights:(MLCTensor*)weights.self
                                                                       biases:(MLCTensor*)biases.self
                                                                   descriptor:(MLCConvolutionDescriptor*)descriptor.self]};
}

CppMLCFullyConnectedLayer::CppMLCFullyConnectedLayer(void *self)
    : CppMLCLayer(self)
    , self{self}
{
}
