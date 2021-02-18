#include "CppMLCConvolutionLayer.h"
#include "CppMLCConvolutionDescriptor.h"
#include "CppMLCTensor.h"
#include "CppMLCTensorParameter.h"

#include <MLCompute/MLCConvolutionLayer.h>

auto CppMLCConvolutionLayer::descriptor() -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{((MLCConvolutionLayer*)self).descriptor};
}

auto CppMLCConvolutionLayer::weights() -> CppMLCTensor {
    return CppMLCTensor{((MLCConvolutionLayer*)self).weights};
}

auto CppMLCConvolutionLayer::biases() -> CppMLCTensor {
    return CppMLCTensor{((MLCConvolutionLayer*)self).biases};
}

auto CppMLCConvolutionLayer::weightsParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCConvolutionLayer*)self).weightsParameter};
}

auto CppMLCConvolutionLayer::biasesParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCConvolutionLayer*)self).biasesParameter};
}

auto CppMLCConvolutionLayer::layerWithWeights(const CppMLCTensor &weights,
                                              const CppMLCTensor &biases,
                                              const CppMLCConvolutionDescriptor &descriptor) -> CppMLCConvolutionLayer {
    return CppMLCConvolutionLayer{[MLCConvolutionLayer layerWithWeights:(MLCTensor*)weights.self
                                                                 biases:(MLCTensor*)biases.self
                                                             descriptor:(MLCConvolutionDescriptor*)descriptor.self]};
}

CppMLCConvolutionLayer::CppMLCConvolutionLayer(void *self)
    : CppMLCLayer{self}
    , self{self}
{
}
