#include "CppMLCEmbeddingLayer.h"
#include "CppMLCTensor.h"

#include <MLCompute/MLCEmbeddingLayer.h>

auto CppMLCEmbeddingLayer::descriptor() -> CppMLCEmbeddingDescriptor {
    return CppMLCEmbeddingDescriptor{((MLCEmbeddingLayer*)self).descriptor};
}

auto CppMLCEmbeddingLayer::weights() -> CppMLCTensor {
    return CppMLCTensor{((MLCEmbeddingLayer*)self).weights};
}

auto CppMLCEmbeddingLayer::weightsParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCEmbeddingLayer*)self).weightsParameter};
}

auto CppMLCEmbeddingLayer::layerWithDescriptor(CppMLCEmbeddingDescriptor const& descriptor,
                                               CppMLCTensor const& weights) -> CppMLCEmbeddingLayer {
    return CppMLCEmbeddingLayer{[MLCEmbeddingLayer layerWithDescriptor:(MLCEmbeddingDescriptor*)descriptor.self
                                                               weights:(MLCTensor*)weights.self]};
}

CppMLCEmbeddingLayer::CppMLCEmbeddingLayer(void *self)
    : CppMLCLayer(self)
    , self{self}
{
}
