#include "CppMLCConcatenationLayer.h"

#import <MLCompute/MLCConcatenationLayer.h>

auto CppMLCConcatenationLayer::dimension() -> uint32_t {
    return (uint32_t)((MLCConcatenationLayer*)self).dimension;
}

CppMLCConcatenationLayer CppMLCConcatenationLayer::layer() {
    return CppMLCConcatenationLayer{[MLCConcatenationLayer layer]};
}

CppMLCConcatenationLayer CppMLCConcatenationLayer::layerWithDimension(uint32_t dimension) {
    return CppMLCConcatenationLayer{[MLCConcatenationLayer layerWithDimension:(NSUInteger)dimension]};
}

CppMLCConcatenationLayer::CppMLCConcatenationLayer(void *self)
    : CppMLCLayer(self)
    , self{self}
{
}
