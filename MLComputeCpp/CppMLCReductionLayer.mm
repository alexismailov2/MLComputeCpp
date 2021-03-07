#include "CppMLCReductionLayer.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCReductionLayer.h>

auto CppMLCReductionLayer::reductionType() -> eMLCReductionType
{
    return MLCReductionTypeToCpp(((MLCReductionLayer*)self).reductionType);
}

auto CppMLCReductionLayer::dimension() -> uint32_t
{
    return (uint32_t)((MLCReductionLayer*)self).dimension;
}

auto CppMLCReductionLayer::layerWithReductionType(eMLCReductionType reductionType,
                                                  uint32_t dimension) -> CppMLCReductionLayer
{
    return CppMLCReductionLayer{[MLCReductionLayer layerWithReductionType:toNative(reductionType)
                                                                dimension:(NSUInteger)dimension]};
}

CppMLCReductionLayer::CppMLCReductionLayer(void *self)
    : CppMLCLayer(self)
{
}
