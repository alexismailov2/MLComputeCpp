#include "CppMLCSoftmaxLayer.h"
#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCSoftmaxLayer.h>

auto CppMLCSoftmaxLayer::operation() -> eMLCSoftmaxOperation
{
    return MLCSoftmaxOperationToCpp(((MLCSoftmaxLayer*)self).operation);
}

auto CppMLCSoftmaxLayer::dimension() -> uint64_t
{
    return (uint64_t)((MLCSoftmaxLayer*)self).dimension;
}

auto CppMLCSoftmaxLayer::layerWithOperation(eMLCSoftmaxOperation operation) -> CppMLCSoftmaxLayer
{
    return CppMLCSoftmaxLayer{[MLCSoftmaxLayer layerWithOperation:toNative(operation)]};
}

auto CppMLCSoftmaxLayer::layerWithOperation(eMLCSoftmaxOperation operation, uint64_t dimension) -> CppMLCSoftmaxLayer
{
    return CppMLCSoftmaxLayer{[MLCSoftmaxLayer layerWithOperation:toNative(operation)
                                                        dimension:(NSUInteger)dimension]};
}

CppMLCSoftmaxLayer::CppMLCSoftmaxLayer(void* self)
    : CppMLCLayer(self)
{
}
