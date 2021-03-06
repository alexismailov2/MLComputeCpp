#include "CppMLCTransposeLayer.h"
#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCTransposeLayer.h>

auto CppMLCTransposeLayer::dimensions() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCTransposeLayer*)self).dimensions);
}

auto CppMLCTransposeLayer::layerWithDimensions(std::vector<uint32_t> dimensions) -> CppMLCTransposeLayer
{
    return CppMLCTransposeLayer{[MLCTransposeLayer layerWithDimensions:CppMLCTypesPrivate::toNSArray(dimensions)]};
}

CppMLCTransposeLayer::CppMLCTransposeLayer(void *self)
    : CppMLCLayer(self)
{
}
