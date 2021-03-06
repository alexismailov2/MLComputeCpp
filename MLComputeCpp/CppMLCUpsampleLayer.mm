#include "CppMLCUpsampleLayer.h"
#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCUpsampleLayer.h>

auto CppMLCUpsampleLayer::shape() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCUpsampleLayer*)self).shape);
}

auto CppMLCUpsampleLayer::sampleMode() -> eMLCSampleMode
{
    return MLCSampleModeToCpp(((MLCUpsampleLayer*)self).sampleMode);
}

bool CppMLCUpsampleLayer::alignsCorners()
{
    return ((MLCUpsampleLayer*)self).alignsCorners == YES;
}

auto CppMLCUpsampleLayer::layerWithShape(std::vector<uint32_t> shape) -> CppMLCUpsampleLayer
{
    return CppMLCUpsampleLayer{[MLCUpsampleLayer layerWithShape:CppMLCTypesPrivate::toNSArray(shape)]};
}

auto CppMLCUpsampleLayer::layerWithShape(std::vector<uint32_t> shape,
                                         eMLCSampleMode sampleMode,
                                         bool alignsCorners) -> CppMLCUpsampleLayer
{
    return CppMLCUpsampleLayer{[MLCUpsampleLayer layerWithShape:CppMLCTypesPrivate::toNSArray(shape)
                                                     sampleMode:toNative(sampleMode)
                                                  alignsCorners:alignsCorners ? YES : NO]};
}

CppMLCUpsampleLayer::CppMLCUpsampleLayer(void* self)
    : CppMLCLayer(self)
{
}
