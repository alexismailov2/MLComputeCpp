#include "CppMLCPaddingLayer.h"
#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCPaddingLayer.h>

auto CppMLCPaddingLayer::paddingType() -> eMLCPaddingType
{
    return MLCPaddingTypeToCpp(((MLCPaddingLayer*)self).paddingType);
}

auto CppMLCPaddingLayer::paddingLeft() -> uint32_t
{
    return (uint32_t)((MLCPaddingLayer*)self).paddingLeft;
}

auto CppMLCPaddingLayer::paddingRight() -> uint32_t
{
    return (uint32_t)((MLCPaddingLayer*)self).paddingRight;
}

auto CppMLCPaddingLayer::paddingTop() -> uint32_t
{
    return (uint32_t)((MLCPaddingLayer*)self).paddingTop;
}

auto CppMLCPaddingLayer::paddingBottom() -> uint32_t
{
    return (uint32_t)((MLCPaddingLayer*)self).paddingBottom;
}

auto CppMLCPaddingLayer::constantValue() -> float
{
    return ((MLCPaddingLayer*)self).constantValue;
}

auto CppMLCPaddingLayer::layerWithReflectionPadding(std::vector<uint32_t> const& padding) -> CppMLCPaddingLayer
{
    return CppMLCPaddingLayer{[MLCPaddingLayer layerWithReflectionPadding:CppMLCTypesPrivate::toNSArray(padding)]};
}

auto CppMLCPaddingLayer::layerWithSymmetricPadding(std::vector<uint32_t> const& padding) -> CppMLCPaddingLayer
{
    return CppMLCPaddingLayer{[MLCPaddingLayer layerWithSymmetricPadding:CppMLCTypesPrivate::toNSArray(padding)]};
}

auto CppMLCPaddingLayer::layerWithZeroPadding(std::vector<uint32_t> const& padding) -> CppMLCPaddingLayer
{
    return CppMLCPaddingLayer{[MLCPaddingLayer layerWithZeroPadding:CppMLCTypesPrivate::toNSArray(padding)]};
}

auto CppMLCPaddingLayer::layerWithConstantPadding(std::vector<uint32_t> const& padding,
                                                  float constantValue) -> CppMLCPaddingLayer
{
    return CppMLCPaddingLayer{[MLCPaddingLayer layerWithConstantPadding:CppMLCTypesPrivate::toNSArray(padding)
                                                          constantValue:constantValue]};

}

CppMLCPaddingLayer::CppMLCPaddingLayer(void *self)
    : CppMLCLayer{self}
{
}
