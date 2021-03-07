#include "CppMLCReshapeLayer.h"

#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCReshapeLayer.h>

auto CppMLCReshapeLayer::layerWithShape(std::vector<uint32_t> const& shape) -> CppMLCReshapeLayer
{
    return CppMLCReshapeLayer{[MLCReshapeLayer layerWithShape:CppMLCTypesPrivate::toNSArray(shape)]};
}

CppMLCReshapeLayer::CppMLCReshapeLayer(void *self)
    : CppMLCLayer(self)
{
}
