#include "CppMLCPoolingLayer.h"

#include "CppMLCPoolingDescriptor.h"

#import <MLCompute/MLCPoolingLayer.h>

auto CppMLCPoolingLayer::descriptor() -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{((MLCPoolingLayer*)self).descriptor};
}

auto CppMLCPoolingLayer::layerWithDescriptor(CppMLCPoolingDescriptor& descriptor) -> CppMLCPoolingLayer
{
    return CppMLCPoolingLayer{[MLCPoolingLayer layerWithDescriptor:(MLCPoolingDescriptor*)descriptor.self]};
}

CppMLCPoolingLayer::CppMLCPoolingLayer(void *self)
    : CppMLCLayer(self)
{

}
