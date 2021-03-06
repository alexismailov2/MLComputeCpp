#include "CppMLCYOLOLossLayer.h"

#import <MLCompute/MLCYOLOLossLayer.h>

auto CppMLCYOLOLossLayer::yoloLossDescriptor() -> CppMLCYOLOLossDescriptor
{
    return CppMLCYOLOLossDescriptor{((MLCYOLOLossLayer*)self).yoloLossDescriptor};
}

auto CppMLCYOLOLossLayer::layerWithDescriptor(CppMLCYOLOLossDescriptor& lossDescriptor) -> CppMLCYOLOLossLayer
{
    return CppMLCYOLOLossLayer{[MLCYOLOLossLayer layerWithDescriptor:(MLCYOLOLossDescriptor*)lossDescriptor.self]};
}

CppMLCYOLOLossLayer::CppMLCYOLOLossLayer(void *self)
    : CppMLCLossLayer(self)
{
}
