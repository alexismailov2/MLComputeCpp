#include "CppMLCLayer.h"
#include "CppMLCDevice.h"
#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCLayer.h>

#include <iostream>

auto CppMLCLayer::getLayerID() const -> uint64_t
{
    return ((MLCLayer*)self).layerID;
}

auto CppMLCLayer::getLabel() const -> std::string
{
    return std::string([((MLCLayer*)self).label UTF8String]);
}

bool CppMLCLayer::isDebuggingEnabled() const
{
    return ((MLCLayer*)self).isDebuggingEnabled == YES;
}

CppMLCLayer::CppMLCLayer(void *self)
    : self{self}
{
    //[(id)self retain];
}

CppMLCLayer::~CppMLCLayer()
{
    //[(id)self release];
}

bool CppMLCLayer::supportsDataType(eMLCDataType dataType, const CppMLCDevice &device)
{
    return [MLCLayer supportsDataType:toNative(dataType) onDevice:((MLCDevice*)device.self)] == YES;
}

auto CppMLCLayer::getSelf() -> void*
{
    return self;
}

std::ostream& operator<<(std::ostream& out, CppMLCLayer const& layer)
{
    out << "layer: { \n"
        << "layerId: " << layer.getLayerID() << "\n"
        << "label: " << layer.getLabel() << "\n}" << std::endl;
    return out;
}