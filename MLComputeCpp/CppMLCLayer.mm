#import "CppMLCLayer.h"
#import "CppMLCDevice.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCLayer.h>

auto CppMLCLayer::getLayerID() -> uint64_t {
    return ((MLCLayer*)self).layerID;
}

auto CppMLCLayer::getLabel() -> std::string {
    return std::string([((MLCLayer*)self).label UTF8String]);
}

bool CppMLCLayer::isDebuggingEnabled() {
    return ((MLCLayer*)self).isDebuggingEnabled == YES;
}

CppMLCLayer::CppMLCLayer(void *self)
    : self{self}
{
}

bool CppMLCLayer::supportsDataType(eMLCDataType dataType, const CppMLCDevice &device) {
    return [MLCLayer supportsDataType:toNative(dataType) onDevice:((MLCDevice*)device.self)] == YES;
}
