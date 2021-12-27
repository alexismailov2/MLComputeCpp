#include "CppMLCDevice.h"

#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCDevice.h>

auto CppMLCDevice::getType() -> eMLCDeviceType
{
    return MLCDeviceTypeToCpp(((MLCDevice*)self).type);
}

CppMLCDevice CppMLCDevice::cpuDevice()
{
    return CppMLCDevice{[MLCDevice cpuDevice]};
}

CppMLCDevice CppMLCDevice::gpuDevice()
{
    return CppMLCDevice{[MLCDevice gpuDevice]};
}

CppMLCDevice CppMLCDevice::deviceWithType(eMLCDeviceType type)
{
    return CppMLCDevice{[MLCDevice deviceWithType:toNative(type)]};
}

CppMLCDevice CppMLCDevice::deviceWithType(eMLCDeviceType type, bool selectsMultipleComputeDevices)
{
    return CppMLCDevice{[MLCDevice deviceWithType:toNative(type)
                    selectsMultipleComputeDevices:selectsMultipleComputeDevices ? YES : NO]};
}

CppMLCDevice::CppMLCDevice(void* self)
    : self{self}
{
    [(id)self retain];
}

CppMLCDevice::~CppMLCDevice()
{

}

