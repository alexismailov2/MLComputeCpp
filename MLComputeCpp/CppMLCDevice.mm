#import "CppMLCDevice.h"

#import "CppMLCTypes.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCDevice.h>

CppMLCDevice::CppMLCDevice(void *mlcDevice)
    : self{mlcDevice}
{
}

CppMLCDevice::~CppMLCDevice() {
// TODO: ????????
}

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


