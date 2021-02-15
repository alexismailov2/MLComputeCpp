#import "CppMLCDevice.h"

#import <MLCompute/MLCDevice.h>

namespace {
    auto toNative(eMLCDeviceType deviceType) -> MLCDeviceType {
        switch(deviceType) {
            case eMLCDeviceType::CPU: return MLCDeviceTypeCPU;
            case eMLCDeviceType::GPU: return MLCDeviceTypeGPU;
            case eMLCDeviceType::Any: return MLCDeviceTypeAny;
            case eMLCDeviceType::Count: return MLCDeviceTypeCount;
            default: return MLCDeviceTypeAny;
        }
    }

    auto MLCDeviceTypeToCpp(MLCDeviceType deviceType) -> eMLCDeviceType {
        switch(deviceType) {
            case MLCDeviceTypeCPU: return eMLCDeviceType::CPU;
            case MLCDeviceTypeGPU: return eMLCDeviceType::GPU;
            case MLCDeviceTypeAny: return eMLCDeviceType::Any;
            case MLCDeviceTypeCount: return eMLCDeviceType::Count;
            default: return eMLCDeviceType::Any;
        }
    }
}

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


