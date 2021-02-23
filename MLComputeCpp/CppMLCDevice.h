#pragma once

#include "CppMLCTypes.h"

class CppMLCTensor;
class CppMLCLayer;
class CppMLCGraph;

class CppMLCDevice
{
public:
    /*! @property   type
        @abstract   The device type.
        @discussion Recommend that developers use MLCDeviceTypeAny as the device type.
                    This will ensure that MLCompute will select the best device to execute the neural network.
                    If developers want to be able to control device selection, they can select CPU or GPU and
                    for the GPU, they can also select a specific Metal device.
     */
    eMLCDeviceType getType();

    //NSArray<id<MTLDevice>> *gpuDevices;

    /*! @abstract   Creates a device which uses the CPU.
     *  @return     A new device.
     */
    static CppMLCDevice cpuDevice();

    /*! @abstract   Creates a device which uses a GPU, if any.
        @return     A new device, or `nil` if no GPU exists.
     */
    static CppMLCDevice gpuDevice();

    /*! @abstract   Create a MLCDevice object
        @param      type    A device type
        @return     A new device object
     */
    static CppMLCDevice deviceWithType(eMLCDeviceType type);

    /*! @abstract   Create a MLCDevice object that uses multiple devices if available
        @param      type    A device type
        @param      selectsMultipleComputeDevices    A boolean to indicate whether to select multiple compute devices
        @return     A new device object
     */
    static CppMLCDevice deviceWithType(eMLCDeviceType type, bool selectsMultipleComputeDevices);

    /*! @abstract   Create a MLCDevice object
        @discussion This method can be used by developers to select specific GPUs
        @param      gpus    List of Metal devices
        @return     A new device object
     */
    //static CppMLCDevice deviceWithGPUDevices((NSArray<id<MTLDevice>>* gpus) {};
private:
    CppMLCDevice(void* self);

private:
    void* self;
    friend CppMLCTensor;
    friend CppMLCLayer;
    friend CppMLCGraph;
};
