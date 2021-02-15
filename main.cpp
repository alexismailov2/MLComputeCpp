#include "MLComputeCpp/CppMLCTensorDescriptor.h"
#include "MLComputeCpp/CppMLCDevice.h"

#include <iostream>

int main() {
    auto device = CppMLCDevice::deviceWithType(eMLCDeviceType::Any);
    if (device.getType() == eMLCDeviceType::GPU) {
        std::cout << "GPU Enabled!" << std::endl;
    } else {
        std::cout << "CPU Enabled!" << std::endl;
    }
    auto mlTensor = CppMLCTensorDescriptor{{10, 10, 10}, eMLCDataType::Float32};
    std::cout << "" << mlTensor.isSortedSequences() << std::endl;
    return 0;
}
