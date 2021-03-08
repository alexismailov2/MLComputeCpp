#include "MLComputeCpp/CppMLCDevice.h"
#include "MLComputeCpp/CppMLCTensorDescriptor.h"
#include "MLComputeCpp/CppMLCTensor.h"

#include <iostream>

int main() {
    auto device = CppMLCDevice::deviceWithType(eMLCDeviceType::Any);
    if (device.getType() == eMLCDeviceType::GPU) {
        std::cout << "GPU Enabled!" << std::endl;
    } else {
        std::cout << "CPU Enabled!" << std::endl;
    }
    auto cppConvTensorWeights = CppMLCTensor{{128, 128, 1}, eMLCDataType::Float32};
    auto cppConvTensorBias = CppMLCTensor{{128, 128, 1}, eMLCDataType::Float32};
    //auto cpp = CppMLCConvolutionDescriptor::
    //auto cppLayer = CppMLCConvolutionLayer::layerWithWeights(cppConvTensorWeights, cppConvTensorBias, );
    auto mlTensor = CppMLCTensorDescriptor{{10, 10, 10}, eMLCDataType::Float32};
    std::cout << "" << mlTensor.isSortedSequences() << std::endl;
    return 0;
}
