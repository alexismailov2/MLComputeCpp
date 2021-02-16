cmake_minimum_required(VERSION 3.17)
project(mlc_lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-x objective-c++")

add_library(${PROJECT_NAME}
    CppMLCTypes.h
    CppMLCTypes.mm
    CppMLCTypesPrivate.h
    CppMLCDevice.h
    CppMLCDevice.mm
    CppMLCTensor.h
    CppMLCTensor.mm
    CppMLCTensorData.h
    CppMLCTensorData.mm
    CppMLCTensorDescriptor.h
    CppMLCTensorDescriptor.mm
    CppMLCActivationDescriptor.h
    CppMLCActivationDescriptor.mm
    CppMLCLayer.h
    CppMLCLayer.mm
    CppMLCActivationLayer.h
    CppMLCActivationLayer.mm
    CppMLCAdamOptimizer.h
    CppMLCAdamOptimizer.mm
    CppMLCOptimizer.h
    CppMLCOptimizer.mm
    CppMLCOptimizerDescriptor.h
    CppMLCOptimizerDescriptor.mm)
