#pragma once

#include "CppMLCTypes.h"

#import <MLCompute/MLCTypes.h>

#include <vector>
#include <map>

auto toNative(eMLCActivationType activationType) -> MLCActivationType;
auto toNative(eMLCDeviceType deviceType) -> MLCDeviceType;
auto toNative(eMLCRandomInitializerType randomInitializerType) -> MLCRandomInitializerType;
auto toNative(eMLCDataType dataType) -> MLCDataType;
auto toNative(eMLCRegularizationType regularizationType) -> MLCRegularizationType;
auto toNative(eMLCArithmeticOperation arithmeticOperation) -> MLCArithmeticOperation;
auto toNative(eMLCConvolutionType convolutionType) -> MLCConvolutionType;
auto toNative(eMLCPaddingPolicy paddingPolicy) -> MLCPaddingPolicy;

auto MLCDataTypeToCpp(MLCDataType dataType) -> eMLCDataType;
auto MLCActivationTypeToCpp(MLCActivationType activationType) -> eMLCActivationType;
auto MLCDeviceTypeToCpp(MLCDeviceType deviceType) -> eMLCDeviceType;
auto MLCRegularizationTypeToCpp(MLCRegularizationType regularizationType) -> eMLCRegularizationType;
auto MLCArithmeticOperationToCpp(MLCArithmeticOperation arithmeticOperation) -> eMLCArithmeticOperation;
auto MLCConvolutionTypeToCpp(MLCConvolutionType convolutionType) -> eMLCConvolutionType;
auto MLCPaddingPolicyToCpp(MLCPaddingPolicy paddingPolicy) -> eMLCPaddingPolicy;

@class MLCLayer;
@class MLCTensor;
@class MLCTensorData;
class CppMLCLayer;
class CppMLCTensor;
class CppMLCTensorData;

class CppMLCTypesPrivate
{
public:
    static auto toNSArray(std::vector<uint32_t> const& vector) -> NSArray<NSNumber*>*;
    static auto NSNumberArrayTotoVector(NSArray<NSNumber*>*) -> std::vector<uint32_t>;

    static auto toNSArray(std::vector<CppMLCLayer> const& vector) -> NSArray<MLCLayer*>*;
    static auto MLCLayerArrayToVector(NSArray<MLCLayer*>* array) -> std::vector<CppMLCLayer>;

    static auto toNSArray(std::vector<CppMLCTensor> const& vector) -> NSArray<MLCTensor*>*;
    static auto MLCTensorArrayToVector(NSArray<MLCTensor*>* array) -> std::vector<CppMLCTensor>;

    static auto toNSDictionary(std::map<std::string, CppMLCTensorData> const& tensorDataDisctionary) -> NSDictionary<NSString*, MLCTensorData*>*;
    static auto toNSDictionary(std::map<std::string, CppMLCTensor> const& tensorDisctionary) -> NSDictionary<NSString*, MLCTensor*>*;
};


