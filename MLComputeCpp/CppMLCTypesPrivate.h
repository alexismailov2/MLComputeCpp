#pragma once

#import "CppMLCTypes.h"

#import <MLCompute/MLCTypes.h>

#include <vector>

auto toNative(eMLCActivationType activationType) -> MLCActivationType;
auto toNative(eMLCDeviceType deviceType) -> MLCDeviceType;
auto toNative(eMLCRandomInitializerType randomInitializerType) -> MLCRandomInitializerType;
auto toNative(eMLCDataType dataType) -> MLCDataType;
auto toNative(eMLCRegularizationType regularizationType) -> MLCRegularizationType;

auto MLCDataTypeToCpp(MLCDataType dataType) -> eMLCDataType;
auto MLCActivationTypeToCpp(MLCActivationType activationType) -> eMLCActivationType;
auto MLCDeviceTypeToCpp(MLCDeviceType deviceType) -> eMLCDeviceType;
auto MLCRegularizationTypeToCpp(MLCRegularizationType regularizationType) -> eMLCRegularizationType;

auto toNSArray(std::vector<uint32_t> const& vector) -> NSArray<NSNumber*>*;

