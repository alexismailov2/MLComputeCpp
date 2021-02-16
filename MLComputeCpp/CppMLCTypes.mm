#import "CppMLCTypesPrivate.h"

auto toNative(eMLCActivationType activationType) -> MLCActivationType {
    switch(activationType) {
        case eMLCActivationType::None: return MLCActivationTypeNone;
        case eMLCActivationType::ReLU: return MLCActivationTypeReLU;
        case eMLCActivationType::Linear: return MLCActivationTypeLinear;
        case eMLCActivationType::Sigmoid: return MLCActivationTypeSigmoid;
        case eMLCActivationType::HardSigmoid: return MLCActivationTypeHardSigmoid;
        case eMLCActivationType::Tanh: return MLCActivationTypeTanh;
        case eMLCActivationType::Absolute: return MLCActivationTypeAbsolute;
        case eMLCActivationType::SoftPlus: return MLCActivationTypeSoftPlus;
        case eMLCActivationType::SoftSign: return MLCActivationTypeSoftSign;
        case eMLCActivationType::ELU: return MLCActivationTypeELU;
        case eMLCActivationType::ReLUN: return MLCActivationTypeReLUN;
        case eMLCActivationType::LogSigmoid: return MLCActivationTypeLogSigmoid;
        case eMLCActivationType::SELU: return MLCActivationTypeSELU;
        case eMLCActivationType::CELU: return MLCActivationTypeCELU;
        case eMLCActivationType::HardShrink: return MLCActivationTypeHardShrink;
        case eMLCActivationType::SoftShrink: return MLCActivationTypeSoftShrink;
        case eMLCActivationType::TanhShrink: return MLCActivationTypeTanhShrink;
        case eMLCActivationType::Threshold: return MLCActivationTypeThreshold;
        case eMLCActivationType::GELU: return MLCActivationTypeGELU;
        case eMLCActivationType::Count: return MLCActivationTypeCount;
        default: return MLCActivationTypeNone;
    }
}

auto MLCActivationTypeToCpp(MLCActivationType activationType) -> eMLCActivationType {
    switch(activationType) {
        case MLCActivationTypeNone: return eMLCActivationType::None;
        case MLCActivationTypeReLU: return eMLCActivationType::ReLU;
        case MLCActivationTypeLinear: return eMLCActivationType::Linear;
        case MLCActivationTypeSigmoid: return eMLCActivationType::Sigmoid;
        case MLCActivationTypeHardSigmoid: return eMLCActivationType::HardSigmoid;
        case MLCActivationTypeTanh: return eMLCActivationType::Tanh;
        case MLCActivationTypeAbsolute: return eMLCActivationType::Absolute;
        case MLCActivationTypeSoftPlus: return eMLCActivationType::SoftPlus;
        case MLCActivationTypeSoftSign: return eMLCActivationType::SoftSign;
        case MLCActivationTypeELU: return eMLCActivationType::ELU;
        case MLCActivationTypeReLUN: return eMLCActivationType::ReLUN;
        case MLCActivationTypeLogSigmoid: return eMLCActivationType::LogSigmoid;
        case MLCActivationTypeSELU: return eMLCActivationType::SELU;
        case MLCActivationTypeCELU: return eMLCActivationType::CELU;
        case MLCActivationTypeHardShrink: return eMLCActivationType::HardShrink;
        case MLCActivationTypeSoftShrink: return eMLCActivationType::SoftShrink;
        case MLCActivationTypeTanhShrink: return eMLCActivationType::TanhShrink;
        case MLCActivationTypeThreshold: return eMLCActivationType::Threshold;
        case MLCActivationTypeGELU: return eMLCActivationType::GELU;
        case MLCActivationTypeCount: return eMLCActivationType::Count;
        default: return eMLCActivationType::None;
    }
}

auto toNative(eMLCRandomInitializerType randomInitializerType) -> MLCRandomInitializerType {
    switch (randomInitializerType) {
        case eMLCRandomInitializerType::Invalid: return MLCRandomInitializerTypeInvalid;
        case eMLCRandomInitializerType::Uniform: return MLCRandomInitializerTypeUniform;
        case eMLCRandomInitializerType::GlorotUniform: return MLCRandomInitializerTypeGlorotUniform;
        case eMLCRandomInitializerType::Xavier: return MLCRandomInitializerTypeXavier;
        case eMLCRandomInitializerType::Count: return MLCRandomInitializerTypeCount;
        default: return MLCRandomInitializerTypeInvalid;
    }
}

auto toNative(eMLCDataType dataType) -> MLCDataType {
    switch(dataType) {
        case eMLCDataType::Invalid: return MLCDataTypeInvalid;
        case eMLCDataType::Float32: return MLCDataTypeFloat32;
        case eMLCDataType::Boolean: return MLCDataTypeBoolean;
        case eMLCDataType::Int64: return MLCDataTypeInt64;
        case eMLCDataType::Int32: return MLCDataTypeInt32;
        case eMLCDataType::Count: return MLCDataTypeCount;
        default: return MLCDataTypeInvalid;
    }
}

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

auto MLCDataTypeToCpp(MLCDataType dataType) -> eMLCDataType {
    switch(dataType) {
        case MLCDataTypeInvalid: return eMLCDataType::Invalid;
        case MLCDataTypeFloat32: return eMLCDataType::Float32;
        case MLCDataTypeBoolean: return eMLCDataType::Boolean;
        case MLCDataTypeInt64: return eMLCDataType::Int64;
        case MLCDataTypeInt32: return eMLCDataType::Int32;
        case MLCDataTypeCount: return eMLCDataType::Count;
        default: return eMLCDataType::Invalid;
    }
}

auto MLCRegularizationTypeToCpp(MLCRegularizationType regularizationType) -> eMLCRegularizationType {
    switch(regularizationType)
    {
        case MLCRegularizationTypeNone: return eMLCRegularizationType::None;
        case MLCRegularizationTypeL1: return eMLCRegularizationType::L1;
        case MLCRegularizationTypeL2: return eMLCRegularizationType::L2;
        default: return eMLCRegularizationType::None;
    }
}

auto toNSArray(const std::vector<uint32_t> &vector) -> NSArray<NSNumber *> * {
    id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(uint32_t item) {
        [ns addObject:[NSNumber numberWithInteger:(NSInteger)item]];
    });
    return ns;
}

auto toNative(eMLCRegularizationType regularizationType) -> MLCRegularizationType {
    switch(regularizationType)
    {
        case eMLCRegularizationType::None: return MLCRegularizationTypeNone;
        case eMLCRegularizationType::L1: return MLCRegularizationTypeL1;
        case eMLCRegularizationType::L2: return MLCRegularizationTypeL2;
        default: return MLCRegularizationTypeNone;
    }
}
