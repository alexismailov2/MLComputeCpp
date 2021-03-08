#include "CppMLCTypesPrivate.h"
#include "CppMLCLayer.h"
#include "CppMLCTensor.h"
#include "CppMLCTensorData.h"
#include "CppMLCGraph.h"
#include "CppMLCInferenceGraph.h"
#include "CppMLCTrainingGraph.h"
#include "CppMLCTensorOptimizerDeviceData.h"
#include "CppMLCTensorParameter.h"

#import <MLCompute/MLCLayer.h>
#import <MLCompute/MLCTensor.h>

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

auto toNative(eMLCRegularizationType regularizationType) -> MLCRegularizationType {
    switch(regularizationType)
    {
        case eMLCRegularizationType::None: return MLCRegularizationTypeNone;
        case eMLCRegularizationType::L1: return MLCRegularizationTypeL1;
        case eMLCRegularizationType::L2: return MLCRegularizationTypeL2;
        default: return MLCRegularizationTypeNone;
    }
}

auto toNative(eMLCArithmeticOperation arithmeticOperation) -> MLCArithmeticOperation {
    switch (arithmeticOperation) {
        case eMLCArithmeticOperation::Add: return      MLCArithmeticOperationAdd;
        case eMLCArithmeticOperation::Subtract: return MLCArithmeticOperationSubtract;
        case eMLCArithmeticOperation::Multiply: return MLCArithmeticOperationMultiply;
        case eMLCArithmeticOperation::Divide: return   MLCArithmeticOperationDivide;
        case eMLCArithmeticOperation::Floor: return    MLCArithmeticOperationFloor;
        case eMLCArithmeticOperation::Round: return    MLCArithmeticOperationRound;
        case eMLCArithmeticOperation::Ceil: return     MLCArithmeticOperationCeil;
        case eMLCArithmeticOperation::Sqrt: return     MLCArithmeticOperationSqrt;
        case eMLCArithmeticOperation::Rsqrt: return    MLCArithmeticOperationRsqrt;
        case eMLCArithmeticOperation::Sin: return      MLCArithmeticOperationSin;
        case eMLCArithmeticOperation::Cos: return      MLCArithmeticOperationCos;
        case eMLCArithmeticOperation::Tan: return      MLCArithmeticOperationTan;
        case eMLCArithmeticOperation::Asin: return     MLCArithmeticOperationAsin;
        case eMLCArithmeticOperation::Acos: return     MLCArithmeticOperationAcos;
        case eMLCArithmeticOperation::Atan: return     MLCArithmeticOperationAtan;
        case eMLCArithmeticOperation::Sinh: return     MLCArithmeticOperationSinh;
        case eMLCArithmeticOperation::Cosh: return     MLCArithmeticOperationCosh;
        case eMLCArithmeticOperation::Tanh: return     MLCArithmeticOperationTanh;
        case eMLCArithmeticOperation::Asinh: return    MLCArithmeticOperationAsinh;
        case eMLCArithmeticOperation::Acosh: return    MLCArithmeticOperationAcosh;
        case eMLCArithmeticOperation::Atanh: return    MLCArithmeticOperationAtanh;
        case eMLCArithmeticOperation::Pow: return      MLCArithmeticOperationPow;
        case eMLCArithmeticOperation::Exp: return      MLCArithmeticOperationExp;
        case eMLCArithmeticOperation::Exp2: return     MLCArithmeticOperationExp2;
        case eMLCArithmeticOperation::Log: return      MLCArithmeticOperationLog;
        case eMLCArithmeticOperation::Log2: return     MLCArithmeticOperationLog2;
        default:
        case eMLCArithmeticOperation::Count: return    MLCArithmeticOperationCount;
    }
}

auto MLCArithmeticOperationToCpp(MLCArithmeticOperation arithmeticOperation) -> eMLCArithmeticOperation {
    switch (arithmeticOperation) {
        case MLCArithmeticOperationAdd: return      eMLCArithmeticOperation::Add;
        case MLCArithmeticOperationSubtract: return eMLCArithmeticOperation::Subtract;
        case MLCArithmeticOperationMultiply: return eMLCArithmeticOperation::Multiply;
        case MLCArithmeticOperationDivide: return   eMLCArithmeticOperation::Divide;
        case MLCArithmeticOperationFloor: return    eMLCArithmeticOperation::Floor;
        case MLCArithmeticOperationRound: return    eMLCArithmeticOperation::Round;
        case MLCArithmeticOperationCeil: return     eMLCArithmeticOperation::Ceil;
        case MLCArithmeticOperationSqrt: return     eMLCArithmeticOperation::Sqrt;
        case MLCArithmeticOperationRsqrt: return    eMLCArithmeticOperation::Rsqrt;
        case MLCArithmeticOperationSin: return      eMLCArithmeticOperation::Sin;
        case MLCArithmeticOperationCos: return      eMLCArithmeticOperation::Cos;
        case MLCArithmeticOperationTan: return      eMLCArithmeticOperation::Tan;
        case MLCArithmeticOperationAsin: return     eMLCArithmeticOperation::Asin;
        case MLCArithmeticOperationAcos: return     eMLCArithmeticOperation::Acos;
        case MLCArithmeticOperationAtan: return     eMLCArithmeticOperation::Atan;
        case MLCArithmeticOperationSinh: return     eMLCArithmeticOperation::Sinh;
        case MLCArithmeticOperationCosh: return     eMLCArithmeticOperation::Cosh;
        case MLCArithmeticOperationTanh: return     eMLCArithmeticOperation::Tanh;
        case MLCArithmeticOperationAsinh: return    eMLCArithmeticOperation::Asinh;
        case MLCArithmeticOperationAcosh: return    eMLCArithmeticOperation::Acosh;
        case MLCArithmeticOperationAtanh: return    eMLCArithmeticOperation::Atanh;
        case MLCArithmeticOperationPow: return      eMLCArithmeticOperation::Pow;
        case MLCArithmeticOperationExp: return      eMLCArithmeticOperation::Exp;
        case MLCArithmeticOperationExp2: return     eMLCArithmeticOperation::Exp2;
        case MLCArithmeticOperationLog: return      eMLCArithmeticOperation::Log;
        case MLCArithmeticOperationLog2: return     eMLCArithmeticOperation::Log2;
        default:
        case MLCArithmeticOperationCount: return    eMLCArithmeticOperation::Count;
    }
}

auto toNative(eMLCConvolutionType convolutionType) -> MLCConvolutionType {
    switch(convolutionType) {
        case eMLCConvolutionType::Standard: return MLCConvolutionTypeStandard;
        case eMLCConvolutionType::Transposed: return MLCConvolutionTypeTransposed;
        case eMLCConvolutionType::Depthwise: return MLCConvolutionTypeDepthwise;
        default: return MLCConvolutionTypeStandard;
    }
}

auto MLCConvolutionTypeToCpp(MLCConvolutionType convolutionType) -> eMLCConvolutionType {
    switch(convolutionType) {
        case MLCConvolutionTypeStandard: return eMLCConvolutionType::Standard;
        case MLCConvolutionTypeTransposed: return eMLCConvolutionType::Transposed;
        case MLCConvolutionTypeDepthwise: return eMLCConvolutionType::Depthwise;
        default: return eMLCConvolutionType::Standard;
    }
}

auto toNative(eMLCPaddingPolicy paddingPolicy) -> MLCPaddingPolicy {
    switch(paddingPolicy) {
        case eMLCPaddingPolicy::Same: return MLCPaddingPolicySame;
        case eMLCPaddingPolicy::Valid: return MLCPaddingPolicyValid;
        case eMLCPaddingPolicy::UsePaddingSize: return MLCPaddingPolicyUsePaddingSize;
        default: return MLCPaddingPolicySame;
    }
}

auto toNative(eMLCExecutionOptions executionOptions) -> MLCExecutionOptions {
    switch(executionOptions) {
        case eMLCExecutionOptions::None: return MLCExecutionOptionsNone;
        case eMLCExecutionOptions::SkipWritingInputDataToDevice: return MLCExecutionOptionsSkipWritingInputDataToDevice;
        case eMLCExecutionOptions::Synchronous: return MLCExecutionOptionsSynchronous;
        case eMLCExecutionOptions::ForwardForInference: return MLCExecutionOptionsForwardForInference;
        default: return MLCExecutionOptionsNone;
    }
}

auto toNative(eMLCGraphCompilationOptions graphCompilationOptions) -> MLCGraphCompilationOptions {
    switch(graphCompilationOptions) {
        case eMLCGraphCompilationOptions::None: return MLCGraphCompilationOptionsNone;
        case eMLCGraphCompilationOptions::DebugLayers: return MLCGraphCompilationOptionsDebugLayers;
        case eMLCGraphCompilationOptions::DisableLayerFusion: return MLCGraphCompilationOptionsDisableLayerFusion;
        case eMLCGraphCompilationOptions::LinkGraphs: return MLCGraphCompilationOptionsLinkGraphs;
        case eMLCGraphCompilationOptions::ComputeAllGradients: return MLCGraphCompilationOptionsComputeAllGradients;
        default: return MLCGraphCompilationOptionsNone;
    }
}

auto toNative(eMLCSoftmaxOperation softmaxOperation) -> MLCSoftmaxOperation
{
    switch(softmaxOperation)
    {
        case eMLCSoftmaxOperation::Softmax: return MLCSoftmaxOperationSoftmax;
        case eMLCSoftmaxOperation::LogSoftmax: return MLCSoftmaxOperationLogSoftmax;
        default: return MLCSoftmaxOperationSoftmax;
    }
}

auto MLCPaddingPolicyToCpp(MLCPaddingPolicy paddingPolicy) -> eMLCPaddingPolicy {
    switch(paddingPolicy) {
        case MLCPaddingPolicySame: return eMLCPaddingPolicy::Same;
        case MLCPaddingPolicyValid: return eMLCPaddingPolicy::Valid;
        case MLCPaddingPolicyUsePaddingSize: return eMLCPaddingPolicy::UsePaddingSize;
        default: return eMLCPaddingPolicy::Same;
    }
}

auto MLCSoftmaxOperationToCpp(MLCSoftmaxOperation softmaxOperation) -> eMLCSoftmaxOperation
{
    switch(softmaxOperation)
    {
        case MLCSoftmaxOperationSoftmax: return eMLCSoftmaxOperation::Softmax;
        case MLCSoftmaxOperationLogSoftmax: return eMLCSoftmaxOperation::LogSoftmax;
        default: return eMLCSoftmaxOperation::Softmax;
    }
}

auto toNative(eMLCSampleMode sampleMode) -> MLCSampleMode
{
    switch(sampleMode)
    {
        case eMLCSampleMode::Linear: return MLCSampleModeLinear;
        case eMLCSampleMode::Nearest: return MLCSampleModeNearest;
        default: return MLCSampleModeNearest;
    }
}

auto MLCSampleModeToCpp(MLCSampleMode sampleMode) -> eMLCSampleMode
{
    switch(sampleMode)
    {
        case MLCSampleModeLinear: return eMLCSampleMode::Linear;
        case MLCSampleModeNearest: return eMLCSampleMode::Nearest;
        default: return eMLCSampleMode::Nearest;
    }
}

auto toNative(eMLCLossType lossType) -> MLCLossType
{
    switch(lossType)
    {
        case eMLCLossType::MeanAbsoluteError: return MLCLossTypeMeanAbsoluteError;
        case eMLCLossType::MeanSquaredError: return MLCLossTypeMeanSquaredError;
        case eMLCLossType::SoftmaxCrossEntropy: return MLCLossTypeSoftmaxCrossEntropy;
        case eMLCLossType::SigmoidCrossEntropy: return MLCLossTypeSigmoidCrossEntropy;
        case eMLCLossType::CategoricalCrossEntropy: return MLCLossTypeCategoricalCrossEntropy;
        case eMLCLossType::Hinge: return MLCLossTypeHinge;
        case eMLCLossType::Huber: return MLCLossTypeHuber;
        case eMLCLossType::CosineDistance: return MLCLossTypeCosineDistance;
        case eMLCLossType::Log: return MLCLossTypeLog;
        case eMLCLossType::Count:
        default: return MLCLossTypeCount;
    }
}

auto MLCLossTypeToCpp(MLCLossType lossType) -> eMLCLossType
{
    switch(lossType)
    {
        case MLCLossTypeMeanAbsoluteError: return eMLCLossType::MeanAbsoluteError;
        case MLCLossTypeMeanSquaredError: return eMLCLossType::MeanSquaredError;
        case MLCLossTypeSoftmaxCrossEntropy: return eMLCLossType::SoftmaxCrossEntropy;
        case MLCLossTypeSigmoidCrossEntropy: return eMLCLossType::SigmoidCrossEntropy;
        case MLCLossTypeCategoricalCrossEntropy: return eMLCLossType::CategoricalCrossEntropy;
        case MLCLossTypeHinge: return eMLCLossType::Hinge;
        case MLCLossTypeHuber: return eMLCLossType::Huber;
        case MLCLossTypeCosineDistance: return eMLCLossType::CosineDistance;
        case MLCLossTypeLog: return eMLCLossType::Log;
        case MLCLossTypeCount:
        default: return eMLCLossType::Count;
    }
}

auto toNative(eMLCReductionType reductionType) -> MLCReductionType
{
    switch(reductionType)
    {
        case eMLCReductionType::None: return MLCReductionTypeNone;
        case eMLCReductionType::Sum: return MLCReductionTypeSum;
        case eMLCReductionType::Mean: return MLCReductionTypeMean;
        case eMLCReductionType::Max: return MLCReductionTypeMax;
        case eMLCReductionType::Min: return MLCReductionTypeMin;
        case eMLCReductionType::ArgMax: return MLCReductionTypeArgMax;
        case eMLCReductionType::ArgMin: return MLCReductionTypeArgMin;
        case eMLCReductionType::Count:
        default: return MLCReductionTypeCount;
    }
}

auto MLCReductionTypeToCpp(MLCReductionType reductionType) -> eMLCReductionType
{
    switch(reductionType)
    {
        case MLCReductionTypeNone: return eMLCReductionType::None;
        case MLCReductionTypeSum: return eMLCReductionType::Sum;
        case MLCReductionTypeMean: return eMLCReductionType::Mean;
        case MLCReductionTypeMax: return eMLCReductionType::Max;
        case MLCReductionTypeMin: return eMLCReductionType::Min;
        case MLCReductionTypeArgMax: return eMLCReductionType::ArgMax;
        case MLCReductionTypeArgMin: return eMLCReductionType::ArgMin;
        case MLCReductionTypeCount:
        default: return eMLCReductionType::Count;
    }
}

auto toNative(eMLCPoolingType poolingType) -> MLCPoolingType
{
    switch(poolingType)
    {
        case eMLCPoolingType::Max: return MLCPoolingTypeMax;
        case eMLCPoolingType::Average: return MLCPoolingTypeAverage;
        case eMLCPoolingType::L2Norm: return MLCPoolingTypeL2Norm;
        case eMLCPoolingType::Count:
        default: return MLCPoolingTypeCount;
    }
}

auto toNative(eMLCPaddingType paddingType) -> MLCPaddingType
{
    switch(paddingType)
    {
        case eMLCPaddingType::Zero: return MLCPaddingTypeZero;
        case eMLCPaddingType::Reflect: return MLCPaddingTypeReflect;
        case eMLCPaddingType::Symmetric: return MLCPaddingTypeSymmetric;
        case eMLCPaddingType::Constant:
        default: return MLCPaddingTypeConstant;
    }
}

auto MLCPoolingTypeToCpp(MLCPoolingType poolingType) -> eMLCPoolingType
{
    switch(poolingType)
    {
        case MLCPoolingTypeMax: return eMLCPoolingType::Max;
        case MLCPoolingTypeAverage: return eMLCPoolingType::Average;
        case MLCPoolingTypeL2Norm: return eMLCPoolingType::L2Norm;
        case MLCPoolingTypeCount:
        default: return eMLCPoolingType::Count;
    }
}

auto MLCPaddingTypeToCpp(MLCPaddingType paddingType) -> eMLCPaddingType
{
    switch(paddingType)
    {
        case MLCPaddingTypeZero: return eMLCPaddingType::Zero;
        case MLCPaddingTypeReflect: return eMLCPaddingType::Reflect;
        case MLCPaddingTypeSymmetric: return eMLCPaddingType::Symmetric;
        case MLCPaddingTypeConstant:
        default: return eMLCPaddingType::Constant;
    }
}

auto CppMLCTypesPrivate::toNSArray(std::vector<uint32_t> const& vector) -> NSArray<NSNumber *> * {
    id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(uint32_t item) {
        [ns addObject:[NSNumber numberWithInteger:(NSInteger)item]];
    });
    return ns;
}

auto CppMLCTypesPrivate::NSNumberArrayToVector(NSArray<NSNumber *> * array) -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(NSNumber * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSArray(const std::vector<CppMLCLayer> &vector) -> NSArray<MLCLayer *> * {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCLayer const& item) {
        [ns addObject:(MLCLayer*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::MLCLayerArrayToVector(NSArray<MLCLayer *> *array) -> std::vector<CppMLCLayer> {
    __block std::vector<CppMLCLayer> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(MLCLayer * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.emplace_back(CppMLCLayer{obj});
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCTensor> const& vector) -> NSArray<MLCTensor*>* {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCTensor const& item) {
        [ns addObject:(MLCTensor*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::MLCTensorArrayToVector(NSArray<MLCTensor*>* array) -> std::vector<CppMLCTensor> {
    __block std::vector<CppMLCTensor> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(MLCTensor * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.emplace_back(CppMLCTensor{obj});
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCTensorData> const& vector) -> NSArray<MLCTensorData*>* {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCTensorData const& item) {
        [ns addObject:(MLCTensorData*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::MLCTensorDataArrayToVector(NSArray<MLCTensorData*>* array) -> std::vector<CppMLCTensorData> {
    __block std::vector<CppMLCTensorData> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(MLCTensorData * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.emplace_back(CppMLCTensorData{obj});
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCTensorOptimizerDeviceData> const& vector) -> NSArray<MLCTensorOptimizerDeviceData*>* {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCTensorOptimizerDeviceData const& item) {
        [ns addObject:(MLCTensorOptimizerDeviceData*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::MLCTensorOptimizerDeviceDataToVector(NSArray<MLCTensorOptimizerDeviceData*>* array) -> std::vector<CppMLCTensorOptimizerDeviceData> {
    __block std::vector<CppMLCTensorOptimizerDeviceData> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(MLCTensorOptimizerDeviceData * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.emplace_back(CppMLCTensorOptimizerDeviceData{obj});
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCTensorParameter> const& vector) -> NSArray<MLCTensorParameter*>* {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCTensorParameter const& item) {
        [ns addObject:(MLCTensorParameter*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::MLCTensorParameterToVector(NSArray<MLCTensorParameter*>* array) -> std::vector<CppMLCTensorParameter> {
    __block std::vector<CppMLCTensorParameter> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(MLCTensorParameter * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.emplace_back(CppMLCTensorParameter{obj});
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCGraph> const& vector) -> NSArray<MLCGraph*>* {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCGraph const& item) {
        [ns addObject:(MLCGraph*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::MLCGraphToVector(NSArray<MLCGraph*>* array) -> std::vector<CppMLCGraph> {
    __block std::vector<CppMLCGraph> vector;
    vector.reserve([array count]);
    [array enumerateObjectsUsingBlock:^(MLCGraph * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        vector.emplace_back(CppMLCGraph{obj});
    }];
    return vector;
}

auto CppMLCTypesPrivate::toNSDictionary(std::map<std::string, CppMLCTensorData> const& ditcionary) -> NSDictionary<NSString *, MLCTensorData *> * {
    id ns = [NSMutableDictionary new];
    for (auto const& item : ditcionary) {
        [ns setObject:(MLCTensorData*)(item.second.self)
               forKey:[NSString stringWithUTF8String:item.first.c_str()]];
    }
    return ns;
}

auto CppMLCTypesPrivate::toNSDictionary(std::map<std::string, CppMLCTensor> const& ditcionary) -> NSDictionary<NSString *, MLCTensor *> * {
    id ns = [NSMutableDictionary new];
    for (auto const& item : ditcionary) {
        [ns setObject:(MLCTensor*)(item.second.self)
               forKey:[NSString stringWithUTF8String:item.first.c_str()]];
    }
    return ns;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCInferenceGraph> const& vector) -> NSArray<MLCInferenceGraph *> * {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCInferenceGraph const& item) {
        [ns addObject:(MLCInferenceGraph*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::toNSArray(std::vector<CppMLCTrainingGraph> const& vector) -> NSArray<MLCTrainingGraph *> * {
    __block id ns = [NSMutableArray new];
    std::for_each(vector.begin(), vector.end(), ^(CppMLCTrainingGraph const& item) {
        [ns addObject:(MLCTrainingGraph*)item.self];
    });
    return ns;
}

auto CppMLCTypesPrivate::NSDataToVectorFloat(NSData* data) -> std::vector<float>
{
    const size_t count = [data length] / sizeof(float);
    auto first = (float *)[data bytes];
    auto last = first + count;
    return std::vector<float>(first, last);
}

auto CppMLCTypesPrivate::toNSData(std::vector<float> const& data) -> NSData*
{
    return [NSData dataWithBytes:data.data()
                          length:(data.size() * sizeof(float))];
}


