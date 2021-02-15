#ifndef MLCOMPUTECPP_MLCTYPES_H
#define MLCOMPUTECPP_MLCTYPES_H

#include <cstdint>

/*!
 *  @enum      eMLCDataType
 *  @abstract  A tensor data type.
 */
enum class eMLCDataType : int32_t {
    Invalid = 0,
    Float32 = 1,
    Boolean = 4,
    Int64   = 5,
    Int32   = 7,
    Count
};

/*!
 *  @enum      eMLCRandomInitializerType
 *  @abstract  An initializer type you use to create a tensor with random data.
 */
enum class eMLCRandomInitializerType : int32_t {
    Invalid = 0,
    Uniform = 1,
    GlorotUniform = 2,
    Xavier = 3,
    Count
};

/*!
 *  @enum       eMLCDeviceType
 *  @abstract   A device type for execution of a neural network.
 */
enum class eMLCDeviceType : int32_t {
    CPU = 0,
    GPU = 1,
    Any = 2,
    Count
};

/*!
 *  @enum       eMLCGraphCompilationOptions
 *  @abstract   A bitmask that specifies the options you use when compiling a graph.
 *  @discussion This is passed as an argument to the compileWithOptions method avalable on MLCTrainingGraph and MLCInferenceGraph
 */
enum class eMLCGraphCompilationOptions : uint64_t {
    None = 0x00,
    DebugLayers = 0x01,
    DisableLayerFusion = 0x02,
    LinkGraphs = 0x04,
    ComputeAllGradients = 0x08
};

/*!
 *  @enum       eMLCExecutionOptions
 *  @abstract   A bitmask that specifies the options you’ll use when executing a graph.
 */
enum class eMLCExecutionOptions : uint64_t {
    None = 0x00,
    SkipWritingInputDataToDevice = 0x01,
    Synchronous = 0x02,
    Profiling = 0x04,
    ForwardForInference = 0x08
};

/*!
 *  @enum       eMLCArithmeticOperation
 *  @abstract   The list of supported arithmetic operations.
 */
enum class eMLCArithmeticOperation : int32_t {
MLCArithmeticOperationAdd      = 0,
MLCArithmeticOperationSubtract = 1,
MLCArithmeticOperationMultiply = 2,
MLCArithmeticOperationDivide   = 3,
MLCArithmeticOperationFloor    = 4,
MLCArithmeticOperationRound    = 5,
MLCArithmeticOperationCeil     = 6,
MLCArithmeticOperationSqrt     = 7,
MLCArithmeticOperationRsqrt    = 8,
MLCArithmeticOperationSin      = 9,
MLCArithmeticOperationCos      = 10,
MLCArithmeticOperationTan      = 11,
MLCArithmeticOperationAsin     = 12,
MLCArithmeticOperationAcos     = 13,
MLCArithmeticOperationAtan     = 14,
MLCArithmeticOperationSinh     = 15,
MLCArithmeticOperationCosh     = 16,
MLCArithmeticOperationTanh     = 17,
MLCArithmeticOperationAsinh    = 18,
MLCArithmeticOperationAcosh    = 19,
MLCArithmeticOperationAtanh    = 20,
MLCArithmeticOperationPow      = 21,
MLCArithmeticOperationExp      = 22,
MLCArithmeticOperationExp2     = 23,
MLCArithmeticOperationLog      = 24,
MLCArithmeticOperationLog2     = 25,
MLCArithmeticOperationCount,
};

/*!
 *  @enum       eMLCLossType
 *  @abstract   A loss function.
 */
enum class eMLCLossType : int32_t {
MLCLossTypeMeanAbsoluteError       = 0,
MLCLossTypeMeanSquaredError        = 1,
MLCLossTypeSoftmaxCrossEntropy     = 2,
MLCLossTypeSigmoidCrossEntropy     = 3,
MLCLossTypeCategoricalCrossEntropy = 4,
MLCLossTypeHinge                   = 5,
MLCLossTypeHuber                   = 6,
MLCLossTypeCosineDistance          = 7,
MLCLossTypeLog                     = 8,
MLCLossTypeCount
};

/*!
 * @enum       eMLCActivationType
 * @abstract   An activation type that you specify for an activation descriptor.
 */
enum class eMLCActivationType : int32_t {
MLCActivationTypeNone                                               = 0,
MLCActivationTypeReLU                                               = 1,
MLCActivationTypeLinear                                             = 2,
MLCActivationTypeSigmoid                                            = 3,
MLCActivationTypeHardSigmoid                                        = 4,
MLCActivationTypeTanh                                               = 5,
MLCActivationTypeAbsolute                                           = 6,
MLCActivationTypeSoftPlus                                           = 7,
MLCActivationTypeSoftSign                                           = 8,
MLCActivationTypeELU                                                = 9,
MLCActivationTypeReLUN                                              = 10,
MLCActivationTypeLogSigmoid                                         = 11,
MLCActivationTypeSELU                                               = 12,
MLCActivationTypeCELU                                               = 13,
MLCActivationTypeHardShrink                                         = 14,
MLCActivationTypeSoftShrink                                         = 15,
MLCActivationTypeTanhShrink                                         = 16,
MLCActivationTypeThreshold                                          = 17,
MLCActivationTypeGELU                                               = 18,
MLCActivationTypeCount
};

/*!
 * @enum       eMLCConvolutionType
 * @abstract   A convolution type that you specify for a convolution descriptor.
 */
enum class eMLCConvolutionType : int32_t {
MLCConvolutionTypeStandard   = 0,
MLCConvolutionTypeTransposed = 1,
MLCConvolutionTypeDepthwise  = 2,
};

/*!
 * @enum       eMLCPaddingPolicy
 * @abstract   A padding policy that you specify for a convolution or pooling layer.
 */
enum class eMLCPaddingPolicy : int32_t {
MLCPaddingPolicySame           = 0,
MLCPaddingPolicyValid          = 1,
MLCPaddingPolicyUsePaddingSize = 2,
};

/*!
 *  @enum       eMLCPaddingType
 *  @abstract   A padding type that you specify for a padding layer.
 */
enum class eMLCPaddingType : int32_t {
MLCPaddingTypeZero      = 0,
MLCPaddingTypeReflect   = 1,
MLCPaddingTypeSymmetric = 2,
MLCPaddingTypeConstant  = 3,
};

/*!
 *  @enum       eMLCPoolingType
 *  @abstract   A pooling function type for a pooling layer.
 */
enum class eMLCPoolingType : int32_t {
MLCPoolingTypeMax     = 1,
MLCPoolingTypeAverage = 2,
MLCPoolingTypeL2Norm  = 3,
MLCPoolingTypeCount
};

/*!
 *  @enum       eMLCReductionType
 *  @abstract   A reduction operation type.
 */
enum class eMLCReductionType : int32_t {
MLCReductionTypeNone                 = 0,
MLCReductionTypeSum                  = 1,
MLCReductionTypeMean                 = 2,
MLCReductionTypeMax                  = 3,
MLCReductionTypeMin                  = 4,
MLCReductionTypeArgMax               = 5,
MLCReductionTypeArgMin               = 6,
MLCReductionTypeCount
};

/*!
 *  @enum       eMLCRegularizationType
 *  @abstract
 */
enum class eMLCRegularizationType : int32_t {
MLCRegularizationTypeNone                   = 0,
MLCRegularizationTypeL1                     = 1,
MLCRegularizationTypeL2                     = 2,
};

/*!
 *  @enum       eMLCSampleMode
 *  @abstract   A sampling mode for an upsample layer.
 */
enum class eMLCSampleMode : int32_t {
MLCSampleModeNearest = 0,
MLCSampleModeLinear  = 1,
};

/*!
 *  @enum       eMLCSoftmaxOperation
 *  @abstract   A softmax operation.
 */
enum class eMLCSoftmaxOperation : int32_t {
MLCSoftmaxOperationSoftmax    = 0,
MLCSoftmaxOperationLogSoftmax = 1,
};

/*!
 * @enum       eMLCLSTMResultMode
 * @abstract   A result mode for an LSTM layer.
 */
enum class eMLCLSTMResultMode : uint64_t {
MLCLSTMResultModeOutput = 0,
MLCLSTMResultModeOutputAndStates = 1,
};

//NSString *MLCActivationTypeDebugDescription(MLCActivationType activationType)
//NSString *MLCArithmeticOperationDebugDescription(MLCArithmeticOperation operation)
//NSString *MLCPaddingPolicyDebugDescription(MLCPaddingPolicy paddingPolicy)
//NSString *MLCLossTypeDebugDescription(MLCLossType lossType)
//NSString *MLCReductionTypeDebugDescription(MLCReductionType reductionType)
//NSString *MLCPaddingTypeDebugDescription(MLCPaddingType paddingType)
//NSString *MLCConvolutionTypeDebugDescription(MLCConvolutionType convolutionType)
//NSString *MLCPoolingTypeDebugDescription(MLCPoolingType poolingType)
//NSString *MLCSoftmaxOperationDebugDescription(MLCSoftmaxOperation operation)
//NSString *MLCSampleModeDebugDescription(MLCSampleMode mode)
//NSString *MLCLSTMResultModeDebugDescription(MLCLSTMResultMode mode)

#endif