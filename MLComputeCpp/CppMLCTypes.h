#pragma once

#include <cstdint>
#include <functional>

class CppMLCTensor;

/*! @abstract A callback completion handler you execute when a graph finishes execution.
 */
using CppMLCGraphCompletionHandler = std::function<void(CppMLCTensor const&, std::string, std::string)>;

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
    Add,
    Subtract,
    Multiply,
    Divide,
    Floor,
    Round,
    Ceil,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Pow,
    Exp,
    Exp2,
    Log,
    Log2,
    Count
};

/*!
 *  @enum       eMLCLossType
 *  @abstract   A loss function.
 */
enum class eMLCLossType : int32_t {
    MeanAbsoluteError,
    MeanSquaredError,
    SoftmaxCrossEntropy,
    SigmoidCrossEntropy,
    CategoricalCrossEntropy,
    Hinge,
    Huber,
    CosineDistance,
    Log,
    Count
};

/*!
 * @enum       eMLCActivationType
 * @abstract   An activation type that you specify for an activation descriptor.
 */
enum class eMLCActivationType : int32_t {
    None,
    ReLU,
    Linear,
    Sigmoid,
    HardSigmoid,
    Tanh,
    Absolute,
    SoftPlus,
    SoftSign,
    ELU,
    ReLUN,
    LogSigmoid,
    SELU,
    CELU,
    HardShrink,
    SoftShrink,
    TanhShrink,
    Threshold,
    GELU,
    Count
};

/*!
 * @enum       eMLCConvolutionType
 * @abstract   A convolution type that you specify for a convolution descriptor.
 */
enum class eMLCConvolutionType : int32_t {
    Standard,
    Transposed,
    Depthwise
};

/*!
 * @enum       eMLCPaddingPolicy
 * @abstract   A padding policy that you specify for a convolution or pooling layer.
 */
enum class eMLCPaddingPolicy : int32_t {
    Same,
    Valid,
    UsePaddingSize
};

/*!
 *  @enum       eMLCPaddingType
 *  @abstract   A padding type that you specify for a padding layer.
 */
enum class eMLCPaddingType : int32_t {
    Zero,
    Reflect,
    Symmetric,
    Constant
};

/*!
 *  @enum       eMLCPoolingType
 *  @abstract   A pooling function type for a pooling layer.
 */
enum class eMLCPoolingType : int32_t {
    Max = 1,
    Average,
    L2Norm,
    Count
};

/*!
 *  @enum       eMLCReductionType
 *  @abstract   A reduction operation type.
 */
enum class eMLCReductionType : int32_t {
    None,
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
    Count
};

/*!
 *  @enum       eMLCRegularizationType
 *  @abstract
 */
enum class eMLCRegularizationType : int32_t {
    None,
    L1,
    L2
};

/*!
 *  @enum       eMLCSampleMode
 *  @abstract   A sampling mode for an upsample layer.
 */
enum class eMLCSampleMode : int32_t {
    Nearest,
    Linear
};

/*!
 *  @enum       eMLCSoftmaxOperation
 *  @abstract   A softmax operation.
 */
enum class eMLCSoftmaxOperation : int32_t {
    Softmax,
    LogSoftmax
};

/*!
 * @enum       eMLCLSTMResultMode
 * @abstract   A result mode for an LSTM layer.
 */
enum class eMLCLSTMResultMode : uint64_t {
    Output,
    OutputAndStates,
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