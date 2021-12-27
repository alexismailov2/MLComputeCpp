#pragma once

#include "CppMLCTypes.h"

#include <vector>

class CppMLCConvolutionLayer;
class CppMLCFullyConnectedLayer;

/*! @class      CppMLCConvolutionDescriptor
    @discussion The CppMLCConvolutionDescriptor specifies a convolution descriptor
 */
class CppMLCConvolutionDescriptor
{
public:
    /*! @property   convolutionType
        @abstract   The type of convolution.
     */
    auto convolutionType() -> eMLCConvolutionType;

    /*! @property   kernelWidth
        @abstract   The convolution kernel size in x.
     */
    auto kernelWidth() -> uint32_t;

    /*! @property   kernelHeight
        @abstract   The convolution kernel size in y.
     */
    auto kernelHeight() -> uint32_t;

    /*! @property   inputFeatureChannelCount
        @abstract   Number of channels in the input tensor
     */
    auto inputFeatureChannelCount() -> uint32_t;

    /*! @property   outputFeatureChannelCount
        @abstract   Number of channels in the output tensor
     */
    auto outputFeatureChannelCount() -> uint32_t;

    /*! @property   strideInX
        @abstract   The stride of the kernel in x.
     */
    auto strideInX() -> uint32_t;

    /*! @property   strideInY
        @abstract   The stride of the kernel in y.
     */
    auto strideInY() -> uint32_t;

    /*! @property   dilationRateInX
        @abstract   The dilation rate i.e. stride of elements in the kernel in x.
     */
    auto dilationRateInX() -> uint32_t;

    /*! @property   dilationRateInY
        @abstract   The dilation rate i.e. stride of elements in the kernel in y.
     */
    auto dilationRateInY() -> uint32_t;

    /*! @property   groupCount
        @abstract   Number of blocked connections from input channels to output channels
     */
    auto groupCount() -> uint32_t;

    /*! @property   paddingPolicy
        @abstract   The padding policy to use.
     */
    auto paddingPolicy() -> eMLCPaddingPolicy;

    /*! @property   paddingSizeInX
        @abstract   The pooling size in x (left and right) to use if paddingPolicy is MLCPaddingPolicyUsePaddingSize
     */
    auto paddingSizeInX() -> uint32_t;

    /*! @property   paddingSizeInY
        @abstract   The pooling size in y (top and bottom) to use if paddingPolicy is MLCPaddingPolicyUsePaddingSize
     */
    auto paddingSizeInY() -> uint32_t;

    /*! @property   isConvolutionTranspose
        @abstract   A flag to indicate if this is a convolution transpose
     */
    bool isConvolutionTranspose();

    /*! @property   usesDepthwiseConvolution
        @abstract   A flag to indicate depthwise convolution
     */
    bool usesDepthwiseConvolution();

    /*! @abstract   Creates a convolution descriptor with the specified convolution type.
        @param      convolutionType              The type of convolution.
        @param      kernelSizes                  The kernel sizes in x and y.
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor.
        @param      outputFeatureChannelCount    The number of feature channels in the output tensor. When the convolution type is \p MLCConvolutionTypeDepthwise , this value must be a multiple of \p inputFeatureChannelCount .
        @param      groupCount                   The number of groups.
        @param      strides                      The kernel strides in x and y.
        @param      dilationRates                The dilation rates in x and y.
        @param      paddingPolicy                The padding policy.
        @param      paddingSizes                 The padding sizes in x and y if padding policy is \p MLCPaddingPolicyUsePaddingSize .
        @return     A new convolution descriptor.
     */
    static auto descriptorWithType(eMLCConvolutionType convolutionType,
                              std::vector<uint32_t> kernelSizes,
                              uint32_t inputFeatureChannelCount,
                              uint32_t outputFeatureChannelCount,
                              uint32_t groupCount,
                              std::vector<uint32_t> strides,
                              std::vector<uint32_t> dilationRates,
                              eMLCPaddingPolicy paddingPolicy,
                              std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object
        @param      kernelWidth                         The kernel size in x
        @param      kernelHeight                       The kernel size in x
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      outputFeatureChannelCount   The number of feature channels in the output tensor
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto descriptorWithKernelWidth(uint32_t kernelWidth,
                                          uint32_t kernelHeight,
                                          uint32_t inputFeatureChannelCount,
                                          uint32_t outputFeatureChannelCount) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object
        @param      kernelSizes                         The kernel sizes in x and y
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      outputFeatureChannelCount   The number of feature channels in the output tensor
        @param      strides                                  The kernel strides in x and y
        @param      paddingPolicy                    The padding policy
        @param      paddingSizes                      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto descriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                          uint32_t inputFeatureChannelCount,
                                          uint32_t outputFeatureChannelCount,
                                          std::vector<uint32_t> strides,
                                          eMLCPaddingPolicy paddingPolicy,
                                          std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object
        @param      kernelSizes                         The kernel sizes in x and y
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      outputFeatureChannelCount   The number of feature channels in the output tensor
        @param      groupCount                           Number of groups
        @param      strides                                  The kernel strides in x and y
        @param      dilationRates                    The dilation rates in x and y
        @param      paddingPolicy                    The padding policy
        @param      paddingSizes                      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto descriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                          uint32_t inputFeatureChannelCount,
                                          uint32_t outputFeatureChannelCount,
                                          uint32_t groupCount,
                                          std::vector<uint32_t> strides,
                                          std::vector<uint32_t> dilationRates,
                                          eMLCPaddingPolicy paddingPolicy,
                                          std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object for convolution transpose
        @param      kernelWidth                         The kernel size in x
        @param      kernelHeight                       The kernel size in x
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      outputFeatureChannelCount   The number of feature channels in the output tensor
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto convolutionTransposeDescriptorWithKernelWidth(uint32_t kernelWidth,
                                                              uint32_t kernelHeight,
                                                              uint32_t inputFeatureChannelCount,
                                                              uint32_t outputFeatureChannelCount) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object for convolution transpose
        @param      kernelSizes                         The kernel sizes in x and y
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      outputFeatureChannelCount   The number of feature channels in the output tensor
        @param      strides                                  The kernel strides in x and y
        @param      paddingPolicy                    The padding policy
        @param      paddingSizes                      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto convolutionTransposeDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                              uint32_t inputFeatureChannelCount,
                                                              uint32_t outputFeatureChannelCount,
                                                              std::vector<uint32_t> strides,
                                                              eMLCPaddingPolicy paddingPolicy,
                                                              std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object for convolution transpose
        @param      kernelSizes                         The kernel sizes in x and y
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      outputFeatureChannelCount   The number of feature channels in the output tensor
        @param      groupCount                           Number of groups
        @param      strides                                  The kernel strides in x and y
        @param      dilationRates                    The dilation rates in x and y
        @param      paddingPolicy                    The padding policy
        @param      paddingSizes                      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto convolutionTransposeDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                              uint32_t inputFeatureChannelCount,
                                                              uint32_t outputFeatureChannelCount,
                                                              uint32_t groupCount,
                                                              std::vector<uint32_t> strides,
                                                              std::vector<uint32_t> dilationRates,
                                                              eMLCPaddingPolicy paddingPolicy,
                                                              std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object for depthwise convolution
        @param      kernelWidth                         The kernel size in x
        @param      kernelHeight                       The kernel size in x
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      channelMultiplier            The channel multiplier
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto depthwiseConvolutionDescriptorWithKernelWidth(uint32_t kernelWidth,
                                                              uint32_t kernelHeight,
                                                              uint32_t inputFeatureChannelCount,
                                                              uint32_t channelMultiplier) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object for depthwise convolution
        @param      kernelSizes                         The kernel sizes in x and y
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      channelMultiplier            The channel multiplier
        @param      strides                                  The kernel strides in x and y
        @param      paddingPolicy                    The padding policy
        @param      paddingSizes                      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto depthwiseConvolutionDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                              uint32_t inputFeatureChannelCount,
                                                              uint32_t channelMultiplier,
                                                              std::vector<uint32_t> strides,
                                                              eMLCPaddingPolicy paddingPolicy,
                                                              std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;

    /*! @abstract   Create a MLCConvolutionDescriptor object for depthwise convolution
        @param      kernelSizes                         The kernel sizes in x and y
        @param      inputFeatureChannelCount     The number of feature channels in the input tensor
        @param      channelMultiplier            The channel multiplier
        @param      strides                                  The kernel strides in x and y
        @param      dilationRates                    The dilation rates in x and y
        @param      paddingPolicy                    The padding policy
        @param      paddingSizes                      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCConvolutionDescriptor object.
     */
    static auto depthwiseConvolutionDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                              uint32_t inputFeatureChannelCount,
                                                              uint32_t channelMultiplier,
                                                              std::vector<uint32_t> strides,
                                                              std::vector<uint32_t> dilationRates,
                                                              eMLCPaddingPolicy paddingPolicy,
                                                              std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor;
    ~CppMLCConvolutionDescriptor();
private:
    CppMLCConvolutionDescriptor(void* self);

private:
    void* self;
    friend CppMLCConvolutionLayer;
    friend CppMLCFullyConnectedLayer;
};
