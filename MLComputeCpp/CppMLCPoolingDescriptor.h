#pragma once

#include "CppMLCTypes.h"

#include <vector>

class CppMLCPoolingLayer;

/*! @class      MLCPoolingDescriptor
    @discussion The MLCPoolingDescriptor specifies a pooling descriptor.
 */
class CppMLCPoolingDescriptor
{
public:
    /*! @property   poolingType
        @abstract   The pooling operation
     */
    auto poolingType() -> eMLCPoolingType;

    /*! @property   kernelWidth
        @abstract   The pooling kernel size in x.
     */
    auto kernelWidth() -> uint32_t;

    /*! @property   kernelHeight
        @abstract   The pooling kernel size in y.
     */
    auto kernelHeight() -> uint32_t;

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

    /*! @property   paddingPolicy
        @abstract   The padding policy to use.
     */
    auto paddingPolicy() -> eMLCPaddingPolicy;

    /*! @property   paddingSizeInX
        @abstract   The padding size in x (left and right) to use if paddingPolicy is MLCPaddingPolicyUsePaddingSize
     */
    auto paddingSizeInX() -> uint32_t;

    /*! @property   paddingSizeInY
        @abstract   The padding size in y (top and bottom) to use if paddingPolicy is MLCPaddingPolicyUsePaddingSize
     */
    auto paddingSizeInY() -> uint32_t;

    /*! @property   countIncludesPadding
        @abstract   Include the zero-padding in the averaging calculation if true.  Used only with average pooling.
     */
    bool countIncludesPadding();

    /*! @abstract   Create a MLCPoolingDescriptor object
        @param      poolingType    The pooling function
        @param      kernelSize      The kernel sizes in x and y
        @param      stride               The kernel strides in x and y
        @return     A new MLCPoolingDescriptor object.
     */
    auto poolingDescriptorWithType(eMLCPoolingType poolingType,
                                   uint32_t kernelSize,
                                   uint32_t stride) -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a MLCPoolingDescriptor object for a max pooling function
        @param      kernelSizes        The kernel sizes in x and y
        @param      strides                 The kernel strides in x and y
        @param      paddingPolicy    The padding policy
        @param      paddingSizes      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCPoolingDescriptor object.
     */
    auto maxPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                             std::vector<uint32_t> const& strides,
                                             eMLCPaddingPolicy paddingPolicy,
                                             std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a MLCPoolingDescriptor object for a max pooling function
        @param      kernelSizes        The kernel sizes in x and y
        @param      strides                 The kernel strides in x and y
        @param      dilationRates    The kernel dilation rates in x and y
        @param      paddingPolicy    The padding policy
        @param      paddingSizes      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCPoolingDescriptor object.
     */
    auto maxPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                             std::vector<uint32_t> const& strides,
                                             std::vector<uint32_t> const& dilationRates,
                                             eMLCPaddingPolicy paddingPolicy,
                                             std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a MLCPoolingDescriptor object for an average pooling function
        @param      kernelSizes                      The kernel sizes in x and y
        @param      strides                               The kernel strides in x and y
        @param      paddingPolicy                  The padding policy
        @param      paddingSizes                    The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @param      countIncludesPadding   Whether to include zero padding in the averaging calculation
        @return     A new MLCPoolingDescriptor object.
     */
    auto averagePoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                 std::vector<uint32_t> const& strides,
                                                 eMLCPaddingPolicy paddingPolicy,
                                                 std::vector<uint32_t> const& paddingSizes,
                                                 bool countIncludesPadding) -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a MLCPoolingDescriptor object for an average pooling function
        @param      kernelSizes                      The kernel sizes in x and y
        @param      strides                               The kernel strides in x and y
        @param      dilationRates                  The kernel dilation rates in x and y
        @param      paddingPolicy                  The padding policy
        @param      paddingSizes                    The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @param      countIncludesPadding   Whether to include zero padding in the averaging calculation
        @return     A new MLCPoolingDescriptor object.
     */
    auto averagePoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                 std::vector<uint32_t> const& strides,
                                                 std::vector<uint32_t> const& dilationRates,
                                                 eMLCPaddingPolicy paddingPolicy,
                                                 std::vector<uint32_t> const& paddingSizes,
                                                 bool countIncludesPadding) -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a MLCPoolingDescriptor object for a L2 norm pooling function
        @param      kernelSizes        The kernel sizes in x and y
        @param      strides                 The kernel strides in x and y
        @param      paddingPolicy    The padding policy
        @param      paddingSizes      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCPoolingDescriptor object.
     */
    auto l2NormPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                std::vector<uint32_t> const& strides,
                                                eMLCPaddingPolicy paddingPolicy,
                                                std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a MLCPoolingDescriptor object for a L2 norm pooling function
        @param      kernelSizes        The kernel sizes in x and y
        @param      strides                 The kernel strides in x and y
        @param      dilationRates    The kernel dilation rates in x and y
        @param      paddingPolicy    The padding policy
        @param      paddingSizes      The padding sizes in x and y if padding policy is MLCPaddingPolicyUsePaddingSIze
        @return     A new MLCPoolingDescriptor object.
     */
    auto l2NormPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                std::vector<uint32_t> const& strides,
                                                std::vector<uint32_t> const& dilationRates,
                                                eMLCPaddingPolicy paddingPolicy,
                                                std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor;

private:
    CppMLCPoolingDescriptor(void* self);

private:
    void* self;
    friend CppMLCPoolingLayer;
};
