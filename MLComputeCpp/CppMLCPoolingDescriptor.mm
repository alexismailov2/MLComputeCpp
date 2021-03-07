#include "CppMLCPoolingDescriptor.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCPoolingDescriptor.h>

auto CppMLCPoolingDescriptor::poolingType() -> eMLCPoolingType
{
    return MLCPoolingTypeToCpp(((MLCPoolingDescriptor*)self).poolingType);
}

auto CppMLCPoolingDescriptor::kernelWidth() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).kernelWidth;
}

auto CppMLCPoolingDescriptor::kernelHeight() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).kernelHeight;
}

auto CppMLCPoolingDescriptor::strideInX() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).strideInX;
}

auto CppMLCPoolingDescriptor::strideInY() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).strideInY;
}

auto CppMLCPoolingDescriptor::dilationRateInX() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).dilationRateInX;
}

auto CppMLCPoolingDescriptor::dilationRateInY() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).dilationRateInY;
}

auto CppMLCPoolingDescriptor::paddingPolicy() -> eMLCPaddingPolicy
{
    return MLCPaddingPolicyToCpp(((MLCPoolingDescriptor*)self).paddingPolicy);
}

auto CppMLCPoolingDescriptor::paddingSizeInX() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).paddingSizeInX;
}

auto CppMLCPoolingDescriptor::paddingSizeInY() -> uint32_t
{
    return (uint32_t)((MLCPoolingDescriptor*)self).paddingSizeInY;
}

bool CppMLCPoolingDescriptor::countIncludesPadding()
{
    return ((MLCPoolingDescriptor*)self).countIncludesPadding == YES;
}

auto CppMLCPoolingDescriptor::poolingDescriptorWithType(eMLCPoolingType poolingType,
                                                        uint32_t kernelSize,
                                                        uint32_t stride) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor poolingDescriptorWithType:toNative(poolingType)
                                                                        kernelSize:(NSUInteger)kernelSize
                                                                            stride:(NSUInteger)stride]};
}

auto CppMLCPoolingDescriptor::maxPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                                  std::vector<uint32_t> const& strides,
                                                                  eMLCPaddingPolicy paddingPolicy,
                                                                  std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor maxPoolingDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                                                     strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                               paddingPolicy:toNative(paddingPolicy)
                                                                                paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCPoolingDescriptor::maxPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                                  std::vector<uint32_t> const& strides,
                                                                  std::vector<uint32_t> const& dilationRates,
                                                                  eMLCPaddingPolicy paddingPolicy,
                                                                  std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor maxPoolingDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                                                     strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                               dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                                                               paddingPolicy:toNative(paddingPolicy)
                                                                                paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCPoolingDescriptor::averagePoolingDescriptorWithKernelSizes(const std::vector<uint32_t> &kernelSizes,
                                                                      const std::vector<uint32_t> &strides,
                                                                      eMLCPaddingPolicy paddingPolicy,
                                                                      const std::vector<uint32_t> &paddingSizes,
                                                                      bool countIncludesPadding) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor averagePoolingDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                                                         strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                                   paddingPolicy:toNative(paddingPolicy)
                                                                                    paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)
                                                                            countIncludesPadding:countIncludesPadding ? YES : NO]};
}

auto CppMLCPoolingDescriptor::averagePoolingDescriptorWithKernelSizes(const std::vector<uint32_t> &kernelSizes,
                                                                      const std::vector<uint32_t> &strides,
                                                                      const std::vector<uint32_t> &dilationRates,
                                                                      eMLCPaddingPolicy paddingPolicy,
                                                                      const std::vector<uint32_t> &paddingSizes,
                                                                      bool countIncludesPadding) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor averagePoolingDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                                                         strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                                   dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                                                                   paddingPolicy:toNative(paddingPolicy)
                                                                                    paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)
                                                                            countIncludesPadding:countIncludesPadding ? YES : NO]};
}

auto CppMLCPoolingDescriptor::l2NormPoolingDescriptorWithKernelSizes(std::vector<uint32_t> const& kernelSizes,
                                                                     std::vector<uint32_t> const& strides,
                                                                     eMLCPaddingPolicy paddingPolicy,
                                                                     std::vector<uint32_t> const& paddingSizes) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor l2NormPoolingDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                                                        strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                                  paddingPolicy:toNative(paddingPolicy)
                                                                                    paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCPoolingDescriptor::l2NormPoolingDescriptorWithKernelSizes(const std::vector<uint32_t> &kernelSizes,
                                                                     const std::vector<uint32_t> &strides,
                                                                     const std::vector<uint32_t> &dilationRates,
                                                                     eMLCPaddingPolicy paddingPolicy,
                                                                     const std::vector<uint32_t> &paddingSizes) -> CppMLCPoolingDescriptor
{
    return CppMLCPoolingDescriptor{[MLCPoolingDescriptor l2NormPoolingDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                                                        strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                                  dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                                                                  paddingPolicy:toNative(paddingPolicy)
                                                                                   paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

CppMLCPoolingDescriptor::CppMLCPoolingDescriptor(void *self)
    : self{self} 
{
}
