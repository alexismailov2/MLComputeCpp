#include "CppMLCConvolutionDescriptor.h"
#include "CppMLCTypesPrivate.h"

#import <MLCompute//MLCConvolutionDescriptor.h>

auto CppMLCConvolutionDescriptor::convolutionType() -> eMLCConvolutionType {
    return MLCConvolutionTypeToCpp(((MLCConvolutionDescriptor*)self).convolutionType);
}

auto CppMLCConvolutionDescriptor::kernelWidth() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).kernelWidth;
}

auto CppMLCConvolutionDescriptor::kernelHeight() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).kernelHeight;
}

auto CppMLCConvolutionDescriptor::inputFeatureChannelCount() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).inputFeatureChannelCount;
}

auto CppMLCConvolutionDescriptor::outputFeatureChannelCount() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).outputFeatureChannelCount;
}

auto CppMLCConvolutionDescriptor::strideInX() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).strideInX;
}

auto CppMLCConvolutionDescriptor::strideInY() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).strideInY;
}

auto CppMLCConvolutionDescriptor::dilationRateInX() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).dilationRateInX;
}

auto CppMLCConvolutionDescriptor::dilationRateInY() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).dilationRateInY;
}

auto CppMLCConvolutionDescriptor::groupCount() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).groupCount;
}

auto CppMLCConvolutionDescriptor::paddingPolicy() -> eMLCPaddingPolicy {
    return MLCPaddingPolicyToCpp(((MLCConvolutionDescriptor*)self).paddingPolicy);
}

auto CppMLCConvolutionDescriptor::paddingSizeInX() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).paddingSizeInX;
}

auto CppMLCConvolutionDescriptor::paddingSizeInY() -> uint32_t {
    return (uint32_t)((MLCConvolutionDescriptor*)self).paddingSizeInY;
}

bool CppMLCConvolutionDescriptor::isConvolutionTranspose() {
    return ((MLCConvolutionDescriptor*)self).isConvolutionTranspose == YES;
}

bool CppMLCConvolutionDescriptor::usesDepthwiseConvolution() {
    return ((MLCConvolutionDescriptor*)self).usesDepthwiseConvolution == YES;
}

auto
CppMLCConvolutionDescriptor::descriptorWithType(eMLCConvolutionType convolutionType,
                                                std::vector<uint32_t> kernelSizes,
                                                uint32_t inputFeatureChannelCount,
                                                uint32_t outputFeatureChannelCount,
                                                uint32_t groupCount,
                                                std::vector<uint32_t> strides,
                                                std::vector<uint32_t> dilationRates,
                                                eMLCPaddingPolicy paddingPolicy,
                                                std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor descriptorWithType:toNative(convolutionType)
                                             kernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                       inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                      outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount
                                              groupCount:(NSUInteger)groupCount
                                                 strides:CppMLCTypesPrivate::toNSArray(strides)
                                           dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                           paddingPolicy:toNative(paddingPolicy)
                                            paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCConvolutionDescriptor::descriptorWithKernelWidth(uint32_t kernelWidth,
                                                            uint32_t kernelHeight,
                                                            uint32_t inputFeatureChannelCount,
                                                            uint32_t outputFeatureChannelCount) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor descriptorWithKernelWidth:(NSUInteger)kernelWidth
                                                   kernelHeight:(NSUInteger)kernelHeight
                                       inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                      outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount]};
}

auto CppMLCConvolutionDescriptor::descriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                            uint32_t inputFeatureChannelCount,
                                                            uint32_t outputFeatureChannelCount,
                                                            std::vector<uint32_t> strides,
                                                            eMLCPaddingPolicy paddingPolicy,
                                                            std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor descriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                       inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                      outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount
                                                        strides:CppMLCTypesPrivate::toNSArray(strides)
                                                  paddingPolicy:toNative(paddingPolicy)
                                                   paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCConvolutionDescriptor::descriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                            uint32_t inputFeatureChannelCount,
                                                            uint32_t outputFeatureChannelCount,
                                                            uint32_t groupCount,
                                                            std::vector<uint32_t> strides,
                                                            std::vector<uint32_t> dilationRates,
                                                            eMLCPaddingPolicy paddingPolicy,
                                                            std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor descriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                       inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                      outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount
                                                     groupCount:(NSUInteger)groupCount
                                                        strides:CppMLCTypesPrivate::toNSArray(strides)
                                                  dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                                  paddingPolicy:toNative(paddingPolicy)
                                                   paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto
CppMLCConvolutionDescriptor::convolutionTransposeDescriptorWithKernelWidth(uint32_t kernelWidth,
                                                                           uint32_t kernelHeight,
                                                                           uint32_t inputFeatureChannelCount,
                                                                           uint32_t outputFeatureChannelCount) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor convolutionTransposeDescriptorWithKernelWidth:(NSUInteger)kernelWidth
                                                                       kernelHeight:(NSUInteger)kernelHeight
                                                           inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                                          outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount]};
}

auto CppMLCConvolutionDescriptor::convolutionTransposeDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                                                uint32_t inputFeatureChannelCount,
                                                                                uint32_t outputFeatureChannelCount,
                                                                                std::vector<uint32_t> strides,
                                                                                eMLCPaddingPolicy paddingPolicy,
                                                                                std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor convolutionTransposeDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                           inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                                          outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount
                                                                            strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                      paddingPolicy:toNative(paddingPolicy)
                                                                       paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCConvolutionDescriptor::convolutionTransposeDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                                                uint32_t inputFeatureChannelCount,
                                                                                uint32_t outputFeatureChannelCount,
                                                                                uint32_t groupCount,
                                                                                std::vector<uint32_t> strides,
                                                                                std::vector<uint32_t> dilationRates,
                                                                                eMLCPaddingPolicy paddingPolicy,
                                                                                std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor convolutionTransposeDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                           inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                                          outputFeatureChannelCount:(NSUInteger)outputFeatureChannelCount
                                                                         groupCount:(NSUInteger)groupCount
                                                                            strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                      dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                                                      paddingPolicy:toNative(paddingPolicy)
                                                                       paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto
CppMLCConvolutionDescriptor::depthwiseConvolutionDescriptorWithKernelWidth(uint32_t kernelWidth,
                                                                           uint32_t kernelHeight,
                                                                           uint32_t inputFeatureChannelCount,
                                                                           uint32_t channelMultiplier) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor depthwiseConvolutionDescriptorWithKernelWidth:(NSUInteger)kernelWidth
                                                                       kernelHeight:(NSUInteger)kernelHeight
                                                           inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                                                  channelMultiplier:(NSUInteger)channelMultiplier]};

}

auto CppMLCConvolutionDescriptor::depthwiseConvolutionDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                                                uint32_t inputFeatureChannelCount,
                                                                                uint32_t channelMultiplier,
                                                                                std::vector<uint32_t> strides,
                                                                                eMLCPaddingPolicy paddingPolicy,
                                                                                std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
            [MLCConvolutionDescriptor depthwiseConvolutionDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                           inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                                                  channelMultiplier:(NSUInteger)channelMultiplier
                                                                            strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                      paddingPolicy:toNative(paddingPolicy)
                                                                       paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

auto CppMLCConvolutionDescriptor::depthwiseConvolutionDescriptorWithKernelSizes(std::vector<uint32_t> kernelSizes,
                                                                                uint32_t inputFeatureChannelCount,
                                                                                uint32_t channelMultiplier,
                                                                                std::vector<uint32_t> strides,
                                                                                std::vector<uint32_t> dilationRates,
                                                                                eMLCPaddingPolicy paddingPolicy,
                                                                                std::vector<uint32_t> paddingSizes) -> CppMLCConvolutionDescriptor {
    return CppMLCConvolutionDescriptor{
        [MLCConvolutionDescriptor depthwiseConvolutionDescriptorWithKernelSizes:CppMLCTypesPrivate::toNSArray(kernelSizes)
                                                       inputFeatureChannelCount:(NSUInteger)inputFeatureChannelCount
                                                              channelMultiplier:(NSUInteger)channelMultiplier
                                                                        strides:CppMLCTypesPrivate::toNSArray(strides)
                                                                  dilationRates:CppMLCTypesPrivate::toNSArray(dilationRates)
                                                                  paddingPolicy:toNative(paddingPolicy)
                                                                   paddingSizes:CppMLCTypesPrivate::toNSArray(paddingSizes)]};
}

CppMLCConvolutionDescriptor::CppMLCConvolutionDescriptor(void *self)
    : self{self}
{
}
