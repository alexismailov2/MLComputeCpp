#include "CppMLCEmbeddingDescriptor.h"

#import <MLCompute/MLCEmbeddingDescriptor.h>

auto CppMLCEmbeddingDescriptor::embeddingCount() -> uint32_t {
    return (uint32_t)((MLCEmbeddingDescriptor*)self).embeddingCount.unsignedIntegerValue;
}

auto CppMLCEmbeddingDescriptor::embeddingDimension() -> uint32_t {
    return (uint32_t)((MLCEmbeddingDescriptor*)self).embeddingDimension.unsignedIntegerValue;
}

auto CppMLCEmbeddingDescriptor::paddingIndex() -> uint32_t {
    return (uint32_t)((MLCEmbeddingDescriptor*)self).paddingIndex.unsignedIntegerValue;
}

auto CppMLCEmbeddingDescriptor::maximumNorm() -> float {
    return (uint32_t)((MLCEmbeddingDescriptor*)self).maximumNorm.floatValue;
}

auto CppMLCEmbeddingDescriptor::pNorm() -> float {
    return (uint32_t)((MLCEmbeddingDescriptor*)self).pNorm.floatValue;
}

bool CppMLCEmbeddingDescriptor::scalesGradientByFrequency() {
    return ((MLCEmbeddingDescriptor*)self).scalesGradientByFrequency == YES;
}

auto CppMLCEmbeddingDescriptor::descriptorWithEmbeddingCount(uint32_t embeddingCount,
                                                             uint32_t embeddingDimension) -> CppMLCEmbeddingDescriptor {
    return CppMLCEmbeddingDescriptor{[MLCEmbeddingDescriptor descriptorWithEmbeddingCount:[NSNumber numberWithUnsignedInteger:(NSUInteger)embeddingCount]
                                                                       embeddingDimension:[NSNumber numberWithUnsignedInteger:(NSUInteger)embeddingDimension]]};
}

auto CppMLCEmbeddingDescriptor::descriptorWithEmbeddingCount(uint32_t embeddingCount,
                                                             uint32_t embeddingDimension,
                                                             uint32_t paddingIndex,
                                                             uint32_t maximumNorm,
                                                             uint32_t pNorm,
                                                             bool scalesGradientByFrequency) -> CppMLCEmbeddingDescriptor {
return CppMLCEmbeddingDescriptor{[MLCEmbeddingDescriptor descriptorWithEmbeddingCount:[NSNumber numberWithUnsignedInteger:(NSUInteger)embeddingCount]
                                                                   embeddingDimension:[NSNumber numberWithUnsignedInteger:(NSUInteger)embeddingDimension]
                                                                         paddingIndex:[NSNumber numberWithUnsignedInteger:(NSUInteger)paddingIndex]
                                                                          maximumNorm:[NSNumber numberWithUnsignedInteger:(NSUInteger)maximumNorm]
                                                                                pNorm:[NSNumber numberWithUnsignedInteger:(NSUInteger)pNorm]
                                                            scalesGradientByFrequency:scalesGradientByFrequency ? YES : NO]};
}

CppMLCEmbeddingDescriptor::CppMLCEmbeddingDescriptor(void* self)
    : self{self}
{
}
