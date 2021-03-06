#include "CppMLCLossDescriptor.h"

#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCLossDescriptor.h>

auto CppMLCLossDescriptor::lossType() -> eMLCLossType
{
    return MLCLossTypeToCpp(((MLCLossDescriptor*)self).lossType);
}

auto CppMLCLossDescriptor::reductionType() -> eMLCReductionType
{
    return MLCReductionTypeToCpp(((MLCLossDescriptor*)self).reductionType);
}

auto CppMLCLossDescriptor::weight() -> float
{
    return ((MLCLossDescriptor*)self).weight;
}

auto CppMLCLossDescriptor::labelSmoothing() -> float
{
    return ((MLCLossDescriptor*)self).labelSmoothing;
}

auto CppMLCLossDescriptor::classCount() -> uint32_t
{
    return (uint32_t)((MLCLossDescriptor*)self).classCount;
}

auto CppMLCLossDescriptor::epsilon() -> float
{
    return ((MLCLossDescriptor*)self).epsilon;
}

auto CppMLCLossDescriptor::delta() -> float
{
    return ((MLCLossDescriptor*)self).delta;
}

auto CppMLCLossDescriptor::descriptorWithType(eMLCLossType lossType,
                                              eMLCReductionType reductionType) -> CppMLCLossDescriptor
{
    return CppMLCLossDescriptor{[MLCLossDescriptor descriptorWithType:toNative(lossType)
                                                        reductionType:toNative(reductionType)]};
}

auto CppMLCLossDescriptor::descriptorWithType(eMLCLossType lossType,
                                              eMLCReductionType reductionType,
                                              float weight) -> CppMLCLossDescriptor
{
    return CppMLCLossDescriptor{[MLCLossDescriptor descriptorWithType:toNative(lossType)
                                                        reductionType:toNative(reductionType)
                                                               weight:weight]};
}

auto CppMLCLossDescriptor::descriptorWithType(eMLCLossType lossType,
                                              eMLCReductionType reductionType,
                                              float weight,
                                              float labelSmoothing,
                                              uint32_t classCount) -> CppMLCLossDescriptor
{
    return CppMLCLossDescriptor{[MLCLossDescriptor descriptorWithType:toNative(lossType)
                                                        reductionType:toNative(reductionType)
                                                               weight:weight
                                                       labelSmoothing:labelSmoothing
                                                           classCount:(NSUInteger)classCount]};
}

auto CppMLCLossDescriptor::descriptorWithType(eMLCLossType lossType,
                                              eMLCReductionType reductionType,
                                              float weight,
                                              float labelSmoothing,
                                              uint32_t classCount,
                                              float epsilon,
                                              float delta) -> CppMLCLossDescriptor
{
    return CppMLCLossDescriptor{[MLCLossDescriptor descriptorWithType:toNative(lossType)
                                                        reductionType:toNative(reductionType)
                                                               weight:weight
                                                       labelSmoothing:labelSmoothing
                                                           classCount:(NSUInteger)classCount
                                                              epsilon:epsilon
                                                                delta:delta]};
}

CppMLCLossDescriptor::CppMLCLossDescriptor(void *self)
     : self{self}
{
}
