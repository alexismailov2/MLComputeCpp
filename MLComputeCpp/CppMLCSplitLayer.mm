#include "CppMLCSplitLayer.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCSplitLayer.h>

auto CppMLCSplitLayer::dimension() -> uint32_t
{
    return (uint32_t)((MLCSplitLayer*)self).dimension;
}

auto CppMLCSplitLayer::splitCount() -> uint32_t
{
    return (uint32_t)((MLCSplitLayer*)self).splitCount;
}

auto CppMLCSplitLayer::splitSectionLengths() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCSplitLayer*)self).splitSectionLengths);
}

auto CppMLCSplitLayer::layerWithSplitCount(uint32_t splitCount,
                                           uint32_t dimension) -> CppMLCSplitLayer
{
    return CppMLCSplitLayer{[MLCSplitLayer layerWithSplitCount:(NSUInteger)splitCount
                                                     dimension:(NSUInteger)dimension]};
}

auto CppMLCSplitLayer::layerWithSplitSectionLengths(std::vector<uint32_t> const& splitSectionLengths,
                                                    uint32_t dimension) -> CppMLCSplitLayer
{
    return CppMLCSplitLayer{[MLCSplitLayer layerWithSplitSectionLengths:CppMLCTypesPrivate::toNSArray(splitSectionLengths)
                                                              dimension:(NSUInteger)dimension]};
}

CppMLCSplitLayer::CppMLCSplitLayer(void *self)
    : CppMLCLayer(self)
{
}
