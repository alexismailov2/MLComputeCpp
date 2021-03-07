#include "CppMLCSliceLayer.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCSliceLayer.h>

auto CppMLCSliceLayer::start() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCSliceLayer*)self).start);
}

auto CppMLCSliceLayer::end() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCSliceLayer*)self).end);
}

auto CppMLCSliceLayer::stride() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCSliceLayer*)self).stride);
}

auto CppMLCSliceLayer::sliceLayerWithStart(std::vector<uint32_t> start,
                                           std::vector<uint32_t> end,
                                           std::vector<uint32_t> stride) -> CppMLCSliceLayer
{
    return CppMLCSliceLayer{[MLCSliceLayer sliceLayerWithStart:CppMLCTypesPrivate::toNSArray(start)
                                                           end:CppMLCTypesPrivate::toNSArray(end)
                                                        stride:CppMLCTypesPrivate::toNSArray(stride)]};
}

CppMLCSliceLayer::CppMLCSliceLayer(void *self)
    : CppMLCLayer(self)
{
}
