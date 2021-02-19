#include "CppMLCDropoutLayer.h"

#import <MLCompute/MLCDropoutLayer.h>

auto CppMLCDropoutLayer::rate() -> float {
    return ((MLCDropoutLayer*)self).rate;
}

auto CppMLCDropoutLayer::seed() -> uint32_t {
    return (uint32_t)((MLCDropoutLayer*)self).seed;
}

auto CppMLCDropoutLayer::layerWithRate(float rate, uint32_t seed) -> CppMLCDropoutLayer {
    return CppMLCDropoutLayer{[MLCDropoutLayer layerWithRate:rate
                                                        seed:(NSUInteger)seed]};
}

CppMLCDropoutLayer::CppMLCDropoutLayer(void* self)
    : CppMLCLayer{self}
    , self{self}
{
}
