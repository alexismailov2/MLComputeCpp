#include "CppMLCGroupNormalizationLayer.h"

#import <MLCompute/MLCGroupNormalizationLayer.h>

auto CppMLCGroupNormalizationLayer::featureChannelCount() -> uint32_t {
    return (uint32_t)((MLCGroupNormalizationLayer*)self).featureChannelCount;
}

auto CppMLCGroupNormalizationLayer::groupCount() -> uint32_t {
    return (uint32_t)((MLCGroupNormalizationLayer*)self).groupCount;
}

auto CppMLCGroupNormalizationLayer::beta() -> CppMLCTensor {
    return CppMLCTensor{((MLCGroupNormalizationLayer*)self).beta};
}

auto CppMLCGroupNormalizationLayer::gamma() -> CppMLCTensor {
    return CppMLCTensor{((MLCGroupNormalizationLayer*)self).gamma};
}

auto CppMLCGroupNormalizationLayer::betaParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCGroupNormalizationLayer*)self).betaParameter};
}

auto CppMLCGroupNormalizationLayer::gammaParameter() -> CppMLCTensorParameter {
    return CppMLCTensorParameter{((MLCGroupNormalizationLayer*)self).gammaParameter};
}

auto CppMLCGroupNormalizationLayer::varianceEpsilon() -> float {
    return ((MLCGroupNormalizationLayer*)self).varianceEpsilon;
}

auto CppMLCGroupNormalizationLayer::layerWithFeatureChannelCount(uint32_t featureChannelCount,
                                                                 uint32_t groupCount,
                                                                 CppMLCTensor const& beta,
                                                                 CppMLCTensor const& gamma,
                                                                 float varianceEpsilon) -> CppMLCGroupNormalizationLayer {
    return CppMLCGroupNormalizationLayer{
            [MLCGroupNormalizationLayer layerWithFeatureChannelCount:(NSUInteger)featureChannelCount
                                                          groupCount:(NSUInteger)groupCount
                                                                beta:(MLCTensor*)beta.self
                                                               gamma:(MLCTensor*)gamma.self
                                                     varianceEpsilon:varianceEpsilon]};
}

CppMLCGroupNormalizationLayer::CppMLCGroupNormalizationLayer(void *self)
    : CppMLCLayer(self)
    , self{self}
{
}
