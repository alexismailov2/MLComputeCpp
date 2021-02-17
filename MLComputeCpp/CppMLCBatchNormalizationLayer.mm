#include "CppMLCBatchNormalizationLayer.h"
#include "CppMLCTensor.h"
#include "CppMLCTensorParameter.h"

#import <MLCompute/MLCBatchNormalizationLayer.h>

CppMLCBatchNormalizationLayer::CppMLCBatchNormalizationLayer(void *self)
    : CppMLCLayer(self)
    , self{self}
{
}

auto CppMLCBatchNormalizationLayer::featureChannelCount() -> uint32_t { return (uint32_t)((MLCBatchNormalizationLayer*)self).featureChannelCount; }

auto CppMLCBatchNormalizationLayer::mean() -> CppMLCTensor { return CppMLCTensor{((MLCBatchNormalizationLayer*)self).mean}; }

auto CppMLCBatchNormalizationLayer::variance() -> CppMLCTensor { return CppMLCTensor{((MLCBatchNormalizationLayer*)self).variance}; }

auto CppMLCBatchNormalizationLayer::beta() -> CppMLCTensor { return CppMLCTensor{((MLCBatchNormalizationLayer*)self).beta}; }

auto CppMLCBatchNormalizationLayer::gamma() -> CppMLCTensor { return CppMLCTensor{((MLCBatchNormalizationLayer*)self).gamma}; }

auto CppMLCBatchNormalizationLayer::betaParameter() -> CppMLCTensorParameter { return CppMLCTensorParameter{((MLCBatchNormalizationLayer*)self).betaParameter}; }

auto CppMLCBatchNormalizationLayer::gammaParameter() -> CppMLCTensorParameter { return CppMLCTensorParameter{((MLCBatchNormalizationLayer*)self).gammaParameter}; }

auto CppMLCBatchNormalizationLayer::varianceEpsilon() -> float { return ((MLCBatchNormalizationLayer*)self).varianceEpsilon; }

auto CppMLCBatchNormalizationLayer::momentum() -> float { return ((MLCBatchNormalizationLayer*)self).momentum; }

CppMLCBatchNormalizationLayer
CppMLCBatchNormalizationLayer::layerWithFeatureChannelCount(uint32_t featureChannelCount,
                                                            const CppMLCTensor &mean,
                                                            const CppMLCTensor &variance,
                                                            const CppMLCTensor &beta,
                                                            const CppMLCTensor &gamma,
                                                            float varianceEpsilon) {
    return CppMLCBatchNormalizationLayer{[MLCBatchNormalizationLayer layerWithFeatureChannelCount:(NSUInteger)featureChannelCount
                                                                                             mean:(MLCTensor*)mean.self
                                                                                         variance:(MLCTensor*)variance.self
                                                                                             beta:(MLCTensor*)beta.self
                                                                                            gamma:(MLCTensor*)gamma.self
                                                                                  varianceEpsilon:varianceEpsilon]};

}

CppMLCBatchNormalizationLayer
CppMLCBatchNormalizationLayer::layerWithFeatureChannelCount(uint32_t featureChannelCount,
                                                            const CppMLCTensor &mean,
                                                            const CppMLCTensor &variance,
                                                            const CppMLCTensor &beta,
                                                            const CppMLCTensor &gamma,
                                                            float varianceEpsilon,
                                                            float momentum) {
    return CppMLCBatchNormalizationLayer{[MLCBatchNormalizationLayer layerWithFeatureChannelCount:(NSUInteger)featureChannelCount
                                                                                             mean:(MLCTensor*)mean.self
                                                                                         variance:(MLCTensor*)variance.self
                                                                                             beta:(MLCTensor*)beta.self
                                                                                            gamma:(MLCTensor*)gamma.self
                                                                                  varianceEpsilon:varianceEpsilon
                                                                                         momentum:momentum]};
}
