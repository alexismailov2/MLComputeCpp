#include "CppMLCLayerNormalizationLayer.h"

#include "CppMLCTensorParameter.h"
#include "CppMLCTypesPrivate.h"
#include "CppMLCTensor.h"

#import <MLCompute/MLCLayerNormalizationLayer.h>
#import <MLCompute/MLCTensor.h>

auto CppMLCLayerNormalizationLayer::normalizedShape() -> std::vector<uint32_t>
{
    return CppMLCTypesPrivate::NSNumberArrayToVector(((MLCLayerNormalizationLayer*)self).normalizedShape);
}

auto CppMLCLayerNormalizationLayer::beta() -> CppMLCTensor
{
    return CppMLCTensor{((MLCLayerNormalizationLayer*)self).beta};
}

auto CppMLCLayerNormalizationLayer::gamma() -> CppMLCTensor
{
    return CppMLCTensor{((MLCLayerNormalizationLayer*)self).gamma};
}

auto CppMLCLayerNormalizationLayer::betaParameter() -> CppMLCTensorParameter
{
    return CppMLCTensorParameter{((MLCLayerNormalizationLayer*)self).betaParameter};
}

auto CppMLCLayerNormalizationLayer::gammaParameter() -> CppMLCTensorParameter
{
    return CppMLCTensorParameter{((MLCLayerNormalizationLayer*)self).gammaParameter};
}

auto CppMLCLayerNormalizationLayer::varianceEpsilon() -> float
{
    return ((MLCLayerNormalizationLayer*)self).varianceEpsilon;
}

auto CppMLCLayerNormalizationLayer::layerWithNormalizedShape(std::vector<uint32_t> const& normalizedShape,
                                                             CppMLCTensor& beta,
                                                             CppMLCTensor& gamma,
                                                             float varianceEpsilon) -> CppMLCLayerNormalizationLayer
{
    return CppMLCLayerNormalizationLayer{[MLCLayerNormalizationLayer layerWithNormalizedShape:CppMLCTypesPrivate::toNSArray(normalizedShape)
                                                                                         beta:(MLCTensor*)beta.self
                                                                                        gamma:(MLCTensor*)gamma.self
                                                                              varianceEpsilon:varianceEpsilon]};
}

CppMLCLayerNormalizationLayer::CppMLCLayerNormalizationLayer(void *self)
    : CppMLCLayer(self)
{
}
