#pragma once

#include "CppMLCLayer.h"
#include "CppMLCTensorParameter.h"
#include "CppMLCTensor.h"

/*! @class      MLCGroupNormalizationLayer
    @abstract   A group normalizaion layer.  For more information, refer to https://pytorch.org/docs/stable/nn.html#groupnorm
 */
class CppMLCGroupNormalizationLayer : public CppMLCLayer
{
public:
    /*! @property   featureChannelCount
        @abstract   The number of feature channels
     */
    auto featureChannelCount() -> uint32_t;

    /*! @property   groupCount
        @abstract   The number of groups to separate the channels into
     */
    auto groupCount() -> uint32_t;

    /*! @property   beta
        @abstract   The beta tensor
     */
    auto beta() -> CppMLCTensor;

    /*! @property   gamma
        @abstract   The gamma tensor
     */
    auto gamma() -> CppMLCTensor;

    /*! @property   betaParameter
        @abstract   The beta tensor parameter used for optimizer update
     */
    auto betaParameter() -> CppMLCTensorParameter;

    /*! @property   gammaParameter
        @abstract   The gamma tensor parameter used for optimizer update
     */
    auto gammaParameter() -> CppMLCTensorParameter;

    /*! @property   varianceEpsilon
        @abstract   A value used for numerical stability
     */
    auto varianceEpsilon() -> float;

    /*! @abstract Create a group normalization layer
        @param featureChannelCount  The number of feature channels
        @param beta  Training parameter
        @param gamma  Training parameter
        @param groupCount  The number of groups to divide the feature channels into
        @param varianceEpsilon  A small numerical value added to variance for stability
        @return A new group normalization layer.
     */
    static auto layerWithFeatureChannelCount(uint32_t featureChannelCount,
                                             uint32_t groupCount,
                                             CppMLCTensor const& beta,
                                             CppMLCTensor const& gamma,
                                             float varianceEpsilon) -> CppMLCGroupNormalizationLayer;

private:
    CppMLCGroupNormalizationLayer(void* self);

private:
    void* self;
};
