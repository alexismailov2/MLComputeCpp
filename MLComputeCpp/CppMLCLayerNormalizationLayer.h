#pragma once

#include "CppMLCLayer.h"

class CppMLCTensorParameter;

/*! @class      MLCLayerNormalizationLayer
    @abstract   The layer normalizaion layer.  For more information, refer to https://pytorch.org/docs/stable/nn.html#layernorm.
 */
class CppMLCLayerNormalizationLayer : public CppMLCLayer
{
public:
    /*! @property   normalizedShape
        @abstract   The shape of the axes over which normalization occurs, (W), (H,W) or (C,H,W)
     */
    auto normalizedShape() -> std::vector<uint32_t>;

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

    /*! @abstract Create a layer normalization layer
        @param normalizedShape The shape of the axes over which normalization occurs, currently (C,H,W) only
        @param beta Training parameter
        @param gamma Training parameter
        @param varianceEpsilon  A small numerical value added to variance for stability
        @return A new layer normalization layer.
     */
    auto layerWithNormalizedShape(std::vector<uint32_t> const& normalizedShape,
                                  CppMLCTensor& beta,
                                  CppMLCTensor& gamma,
                                  float varianceEpsilon) -> CppMLCLayerNormalizationLayer;

private:
    CppMLCLayerNormalizationLayer(void* self);
};
