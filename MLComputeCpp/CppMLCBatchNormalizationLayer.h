#pragma once

#import "CppMLCLayer.h"

class CppMLCDevice;
class CppMLCTensor;
class CppMLCTensorParameter;
class CppMLCActivationLayer;

/*! @class      MLCBatchNormalizationLayer
    @abstract   A batch normalizaion layer
 */
class CppMLCBatchNormalizationLayer : public CppMLCLayer
{
public:
    /*! @property   featureChannelCount
        @abstract   The number of feature channels
     */
    auto featureChannelCount() -> uint32_t;

    /*! @property   mean
        @abstract   The mean tensor
     */
    auto mean() -> CppMLCTensor;

    /*! @property   variance
        @abstract   The variance tensor
     */
    auto variance() -> CppMLCTensor;

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

    /*! @property   momentum
        @abstract   The value used for the running mean and variance computation
        @discussion The default is 0.99f.
     */
    auto momentum() -> float;

    /*! @abstract   Create a batch normalization layer
        @param featureChannelCount The number of feature channels
        @param mean The mean tensor
        @param variance The variance tensor
        @param beta The beta tensor
        @param gamma The gamma tensor
        @param varianceEpsilon The  epslion value
        @return     A new batch normalization layer.
     */
    static CppMLCBatchNormalizationLayer layerWithFeatureChannelCount(uint32_t featureChannelCount,
                                                                      CppMLCTensor const& mean,
                                                                      CppMLCTensor const& variance,
                                                                      CppMLCTensor const& beta,
                                                                      CppMLCTensor const& gamma,
                                                                      float varianceEpsilon);

    /*! @abstract   Create a batch normalization layer
        @param featureChannelCount The number of feature channels
        @param mean The mean tensor
        @param variance The variance tensor
        @param beta The beta tensor
        @param gamma The gamma tensor
        @param varianceEpsilon The  epslion value
        @param momentum The  momentum value for the running mean and variance computation
        @return A new batch normalization layer.
     */
    static CppMLCBatchNormalizationLayer layerWithFeatureChannelCount(uint32_t featureChannelCount,
                                                                      CppMLCTensor const& mean,
                                                                      CppMLCTensor const& variance,
                                                                      CppMLCTensor const& beta,
                                                                      CppMLCTensor const& gamma,
                                                                      float varianceEpsilon,
                                                                      float momentum);

protected:
    CppMLCBatchNormalizationLayer(void* self);

private:
    void* self;
};
