#pragma once

#import "CppMLCTypes.h"

class CppMLCAdamOptimizer;
class CppMLCSGDOptimizer;
class CppMLCRMSPropOptimizer;

/*! @class      MLCOptimizerDescriptor
    @discussion The MLCOptimizerDescriptor specifies an optimizer descriptor.
 */
class CppMLCOptimizerDescriptor //: NSObject <NSCopying>
{
public:
    /*! @property   learningRate
        @abstract   The learning rate.  This property is 'readwrite' so that callers can implement a 'decay' during training
     */
    auto learningRate() -> float;

    /*! @property   gradientRescale
        @abstract   The rescale value applied to gradients during optimizer update
     */
    auto gradientRescale() -> float;

    /*! @property   appliesGradientClipping
        @abstract   Whether gradient clipping should be applied or not.
        @discussion The default is false
     */
    bool appliesGradientClipping();

    /*! @property   gradientClipMax
        @abstract   The maximum gradient value if gradient clipping is enabled before gradient is rescaled.
     */
    auto gradientClipMax() -> float;

    /*! @property   gradientClipMin
        @abstract   The minimum gradient value if gradient clipping is enabled before gradient is rescaled.
     */
    auto gradientClipMin() -> float;

    /*! @property   regularizationScale
        @abstract   The regularization scale.
     */
    auto regularizationScale() -> float;

    /*! @property   regularizationType
        @abstract   The regularization type.
     */
    auto regularizationType() -> eMLCRegularizationType;

    /*! @abstract   Create a MLCOptimizerDescriptor object
        @param      learningRate                    The learning rate
        @param      gradientRescale              The gradient rescale value
        @param      regularizationType       The regularization type
        @param      regularizationScale     The regularization scale
        @return     A new MLCOptimizerDescriptor object.
     */
    static CppMLCOptimizerDescriptor descriptorWithLearningRate(float learningRate,
                                                                float gradientRescale,
                                                                eMLCRegularizationType regularizationType,
                                                                float regularizationScale);

    /*! @abstract   Create a MLCOptimizerDescriptor object
        @param      learningRate                      The learning rate
        @param      gradientRescale                The gradient rescale value
        @param      appliesGradientClipping   Whether to apply gradient clipping
        @param      gradientClipMax                The maximum gradient value to be used with gradient clipping
        @param      gradientClipMin                The minimum gradient value to be used with gradient clipping
        @param      regularizationType          The regularization type
        @param      regularizationScale        The regularization scale
        @return     A new MLCOptimizerDescriptor object.
     */
    static CppMLCOptimizerDescriptor descriptorWithLearningRate(float learningRate,
                                                                float gradientRescale,
                                                                bool appliesGradientClipping,
                                                                float gradientClipMax,
                                                                float gradientClipMin,
                                                                eMLCRegularizationType regularizationType,
                                                                float regularizationScale);
private:
    CppMLCOptimizerDescriptor(void* self);

private:
    void* self;
    friend CppMLCAdamOptimizer;
    friend CppMLCSGDOptimizer;
    friend CppMLCRMSPropOptimizer;
};

