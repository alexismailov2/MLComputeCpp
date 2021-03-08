#pragma once

#import "CppMLCTypes.h"

class CppMLCTrainingGraph;

/*! @class      MLCOptimizer
    @discussion The MLCOptimizer specifies a base optimizer.
 */
class CppMLCOptimizer
{
public:
    /*! @property   learningRate
        @abstract   The learning rate.  This property is 'readwrite' so that callers can implement a 'decay' during training
     */
    auto learningRate() -> float;
    void learningRate(float lr);

    /*! @property   gradientRescale
        @abstract   The rescale value applied to gradients during optimizer update
     */
    auto gradientRescale() -> float;

    /*! @property   appliesGradientClipping
        @abstract   Whether gradient clipping should be applied or not.
     */
    bool appliesGradientClipping();
    void appliesGradientClipping(bool enable);

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

protected:
    CppMLCOptimizer(void* self);

protected:
    void* self;
    friend CppMLCTrainingGraph;
};
