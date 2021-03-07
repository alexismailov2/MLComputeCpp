#pragma once

#include "CppMLCOptimizer.h"

class CppMLCOptimizerDescriptor;

/*! @class      CppMLCRMSPropOptimizer
    @discussion The MLCRMSPropOptimizer specifies the RMSProp optimizer.
 */
class CppMLCRMSPropOptimizer : public CppMLCOptimizer
{
private:
    /*! @property   momentumScale
        @abstract   The momentum factor.  A hyper-parameter.
        @discussion The default is 0.0.
     */
    auto momentumScale() -> float;

    /*! @property   alpha
        @abstract   The smoothing constant.
        @discussion The default is 0.99.
     */
    auto alpha() -> float;

    /*! @property   epsilon
        @abstract   A term added to improve numerical stability.
        @discussion The default is 1e-8.
     */
    auto epsilon() -> float;

    /*! @property   isCentered
        @abstract   If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
        @discussion The default is false.
     */
    bool isCentered();

    /*! @abstract   Create a MLCRMSPropOptimizer object with defaults
        @return     A new MLCRMSPropOptimizer object.
     */
    auto optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor) -> CppMLCRMSPropOptimizer;

    /*! @abstract   Create a MLCRMSPropOptimizer object
        @param      optimizerDescriptor    The optimizer descriptor object
        @param      momentumScale                 The momentum scale
        @param      alpha                                   The smoothing constant value
        @param      epsilon                              The epsilon value to use to improve numerical stability
        @param      isCentered                            A boolean to specify whether to compute the centered RMSProp or not
        @return     A new MLCRMSPropOptimizer object.
    */
    auto optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor,
                                 float momentumScale,
                                 float alpha,
                                 float epsilon,
                                 bool isCentered) -> CppMLCRMSPropOptimizer;

private:
    CppMLCRMSPropOptimizer(void* self);
};
