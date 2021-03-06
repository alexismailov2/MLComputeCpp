#pragma once

#include "CppMLCOptimizer.h"

class CppMLCOptimizerDescriptor;

/*! @class      CppMLCSGDOptimizer
    @discussion The MLCSGDOptimizer specifies a stochastic gradient descent optimizer.
 */
class CppMLCSGDOptimizer : public CppMLCOptimizer
{
public:
    /*! @property   momentumScale
        @abstract   The momentum factor.  A hyper-parameter.
        @discussion The default is 0.0.
     */
    auto momentumScale() -> float;

    /*! @property   usesNesterovMomentum
        @abstract   A boolean that specifies whether to apply nesterov momentum or not.
        @discussion The default is false.
     */
    auto usesNesterovMomentum() -> bool;

    /*! @abstract   Create an MLCSGDOptimizer object with defaults
        @return     A new MLCSGDOptimizer object.
     */
    auto optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor) -> CppMLCSGDOptimizer;

    /*! @abstract   Create an MLCSGDOptimizer object
        @param      optimizerDescriptor    The optimizer descriptor object
        @param      momentumScale                 The momentum scale
        @param      usesNesterovMomentum      A boolean to enable / disable nesterov momentum
        @return     A new MLCSGDOptimizer object.
     */
    auto optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor,
                                 float momentumScale,
                                 bool usesNesterovMomentum) -> CppMLCSGDOptimizer;

private:
    CppMLCSGDOptimizer(void* self);
};