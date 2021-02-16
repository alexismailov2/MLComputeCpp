#pragma once

#include "CppMLCOptimizer.h"

#include <cstdint>

class CppMLCDevice;
class CppMLCOptimizerDescriptor;
class CppMLCAdamOptimizer;

/*! @class      CppMLCAdamOptimizer
    @discussion The MLCAdamOptimizer specifies the Adam optimizer.
 */
class CppMLCAdamOptimizer : public CppMLCOptimizer //<NSCopying>
{
public:
    /*! @property   beta1
        @abstract   Coefficent used for computing running averages of gradient.
        @discussion The default is 0.9.
     */
    auto beta1() -> float;

    /*! @property   beta2
        @abstract   Coefficent used for computing running averages of square of gradient.
        @discussion The default is 0.999.
     */
    auto beta2() -> float;

    /*! @property   epsilon
        @abstract   A term added to improve numerical stability.
        @discussion The default is 1e-8.
     */
    auto epsilon() -> float;

    /*! @property   timeStep
        @abstract   The current timestep used for the update.
        @discussion The default is 1.
     */
    auto timeStep() -> uint32_t;

    /*! @abstract   Create a MLCAdamOptimizer object with defaults
        @return     A new MLCAdamOptimizer object.
     */
    static CppMLCAdamOptimizer optimizerWithDescriptor(CppMLCOptimizerDescriptor const& optimizerDescriptor);

    /*! @abstract   Create a MLCAdamOptimizer object
        @param      optimizerDescriptor    The optimizer descriptor object
        @param      beta1                                   The beta1 value
        @param      beta2                                   The beta2 value
        @param      epsilon                              The epsilon value to use to improve numerical stability
        @param      timeStep                            The initial timestep to use for the update
        @return     A new MLCAdamOptimizer object.
     */
    static CppMLCAdamOptimizer optimizerWithDescriptor(CppMLCOptimizerDescriptor const& optimizerDescriptor,
                                                       float beta1,
                                                       float beta2,
                                                       float epsilon,
                                                       uint32_t timeStep);

private:
    CppMLCAdamOptimizer(void* self);

private:
    void* self;
    friend CppMLCAdamOptimizer;
};
