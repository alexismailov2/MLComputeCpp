#include "CppMLCRMSPropOptimizer.h"

#include "CppMLCOptimizerDescriptor.h"

#import <MLCompute/MLCRMSPropOptimizer.h>

auto CppMLCRMSPropOptimizer::momentumScale() -> float
{
    return ((MLCRMSPropOptimizer*)self).momentumScale;
}

auto CppMLCRMSPropOptimizer::alpha() -> float
{
    return ((MLCRMSPropOptimizer*)self).alpha;
}

auto CppMLCRMSPropOptimizer::epsilon() -> float
{
    return ((MLCRMSPropOptimizer*)self).epsilon;
}

bool CppMLCRMSPropOptimizer::isCentered()
{
    return ((MLCRMSPropOptimizer*)self).isCentered == YES;
}

auto CppMLCRMSPropOptimizer::optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor) -> CppMLCRMSPropOptimizer
{
    return CppMLCRMSPropOptimizer{[MLCRMSPropOptimizer optimizerWithDescriptor:(MLCOptimizerDescriptor*)optimizerDescriptor.self]};
}

auto CppMLCRMSPropOptimizer::optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor,
                                                     float momentumScale,
                                                     float alpha,
                                                     float epsilon,
                                                     bool isCentered) -> CppMLCRMSPropOptimizer
{
    return CppMLCRMSPropOptimizer{[MLCRMSPropOptimizer optimizerWithDescriptor:(MLCOptimizerDescriptor*)optimizerDescriptor.self
                                                                 momentumScale:momentumScale
                                                                         alpha:alpha
                                                                       epsilon:epsilon
                                                                    isCentered:isCentered ? YES : NO]};
}

CppMLCRMSPropOptimizer::CppMLCRMSPropOptimizer(void *self)
    : CppMLCOptimizer(self)
{
}
