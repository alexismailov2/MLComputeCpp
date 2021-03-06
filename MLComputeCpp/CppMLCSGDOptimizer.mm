#include "CppMLCSGDOptimizer.h"

#include "CppMLCOptimizerDescriptor.h"

#import <MLCompute/MLCSGDOptimizer.h>

auto CppMLCSGDOptimizer::momentumScale() -> float
{
    return ((MLCSGDOptimizer*)self).momentumScale;
}

auto CppMLCSGDOptimizer::usesNesterovMomentum() -> bool
{
    return ((MLCSGDOptimizer*)self).usesNesterovMomentum == YES;
}

auto CppMLCSGDOptimizer::optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor) -> CppMLCSGDOptimizer
{
    return CppMLCSGDOptimizer{[MLCSGDOptimizer optimizerWithDescriptor:(MLCOptimizerDescriptor*)optimizerDescriptor.self]};
}

auto CppMLCSGDOptimizer::optimizerWithDescriptor(CppMLCOptimizerDescriptor& optimizerDescriptor,
                                                 float momentumScale,
                                                 bool usesNesterovMomentum) -> CppMLCSGDOptimizer
{
    return CppMLCSGDOptimizer{[MLCSGDOptimizer optimizerWithDescriptor:(MLCOptimizerDescriptor*)optimizerDescriptor.self
                                                         momentumScale:momentumScale
                                                  usesNesterovMomentum:usesNesterovMomentum ? YES : NO]};
}

CppMLCSGDOptimizer::CppMLCSGDOptimizer(void* self)
    : CppMLCOptimizer(self)
{
}
