#import "CppMLCAdamOptimizer.h"
#import "CppMLCOptimizerDescriptor.h"

#import <MLCompute/MLCAdamOptimizer.h>

auto CppMLCAdamOptimizer::beta1() -> float {
    return ((MLCAdamOptimizer*)self).beta1;
}

auto CppMLCAdamOptimizer::beta2() -> float {
    return ((MLCAdamOptimizer*)self).beta2;
}

auto CppMLCAdamOptimizer::epsilon() -> float {
    return ((MLCAdamOptimizer*)self).epsilon;
}

auto CppMLCAdamOptimizer::timeStep() -> uint32_t {
    return (uint32_t)((MLCAdamOptimizer*)self).timeStep;
}

CppMLCAdamOptimizer CppMLCAdamOptimizer::optimizerWithDescriptor(CppMLCOptimizerDescriptor const& optimizerDescriptor) {
    return CppMLCAdamOptimizer{[MLCAdamOptimizer optimizerWithDescriptor:(MLCOptimizerDescriptor*)optimizerDescriptor.self]};
}

CppMLCAdamOptimizer
CppMLCAdamOptimizer::optimizerWithDescriptor(CppMLCOptimizerDescriptor const& optimizerDescriptor,
                                             float beta1,
                                             float beta2,
                                             float epsilon,
                                             uint32_t timeStep) {
    return CppMLCAdamOptimizer{[MLCAdamOptimizer optimizerWithDescriptor:(MLCOptimizerDescriptor*)optimizerDescriptor.self
                                                                   beta1:beta1
                                                                   beta2:beta2
                                                                 epsilon:epsilon
                                                                timeStep:(NSUInteger)timeStep]};

}

CppMLCAdamOptimizer::CppMLCAdamOptimizer(void *self)
    : CppMLCOptimizer{self}
    , self{self}
{
}
