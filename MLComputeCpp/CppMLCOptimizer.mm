#import "CppMLCOptimizer.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCOptimizer.h>

float CppMLCOptimizer::learningRate() {
    return ((MLCOptimizer*)self).learningRate;
}

void CppMLCOptimizer::learningRate(float lr) {
    ((MLCOptimizer*)self).learningRate = lr;
}

auto CppMLCOptimizer::gradientRescale() -> float {
    return ((MLCOptimizer*)self).gradientRescale;
}

bool CppMLCOptimizer::appliesGradientClipping() {
    return ((MLCOptimizer*)self).appliesGradientClipping == YES;
}

void CppMLCOptimizer::appliesGradientClipping(bool enable) {
    ((MLCOptimizer*)self).appliesGradientClipping = enable ? YES : NO;
}

auto CppMLCOptimizer::gradientClipMax() -> float {
    return ((MLCOptimizer*)self).gradientClipMax;
}

auto CppMLCOptimizer::gradientClipMin() -> float {
    return ((MLCOptimizer*)self).gradientClipMin;
}

auto CppMLCOptimizer::regularizationScale() -> float {
    return ((MLCOptimizer*)self).regularizationScale;
}

auto CppMLCOptimizer::regularizationType() -> eMLCRegularizationType {
    return MLCRegularizationTypeToCpp(((MLCOptimizer*)self).regularizationType);
}

CppMLCOptimizer::CppMLCOptimizer(void *self)
    : self{self}
{
}
