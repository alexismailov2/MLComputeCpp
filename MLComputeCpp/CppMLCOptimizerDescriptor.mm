#include "CppMLCOptimizerDescriptor.h"

#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCOptimizerDescriptor.h>

auto CppMLCOptimizerDescriptor::learningRate() -> float {
    return ((MLCOptimizerDescriptor*)self).learningRate;
}

auto CppMLCOptimizerDescriptor::gradientRescale() -> float {
    return ((MLCOptimizerDescriptor*)self).gradientRescale;
}

bool CppMLCOptimizerDescriptor::appliesGradientClipping() {
    return ((MLCOptimizerDescriptor*)self).appliesGradientClipping == YES;
}

auto CppMLCOptimizerDescriptor::gradientClipMax() -> float {
    return ((MLCOptimizerDescriptor*)self).gradientClipMax;
}

auto CppMLCOptimizerDescriptor::gradientClipMin() -> float {
    return ((MLCOptimizerDescriptor*)self).gradientClipMin;
}

auto CppMLCOptimizerDescriptor::regularizationScale() -> float {
    return ((MLCOptimizerDescriptor*)self).regularizationScale;
}

auto CppMLCOptimizerDescriptor::regularizationType() -> eMLCRegularizationType {
    return MLCRegularizationTypeToCpp(((MLCOptimizerDescriptor*)self).regularizationType);
}

CppMLCOptimizerDescriptor
CppMLCOptimizerDescriptor::descriptorWithLearningRate(float learningRate,
                                                      float gradientRescale,
                                                      eMLCRegularizationType regularizationType,
                                                      float regularizationScale)
{
    return CppMLCOptimizerDescriptor{[MLCOptimizerDescriptor descriptorWithLearningRate:learningRate
                                                                        gradientRescale:gradientRescale
                                                                     regularizationType:toNative(regularizationType)
                                                                    regularizationScale:regularizationScale]};
}

CppMLCOptimizerDescriptor
CppMLCOptimizerDescriptor::descriptorWithLearningRate(float learningRate,
                                                      float gradientRescale,
                                                      bool appliesGradientClipping,
                                                      float gradientClipMax,
                                                      float gradientClipMin,
                                                      eMLCRegularizationType regularizationType,
                                                      float regularizationScale)
{
    return CppMLCOptimizerDescriptor{[MLCOptimizerDescriptor descriptorWithLearningRate:learningRate
                                                                        gradientRescale:gradientRescale
                                                                appliesGradientClipping:(appliesGradientClipping ? YES : NO)
                                                                        gradientClipMax:gradientClipMax
                                                                        gradientClipMin:gradientClipMin
                                                                     regularizationType:toNative(regularizationType)
                                                                    regularizationScale:regularizationScale]};
}

CppMLCOptimizerDescriptor::CppMLCOptimizerDescriptor(void *self)
    : self{self}
{
}
