#include "CppMLCYOLOLossDescriptor.h"

#include "CppMLCTypesPrivate.h"

#import <MLCompute/MLCYOLOLossDescriptor.h>

auto CppMLCYOLOLossDescriptor::anchorBoxCount() -> uint32_t
{
    return (uint32_t)((MLCYOLOLossDescriptor*)self).anchorBoxCount;
}

auto CppMLCYOLOLossDescriptor::anchorBoxes() -> std::vector<float>
{
    return CppMLCTypesPrivate::NSDataToVectorFloat(((MLCYOLOLossDescriptor*)self).anchorBoxes);
}

bool CppMLCYOLOLossDescriptor::shouldRescore()
{
    return ((MLCYOLOLossDescriptor*)self).shouldRescore == YES;
}

void CppMLCYOLOLossDescriptor::shouldRescore(bool value)
{
    ((MLCYOLOLossDescriptor*)self).shouldRescore = value ? YES : NO;
}

auto CppMLCYOLOLossDescriptor::scaleSpatialPositionLoss() -> float
{
    return ((MLCYOLOLossDescriptor*)self).scaleSpatialPositionLoss;
}

void CppMLCYOLOLossDescriptor::scaleSpatialPositionLoss(float value)
{
    ((MLCYOLOLossDescriptor*)self).scaleSpatialPositionLoss = value;
}

auto CppMLCYOLOLossDescriptor::scaleSpatialSizeLoss() -> float
{
    return ((MLCYOLOLossDescriptor*)self).scaleSpatialSizeLoss;
}

void CppMLCYOLOLossDescriptor::scaleSpatialSizeLoss(float value)
{
    ((MLCYOLOLossDescriptor*)self).scaleSpatialSizeLoss = value;
}

auto CppMLCYOLOLossDescriptor::scaleNoObjectConfidenceLoss() -> float
{
    return ((MLCYOLOLossDescriptor*)self).scaleNoObjectConfidenceLoss;
}

void CppMLCYOLOLossDescriptor::scaleNoObjectConfidenceLoss(float value)
{
    ((MLCYOLOLossDescriptor*)self).scaleNoObjectConfidenceLoss = value;
}

auto CppMLCYOLOLossDescriptor::scaleObjectConfidenceLoss() -> float
{
    return ((MLCYOLOLossDescriptor*)self).scaleObjectConfidenceLoss;
}

void CppMLCYOLOLossDescriptor::scaleObjectConfidenceLoss(float value)
{
    ((MLCYOLOLossDescriptor*)self).scaleObjectConfidenceLoss = value;
}

auto CppMLCYOLOLossDescriptor::scaleClassLoss() -> float
{
    return ((MLCYOLOLossDescriptor*)self).scaleClassLoss;
}

void CppMLCYOLOLossDescriptor::scaleClassLoss(float value)
{
    ((MLCYOLOLossDescriptor*)self).scaleClassLoss = value;
}

auto CppMLCYOLOLossDescriptor::minimumIOUForObjectPresence() -> float
{
    return ((MLCYOLOLossDescriptor*)self).minimumIOUForObjectPresence;
}

void CppMLCYOLOLossDescriptor::minimumIOUForObjectPresence(float value)
{
    ((MLCYOLOLossDescriptor*)self).minimumIOUForObjectPresence = value;
}

auto CppMLCYOLOLossDescriptor::maximumIOUForObjectAbsence() -> float
{
    return ((MLCYOLOLossDescriptor*)self).maximumIOUForObjectAbsence;
}

void CppMLCYOLOLossDescriptor::maximumIOUForObjectAbsence(float value)
{
    ((MLCYOLOLossDescriptor*)self).maximumIOUForObjectAbsence = value;
}

auto CppMLCYOLOLossDescriptor::descriptorWithAnchorBoxes(std::vector<float> const& anchorBoxes,
                                                         uint32_t anchorBoxCount) -> CppMLCYOLOLossDescriptor
{
    return CppMLCYOLOLossDescriptor{[MLCYOLOLossDescriptor descriptorWithAnchorBoxes:CppMLCTypesPrivate::toNSData(anchorBoxes)
                                                                      anchorBoxCount:(NSUInteger)anchorBoxCount]};
}

CppMLCYOLOLossDescriptor::CppMLCYOLOLossDescriptor(void *self)
    : self{self}
{
}
