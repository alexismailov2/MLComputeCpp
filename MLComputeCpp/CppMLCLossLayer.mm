#include "CppMLCLossLayer.h"

#include "CppMLCTensor.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCLossLayer.h>

auto CppMLCLossLayer::descriptor() -> CppMLCLossDescriptor
{
    return CppMLCLossDescriptor{((MLCLossLayer*)self).descriptor};
}

auto CppMLCLossLayer::weights() -> CppMLCTensor
{
    return CppMLCTensor{((MLCLossLayer*)self).weights};
}

auto CppMLCLossLayer::layerWithDescriptor(CppMLCLossDescriptor& lossDescriptor) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer layerWithDescriptor:(MLCLossDescriptor*)lossDescriptor.self]};
}

auto CppMLCLossLayer::layerWithDescriptor(CppMLCLossDescriptor& lossDescriptor,
                                          CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer layerWithDescriptor:(MLCLossDescriptor*)lossDescriptor.self
                                                     weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::softmaxCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                               float labelSmoothing,
                                                               uint32_t classCount,
                                                               float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer softmaxCrossEntropyLossWithReductionType:toNative(reductionType)
                                                                   labelSmoothing:labelSmoothing
                                                                       classCount:(NSUInteger)classCount
                                                                           weight:weight]};
}

auto CppMLCLossLayer::softmaxCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                               float labelSmoothing,
                                                               uint32_t classCount,
                                                               CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer softmaxCrossEntropyLossWithReductionType:toNative(reductionType)
                                                                   labelSmoothing:labelSmoothing
                                                                       classCount:(NSUInteger)classCount
                                                                           weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::categoricalCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                                   float labelSmoothing,
                                                                   uint32_t classCount,
                                                                   float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer categoricalCrossEntropyLossWithReductionType:toNative(reductionType)
                                                                       labelSmoothing:labelSmoothing
                                                                           classCount:(NSUInteger)classCount
                                                                           weight:weight]};
}

auto CppMLCLossLayer::categoricalCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                                   float labelSmoothing,
                                                                   uint32_t classCount,
                                                                   CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer categoricalCrossEntropyLossWithReductionType:toNative(reductionType)
                                                                       labelSmoothing:labelSmoothing
                                                                           classCount:(NSUInteger)classCount
                                                                              weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::sigmoidCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                               float labelSmoothing,
                                                               float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer sigmoidCrossEntropyLossWithReductionType:toNative(reductionType)
                                                                   labelSmoothing:labelSmoothing
                                                                           weight:weight]};
}

auto CppMLCLossLayer::sigmoidCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                               float labelSmoothing,
                                                               CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer sigmoidCrossEntropyLossWithReductionType:toNative(reductionType)
                                                                   labelSmoothing:labelSmoothing
                                                                          weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::logLossWithReductionType(eMLCReductionType reductionType,
                                               float epsilon,
                                               float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer logLossWithReductionType:toNative(reductionType)
                                                          epsilon:epsilon
                                                           weight:weight]};
}

auto CppMLCLossLayer::logLossWithReductionType(eMLCReductionType reductionType,
                                               float epsilon,
                                               CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer logLossWithReductionType:toNative(reductionType)
                                                          epsilon:epsilon
                                                          weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::huberLossWithReductionType(eMLCReductionType reductionType,
                                                 float delta,
                                                 float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer huberLossWithReductionType:toNative(reductionType)
                                                              delta:delta
                                                             weight:weight]};
}

auto CppMLCLossLayer::huberLossWithReductionType(eMLCReductionType reductionType,
                                                 float delta,
                                                 CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer huberLossWithReductionType:toNative(reductionType)
                                                              delta:delta
                                                            weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::meanAbsoluteErrorLossWithReductionType(eMLCReductionType reductionType,
                                                             float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer meanAbsoluteErrorLossWithReductionType:toNative(reductionType)
                                                                         weight:weight]};
}

auto CppMLCLossLayer::meanAbsoluteErrorLossWithReductionType(eMLCReductionType reductionType,
                                                             CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer meanAbsoluteErrorLossWithReductionType:toNative(reductionType)
                                                                        weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::meanSquaredErrorLossWithReductionType(eMLCReductionType reductionType,
                                                            float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer meanSquaredErrorLossWithReductionType:toNative(reductionType)
                                                                        weight:weight]};
}

auto CppMLCLossLayer::meanSquaredErrorLossWithReductionType(eMLCReductionType reductionType,
                                                            CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer meanSquaredErrorLossWithReductionType:toNative(reductionType)
                                                                       weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::hingeLossWithReductionType(eMLCReductionType reductionType,
                                                 float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer hingeLossWithReductionType:toNative(reductionType)
                                                             weight:weight]};
}

auto CppMLCLossLayer::hingeLossWithReductionType(eMLCReductionType reductionType,
                                                 CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer hingeLossWithReductionType:toNative(reductionType)
                                                            weights:(MLCTensor*)weights.self]};
}

auto CppMLCLossLayer::cosineDistanceLossWithReductionType(eMLCReductionType reductionType,
                                                          float weight) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer cosineDistanceLossWithReductionType:toNative(reductionType)
                                                                      weight:weight]};
}

auto CppMLCLossLayer::cosineDistanceLossWithReductionType(eMLCReductionType reductionType,
                                                          CppMLCTensor& weights) -> CppMLCLossLayer
{
    return CppMLCLossLayer{[MLCLossLayer cosineDistanceLossWithReductionType:toNative(reductionType)
                                                                     weights:(MLCTensor*)weights.self]};
}

CppMLCLossLayer::CppMLCLossLayer(void *self)
    : CppMLCLayer(self)
{
}
