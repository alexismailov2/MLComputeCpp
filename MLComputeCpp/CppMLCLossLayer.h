#pragma once

#include "CppMLCLayer.h"
#include "CppMLCLossDescriptor.h"

/*! @class      CppMLCLossLayer
    @abstract   A loss layer
 */
class CppMLCLossLayer : public CppMLCLayer
{
public:
    /*! @property   descriptor
        @abstract   The loss descriptor
     */
    auto descriptor() -> CppMLCLossDescriptor;

    /*! @property   weights
        @abstract   The loss label weights tensor
     */
    auto weights() -> CppMLCTensor;

    /*! @abstract   Create a loss layer
        @param      lossDescriptor          The loss descriptor
        @return     A new loss layer.
     */
    static
    auto layerWithDescriptor(CppMLCLossDescriptor const& lossDescriptor) -> CppMLCLossLayer;

    /*! @abstract   Create a MLComputeLoss layer
        @param      lossDescriptor          The loss descriptor
        @param      weights                          The loss label weights tensor
        @return     A new loss layer.
     */
    static
    auto layerWithDescriptor(CppMLCLossDescriptor& lossDescriptor, CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType      The reduction type to use
        @param      labelSmoothing     Label smoothing value
        @param      classCount             Number of classes
        @param      weight                      A scalar floating point value
        @return     A new softmax cross entropy loss layer.
     */
    static
    auto softmaxCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                  float labelSmoothing,
                                                  uint32_t classCount,
                                                  float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      labelSmoothing        Label smoothing value
        @param      classCount                 Number of classes
        @param      weights                        The loss label weights tensor
        @return     A new softmax cross entropy loss layer.
     */
    static
    auto softmaxCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                  float labelSmoothing,
                                                  uint32_t classCount,
                                                  CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType      The reduction type to use
        @param      labelSmoothing     Label smoothing value
        @param      classCount              Number of classes
        @param      weight                       A scalar floating point value
        @return     A new categorical cross entropy loss layer.
     */
    static
    auto categoricalCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                      float labelSmoothing,
                                                      uint32_t classCount,
                                                      float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      labelSmoothing        Label smoothing value
        @param      classCount                 Number of classes
        @param      weights                        The loss label weights tensor
        @return     A new categorical cross entropy loss layer.
     */
    static
    auto categoricalCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                      float labelSmoothing,
                                                      uint32_t classCount,
                                                      CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      labelSmoothing       Label smoothing value
        @param      weight                         A scalar floating-point value
        @return     A new sigmoid cross entropy loss layer.
     */
    static
    auto sigmoidCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                  float labelSmoothing,
                                                  float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      labelSmoothing       Label smoothing value
        @param      weights                       The loss label weights tensor
        @return     A new sigmoid cross entropy loss layer.
     */
    static
    auto sigmoidCrossEntropyLossWithReductionType(eMLCReductionType reductionType,
                                                  float labelSmoothing,
                                                  CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      epsilon                       The epsilon parameter
        @param      weight                         A scalar floating-point value
        @return     A new log loss layer.
     */
    static
    auto logLossWithReductionType(eMLCReductionType reductionType,
                                  float epsilon,
                                  float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      epsilon                       The epsilon parameter
        @param      weights                       The loss label weights tensor
        @return     A new log loss layer.
     */
    static
    auto logLossWithReductionType(eMLCReductionType reductionType,
                                  float epsilon,
                                  CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      delta                           The delta parameter
        @param      weight                         A scalar floating-point value
        @return     A new huber loss layer.
     */
    static
    auto huberLossWithReductionType(eMLCReductionType reductionType,
                                    float delta,
                                    float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      delta                            The delta parameter
        @param      weights                        The loss label weights tensor
        @return     A new huber loss layer.
     */
    static
    auto huberLossWithReductionType(eMLCReductionType reductionType,
                                    float delta,
                                    CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      weight                         A scalar floating-point value
        @return     A new L1 i.e. mean absolute error loss layer.
     */
    static
    auto meanAbsoluteErrorLossWithReductionType(eMLCReductionType reductionType,
                                                float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      weights                       The loss label weights tensor
        @return     A new L1 i.e. mean absolute error loss layer.
     */
    static
    auto meanAbsoluteErrorLossWithReductionType(eMLCReductionType reductionType,
                                                CppMLCTensor& weights) -> CppMLCLossLayer;


    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      weight                         A scalar floating-point value
        @return     A new L2 i.e. mean squared error loss layer.
     */
    static
    auto meanSquaredErrorLossWithReductionType(eMLCReductionType reductionType,
                                               float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      weights                       The loss label weights tensor
        @return     A new L2 i.e. mean squared error loss layer.
     */
    static
    auto meanSquaredErrorLossWithReductionType(eMLCReductionType reductionType,
                                               CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType         The reduction type to use
        @param      weight                         A scalar floating-point value
        @return     A new hinge loss layer.
     */
    static
    auto hingeLossWithReductionType(eMLCReductionType reductionType,
                                    float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      weights                       The loss label weights tensor
        @return     A new hinge loss layer.
     */
    static
    auto hingeLossWithReductionType(eMLCReductionType reductionType,
                                    CppMLCTensor& weights) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      weight                         A scalar floating-point value
        @return     A new cosine distance loss layer.
     */
    static
    auto cosineDistanceLossWithReductionType(eMLCReductionType reductionType,
                                             float weight) -> CppMLCLossLayer;

    /*! @abstract   Create a loss layer
        @param      reductionType          The reduction type to use
        @param      weights                       The loss label weights tensor
        @return     A new cosine distance loss layer.
     */
    static
    auto cosineDistanceLossWithReductionType(eMLCReductionType reductionType,
                                             CppMLCTensor& weights) -> CppMLCLossLayer;

protected:
    CppMLCLossLayer(void* self);
};

