#pragma once

#include "CppMLCTypes.h"

class CppMLCLossLayer;

/*! @class      CppMLCLossDescriptor
    @discussion The MLCLossDescriptor specifies a loss filter descriptor.
 */
class CppMLCLossDescriptor
{
public:
    /*! @property   lossType
        @abstract   Specifies the loss function.
     */
    auto lossType() -> eMLCLossType;

    /*! @property   reductionType
        @abstract   The reduction operation performed by the loss function.
     */
    auto  reductionType() -> eMLCReductionType;

    /*! @property   weight
        @abstract   The scale factor to apply to each element of a result.  The default value is 1.0.
     */
    auto weight() -> float;

    /*! @property    labelSmoothing
        @abstract    The label smoothing parameter. The default value is 0.0.
        @discussion  This parameter is valid only for the loss functions of the following type(s):
                         MLCLossTypeSoftmaxCrossEntropy and MLCLossTypeSigmoidCrossEntropy.
     */
    auto labelSmoothing() -> float;

    /*! @property    numberOfClasses
        @abstract    The number of classes parameter. The default value is 1.
        @discussion  This parameter is valid only for the loss function MLCLossTypeSoftmaxCrossEntropy.
     */
    auto classCount() -> uint32_t;

    /*! @property    epsilon
        @abstract    The epsilon parameter. The default value is 1e-7.
        @discussion  This parameter is valid only for the loss function MLCLossTypeLog.
     */
    auto epsilon() -> float;

    /*! @property    delta
        @abstract    The delta parameter. The default value is 1.0f.
        @discussion  This parameter is valid only for the loss function MLCLossTypeHuber.
     */
    auto delta() -> float;

    /*! @abstract   Create a loss descriptor object
        @param      lossType        The loss function.
        @param      reductionType   The reduction operation
        @return     A new MLCLossDescriptor object
     */
    static
    auto descriptorWithType(eMLCLossType lossType,
                            eMLCReductionType reductionType) -> CppMLCLossDescriptor;

    /*! @abstract   Create a loss descriptor object
        @param      lossType        The loss function.
        @param      reductionType   The reduction operation
        @param      weight          The scale factor to apply to each element of a result.
        @return     A new MLCLossDescriptor object
     */
    static
    auto descriptorWithType(eMLCLossType lossType,
                            eMLCReductionType reductionType,
                            float weight) -> CppMLCLossDescriptor;

    /*! @abstract   Create a loss descriptor object
        @param      lossType           The loss function.
        @param      reductionType         The reduction operation
        @param      weight             The scale factor to apply to each element of a result.
        @param      labelSmoothing     The label smoothing parameter.
        @param      classCount         The number of classes parameter.
        @return     A new MLCLossDescriptor object
     */
    static
    auto descriptorWithType(eMLCLossType lossType,
                            eMLCReductionType reductionType,
                            float weight,
                            float labelSmoothing,
                            uint32_t classCount) -> CppMLCLossDescriptor;

    /*! @abstract   Create a loss descriptor object
     @param      lossType            The loss function.
     @param      reductionType          The reduction operation
     @param      weight              The scale factor to apply to each element of a result.
     @param      labelSmoothing      The label smoothing parameter.
     @param      classCount          The number of classes parameter.
     @param      epsilon             The epsilon used by LogLoss
     @param      delta               The delta parameter used by Huber loss
     @return     A new MLCLossDescriptor object
     */
    static
    auto descriptorWithType(eMLCLossType lossType,
                            eMLCReductionType reductionType,
                            float weight,
                            float labelSmoothing,
                            uint32_t classCount,
                            float epsilon,
                            float delta) -> CppMLCLossDescriptor;
private:
    CppMLCLossDescriptor(void* self);

private:
    void* self;
    friend CppMLCLossLayer;
};
