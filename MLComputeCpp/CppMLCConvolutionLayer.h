#pragma once

#include "CppMLCLayer.h"

class CppMLCConvolutionDescriptor;
class CppMLCTensor;
class CppMLCTensorParameter;

/*! @class      CppMLCConvolutionLayer
    @abstract   A convolution layer
 */
class CppMLCConvolutionLayer : public CppMLCLayer
{
public:
/*! @property   descriptor
    @abstract   The convolution descriptor
 */
auto descriptor() -> CppMLCConvolutionDescriptor;;

/*! @property   weights
    @abstract   The weights tensor used by the convolution layer
 */
auto weights() -> CppMLCTensor;;

/*! @property   biases
    @abstract   The bias tensor used by the convolution layer
 */
auto biases() -> CppMLCTensor;

/*! @property   weightsParameter
    @abstract   The weights tensor parameter used for optimizer update
 */
auto weightsParameter() -> CppMLCTensorParameter;

/*! @property   biasesParameter
    @abstract   The bias tensor parameter used for optimizer update
 */
auto biasesParameter() -> CppMLCTensorParameter;

/*! @abstract   Create a convolution layer
    @param      weights        The weights tensor
    @param      biases         The bias tensor
    @param      descriptor     The convolution descriptor
    @return     A new convolution layer.
 */
static auto layerWithWeights(CppMLCTensor const& weights,
                             CppMLCTensor const& biases,
                             CppMLCConvolutionDescriptor const& descriptor) -> CppMLCConvolutionLayer;
private:
    CppMLCConvolutionLayer(void* self);

private:
    void* self;
};
