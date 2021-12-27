#pragma once

#include "CppMLCLayer.h"
#include "CppMLCConvolutionDescriptor.h"
#include "CppMLCTensorParameter.h"

/*! @class      MLCFullyConnectedLayer
    @abstract   A fully connected layer a.k.a a dense layer
    @discussion For C:input feature channel, C':output feature channel, the layer maps (*,C) --> (*,C') where * can be 1, 2 or 3 dimesnion.
                There is an exception for the case of (N,C,1,1) which gets mapped to (N,C',1,1).
 */
class CppMLCFullyConnectedLayer : public CppMLCLayer
{
public:
    /*! @property   descriptor
        @abstract   The convolution descriptor
     */
    auto descriptor() -> CppMLCConvolutionDescriptor;

    /*! @property   weights
        @abstract   The weights tensor used by the convolution layer
     */
    auto weights() -> CppMLCTensor;

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


    /*! @abstract   Create a fully connected layer
        @param      weights        The weights tensor
        @param      biases         The bias tensor
        @param      descriptor     The convolution descriptor
        @return     A new fully connected layer
     */
    static auto layerWithWeights(CppMLCTensor const& weights,
                                 CppMLCTensor const& biases,
                                 CppMLCConvolutionDescriptor const& descriptor) -> CppMLCFullyConnectedLayer;

    ~CppMLCFullyConnectedLayer();
private:
    CppMLCFullyConnectedLayer(void* self);

private:
    //void* self;
};
