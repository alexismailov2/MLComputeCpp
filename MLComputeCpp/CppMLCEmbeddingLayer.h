#pragma once

#include "CppMLCLayer.h"
#include "CppMLCEmbeddingDescriptor.h"
#include "CppMLCTensorParameter.h"

/*! @class      MLCEmbeddingLayer
 *  @abstract   An embedding layer which stores the words embedding
 *  @discussion For details refer to: https://pytorch.org/docs/stable/nn.html#embedding
 *              Only supported on CPU and can only be used as the first layer in a graph. If needs to be used with another graph compiled for a GPU device,
 *              a second graph containing the embedding layer can be created first. The result of this layer can then be fed as an input to the second graph
 *              and respectively the gradient result of the first layer of the second graph can be passed to this graph for weight update.
 */
class CppMLCEmbeddingLayer : public CppMLCLayer
{
public:
    auto descriptor() -> CppMLCEmbeddingDescriptor;

    /*! @property   weights
     *  @abstract   The array of word embeddings
     */
    auto weights() -> CppMLCTensor;

    /*! @property   weightsParameter
        @abstract   The weights tensor parameter used for optimizer update
     */
    auto weightsParameter() -> CppMLCTensorParameter;

    static auto layerWithDescriptor(CppMLCEmbeddingDescriptor const& descriptor,
                                    CppMLCTensor const& weights) -> CppMLCEmbeddingLayer;

private:
    CppMLCEmbeddingLayer(void* self);

private:
    void* self;
};

