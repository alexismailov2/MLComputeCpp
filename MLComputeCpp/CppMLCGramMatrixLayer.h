#pragma once

#include "CppMLCLayer.h"

/*! @class       MLCGramMatrixLayer
    @abstract    A gram matrix layer
    @discussion  The MLComputeGramMatrix  specifies a layer which computes the uncentered cross-correlation
                 values between the spatial planes of each feature channel of a tensor. If the input tensor batch is
                 x = x[b, y, x, c], where 'b' is batch index, 'y' and 'x' are the spatial coordinates and 'c' is the feature channel
                 index then this layer computes the values:

                       y = y[b, 1, f, c] = alpha * sum_{x,y} x[b,y,x,f] * x[b,y,x,c], where 'alpha' is a scaling factor.

                 This operation can be interpreted to be computing all combinations of fully connected layers
                 between the different spatial planes of the input tensor. The results are stored in the feature channel and
                 'x'-coordinate indices of the output batch.

                 The operation is performed independently for each tensor in a batch.
 */
class CppMLCGramMatrixLayer : public CppMLCLayer
{
public:
    /*! @property   scale
        @abstract   The scale factor
     */
    auto scale() -> float;

    /*! @abstract   Create a GramMatrix layer
        @param      scale  The scaling factor for the output.
        @return     A new GramMatrix layer
     */
    static auto layerWithScale(float scale) -> CppMLCGramMatrixLayer;

public:
    CppMLCGramMatrixLayer(void* self);

public:
    void* self;
};