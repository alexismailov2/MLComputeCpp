#pragma once

#include "CppMLCLayer.h"

@class MLCDevice;
@class MLCTensor;

/*! @class      MLCConcatenationLayer
    @abstract   A concatenation layer
 */
class CppMLCConcatenationLayer : public CppMLCLayer
{
public:
    /*! @property   dimension
        @abstract   The dimension (or axis) along which to concatenate tensors
        @discussion The default value is 1 (which typically represents features channels)
     */
    auto dimension() -> uint32_t;

    /*! @abstract   Create a concatenation layer
        @return     A new concatenation layer
     */
    static CppMLCConcatenationLayer layer();

    /*! @abstract   Create a concatenation layer
        @param      dimension  The concatenation dimension
        @return     A new concatenation layer
     */
    static CppMLCConcatenationLayer layerWithDimension(uint32_t dimension);

private:
    CppMLCConcatenationLayer(void* self);

private:
    void* self;
};
