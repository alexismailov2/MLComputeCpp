#pragma once

#include "CppMLCLayer.h"

/*! @class      MLCReductionLayer
    @abstract   Reduce tensor values across a given dimension to a scalar value.
    @discussion The layer is used to perform reductionType operation on a given dimension.
                Result of this layer is a tensor of the same shape as source tensor,
                except for the given dimension which is set to 1.
 */
class CppMLCReductionLayer : public CppMLCLayer
{
public:
    /*! @property   reductionType
        @abstract   The reduction type
     */
    auto  reductionType() -> eMLCReductionType;

    /*! @property   dimension
        @abstract   The dimension over which to perform the reduction operation
     */
    auto dimension() -> uint32_t;

    /*! @abstract Create a reduction layer .
        @param    reductionType        The reduction type.
        @param    dimension          The reduction dimension.
        @return   A new reduction layer.
     */
    auto layerWithReductionType(eMLCReductionType reductionType,
                                uint32_t dimension) -> CppMLCReductionLayer;

private:
    CppMLCReductionLayer(void* self);
};
