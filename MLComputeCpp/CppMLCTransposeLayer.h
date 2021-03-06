#pragma once

#include "CppMLCLayer.h"

#include <vector>

/*! @class      MLCTransposeLayer
    @abstract   A transpose layer
 */
class CppMLCTransposeLayer : public CppMLCLayer
{
public:
    /*! @property   dimensions
        @abstract   Permutes the dimensions according to 'dimensions'.
        @discussion The returned tensor's dimension i will correspond to dimensions[i].
     */
    auto dimensions() -> std::vector<uint32_t>;

    /*! @abstract   Create a transpose layer
        @param      dimensions NSArray<NSNumber *> representing the desired ordering of dimensions
                    The dimensions array specifies the input axis source for each output axis, such that the
                    K'th element in the dimensions array specifies the input axis source for the K'th axis in the
                    output.  The batch dimension which is typically axis 0 cannot be transposed.
        @return     A new transpose layer.
     */
    auto layerWithDimensions(std::vector<uint32_t> dimensions) -> CppMLCTransposeLayer;

private:
    CppMLCTransposeLayer(void* self);
};
