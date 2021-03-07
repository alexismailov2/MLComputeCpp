#pragma once

#include "CppMLCLayer.h"

#include <vector>

/*! @abstract   Slice layer is used to slice a given source.
 *  @discussion Slicing should not decrease the tensor dimension.
 *              The start, end and stride vectors must have the same number of dimension as the source tensor.
 *              Only positive stride is supported.
 */
class CppMLCSliceLayer : public CppMLCLayer
{
public:
    /*! @property   start
        @abstract   A vector of length equal to that of source. The element at index i specifies the beginning of slice in dimension i.
     */
    auto start() -> std::vector<uint32_t>;

    /*! @property   end
        @abstract   A vector of length equal to that of source. The element at index i specifies the end of slice in dimension i.
     */
    auto end() -> std::vector<uint32_t>;

    /*! @property   stride
        @abstract   A vector of length equal to that of source. The element at index i specifies the stride of slice in dimension i.
     */
    auto stride() -> std::vector<uint32_t>;

    /*!
     @abstract Create a slice layer
     @param    stride If set to nil, it will be set to 1.
     @return   A new layer for slicing tensors.
     */
    auto sliceLayerWithStart(std::vector<uint32_t> start,
                             std::vector<uint32_t> end,
                             std::vector<uint32_t> stride) -> CppMLCSliceLayer;
private:
    CppMLCSliceLayer(void* self);
};
