#pragma once

#include "CppMLCLayer.h"

#include <vector>

/*! @class      MLCSplitLayer
    @abstract   A split layer
 */
class CppMLCSplitLayer : public CppMLCLayer
{
private:
    /*! @property   dimension
        @abstract   The dimension (or axis) along which to split tensor
     */
    auto dimension() -> uint32_t;

    /*! @property   splitCount
        @abstract   The number of splits.
        @discussion The tensor will be split into equally sized chunks.  The last chunk may be smaller in size.
     */
    auto splitCount() -> uint32_t;

    /*! @property   splitSectionLengths
        @abstract   Lengths of each split section.
        @discussion The tensor will be split into chunks along dimensions with sizes given in \p splitSectionLengths .
     */
    auto splitSectionLengths() -> std::vector<uint32_t>;

    /*! @abstract   Create a split layer
        @param      splitCount  The number of splits.
        @param      dimension   The dimension along which the tensor should be split.
        @return     A new split layer
    */
    auto layerWithSplitCount(uint32_t splitCount, uint32_t dimension) -> CppMLCSplitLayer;

    /*! @abstract   Create a split layer
        @param      splitSectionLengths   Lengths of each split section.
        @param      dimension             The dimension along which the tensor should be split.
        @return     A new split layer
    */
    auto layerWithSplitSectionLengths(std::vector<uint32_t> const& splitSectionLengths,
                                      uint32_t dimension) -> CppMLCSplitLayer;

private:
    CppMLCSplitLayer(void* self);
};
