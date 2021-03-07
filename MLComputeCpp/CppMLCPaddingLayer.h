#pragma once

#include "CppMLCLayer.h"

/*! @class      MLCPaddingLayer
    @abstract   A padding layer
 */
class CppMLCPaddingLayer : public CppMLCLayer
{
public:
    /*! @property   paddingType
        @abstract   The padding type i.e. constant, zero, reflect or symmetric
     */
    auto paddingType() -> eMLCPaddingType;

    /*! @property   paddingLeft
        @abstract   The left padding size
     */
    auto paddingLeft() -> uint32_t;

    /*! @property   paddingRight
        @abstract   The right padding size
     */
    auto paddingRight() -> uint32_t;

    /*! @property   paddingTop
        @abstract   The top padding size
     */
    auto paddingTop() -> uint32_t;

    /*! @property   paddingBottom
        @abstract   The bottom padding size
     */
    auto paddingBottom() -> uint32_t;

    /*! @property   constantValue
        @abstract   The constant value to use if padding type is constant.
     */
    auto constantValue() -> float;

    /*! @abstract   Create a padding layer with reflection padding
        @param      padding  The padding sizes.
        @return     A new padding layer
     */
    auto layerWithReflectionPadding(std::vector<uint32_t> const& padding) -> CppMLCPaddingLayer;

    /*! @abstract   Create a padding layer with symmetric padding
        @param      padding  The padding sizes.
        @return     A new padding layer
     */
    auto layerWithSymmetricPadding(std::vector<uint32_t> const& padding) -> CppMLCPaddingLayer;

    /*! @abstract   Create a padding layer with zero padding
        @param      padding  The padding sizes.
        @return     A new padding layer
     */
    auto layerWithZeroPadding(std::vector<uint32_t> const& padding) -> CppMLCPaddingLayer;

    /*! @abstract   Create a padding layer with constant padding
        @param      padding                 The padding sizes.
        @param      constantValue   The constant value to pad the source tensor.
        @return     A new padding layer
     */
    auto layerWithConstantPadding(std::vector<uint32_t> const& padding,
                                  float constantValue) -> CppMLCPaddingLayer;
public:
    CppMLCPaddingLayer(void* self);
};
