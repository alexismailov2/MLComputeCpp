#pragma once

#include "CppMLCLayer.h"

class CppMLCPoolingDescriptor;

/*! @class      MLCPoolingLayer
    @abstract   A pooling layer
 */
class CppMLCPoolingLayer : public CppMLCLayer
{
public:
    /*! @property   descriptor
        @abstract   The pooling descriptor
     */
    auto descriptor() -> CppMLCPoolingDescriptor;

    /*! @abstract   Create a pooling layer
        @param      descriptor  The pooling descriptor
        @return     A new pooling layer
     */
    auto layerWithDescriptor(CppMLCPoolingDescriptor& descriptor) -> CppMLCPoolingLayer;

private:
    CppMLCPoolingLayer(void* self);
};
