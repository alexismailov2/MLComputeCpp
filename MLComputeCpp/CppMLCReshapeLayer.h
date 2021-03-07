#pragma once

#include "CppMLCLayer.h"

/*! @class      MLCReshapeLayer
    @abstract   A reshape layer
 */
class CppMLCReshapeLayer : public CppMLCLayer
{
public:
    /*! @abstract   Create a reshape layer
     *  @param      shape NSArray<NSNumber *> representing the shape of result tensor
     *  @return     A new reshape layer.
     */
    auto layerWithShape(std::vector<uint32_t> const& shape) -> CppMLCReshapeLayer;

private:
    CppMLCReshapeLayer(void* self);
};
