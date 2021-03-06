#pragma once

#include "CppMLCLossLayer.h"
#include "CppMLCYOLOLossDescriptor.h"

/*! @class      CppMLCYOLOLossLayer
    @abstract   A YOLO loss layer
 */
class CppMLCYOLOLossLayer : public CppMLCLossLayer
{
public:
    /*! @property   yoloLossDescriptor
        @abstract   The YOLO loss descriptor
     */
    auto yoloLossDescriptor() -> CppMLCYOLOLossDescriptor;

    /*! @abstract   Create a YOLO loss layer
     *  @param      lossDescriptor          The loss descriptor
     *  @return     A new YOLO loss layer.
     */
    auto layerWithDescriptor(CppMLCYOLOLossDescriptor& lossDescriptor) -> CppMLCYOLOLossLayer;

private:
    CppMLCYOLOLossLayer(void* self);
};

