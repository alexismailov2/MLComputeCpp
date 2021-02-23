#pragma once

#include "CppMLCTypes.h"

#include <cstdint>
#include <string>

class CppMLCDevice;
class CppMLCTensor;
class CppMLCTypesPrivate;
class CppMLCGraph;

/*! @class      CppMLCLayer
    @abstract   The base class for all MLCompute layers
    @discussion There are as many MLCLayer subclasses as there are MLCompute neural network layer objects. Make one of those.
                This class defines an polymorphic interface for them.
 */
class CppMLCLayer
{
public:
    /*! @property   layerID
        @abstract   The layer ID
        @discussion A unique number to identify each layer.  Assigned when the layer is created.
     */
    auto getLayerID() -> uint64_t;

    /*! @property   label
        @abstract   A string to help identify this object.
     */
    auto getLabel() -> std::string;

    /*! @property   isDebuggingEnabled
        @abstract   A flag to identify if we want to debug this layer when executing a graph that includes this layer
        @discussion If this is set, we will make sure that the result tensor and gradient tensors are available for reading on CPU
                    The default is NO.  If isDebuggingEnabled is set to YES,  make sure to set options to enable debugging when
                    compiling the graph.  Otherwise this property may be ignored.
     */
    bool isDebuggingEnabled();

    /*! @abstract   Determine whether instances of this layer accept source tensors of the given data type on the given device.
        @param      dataType   A data type of a possible input tensor to the layer
        @param      device     A device
        @return     A boolean indicating whether the data type is supported
     */
    static bool supportsDataType(eMLCDataType dataType, CppMLCDevice const& device);

    auto getSelf() -> void*;

protected:
    CppMLCLayer(void* self);

private:
    void* self;
    friend CppMLCTypesPrivate;
    friend CppMLCGraph;
};