#pragma once

#include "CppMLCTypes.h"
#include "CppMLCLayer.h"

/*! @class      CppMLCSoftmaxLayer
    @abstract   A softmax layer
 */
class CppMLCSoftmaxLayer : public CppMLCLayer
{
public:
    /*! @property   operation
        @abstract   The softmax operation.  Supported values are softmax and log softmax.
     */
    auto operation() -> eMLCSoftmaxOperation;;

    /*! @property   dimension
        @abstract   The  dimension over which softmax operation should be performed
     */
    auto dimension() -> uint64_t;;

    /*! @abstract   Create a softmax layer
        @param      operation  The softmax operation
        @return     A new softmax layer
     */
    auto layerWithOperation(eMLCSoftmaxOperation operation) -> CppMLCSoftmaxLayer;;

    /*! @abstract   Create a softmax layer
        @param      operation  The softmax operation
        @param      dimension  The  dimension over which softmax operation should be performed
        @return     A new softmax layer
     */
    auto layerWithOperation(eMLCSoftmaxOperation operation, uint64_t dimension) -> CppMLCSoftmaxLayer;

private:
    CppMLCSoftmaxLayer(void* self);
};