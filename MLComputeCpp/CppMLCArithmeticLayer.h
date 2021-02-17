#pragma once

#import "CppMLCLayer.h"

class CppMLCDevice;
class CppMLCTensor;

/*! @class      MLCArithmeticLayer
    @abstract   An arithmetic layer
 */
class CppMLCArithmeticLayer : public CppMLCLayer
{
public:
    /*! @property   operation
        @abstract   The arithmetic operation.
     */
    auto operation() -> eMLCArithmeticOperation;;

    /*! @abstract   Create an arithmetic layer
        @param      operation    The arithmetic operation
        @return     A new arithmetic layer
     */
    static auto layerWithOperation(eMLCArithmeticOperation operation) -> CppMLCArithmeticLayer;;

protected:
    CppMLCArithmeticLayer(void* self);

private:
    void* self;
};

