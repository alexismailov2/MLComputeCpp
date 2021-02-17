#include "CppMLCArithmeticLayer.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCArithmeticLayer.h>

CppMLCArithmeticLayer::CppMLCArithmeticLayer(void *self)
    : CppMLCLayer(self)
    , self{self}
{
}

auto CppMLCArithmeticLayer::layerWithOperation(eMLCArithmeticOperation operation) -> CppMLCArithmeticLayer {
    return CppMLCArithmeticLayer{[MLCArithmeticLayer layerWithOperation:toNative(operation)]};
}

auto CppMLCArithmeticLayer::operation() -> eMLCArithmeticOperation {
    return MLCArithmeticOperationToCpp(((MLCArithmeticLayer *) self).operation);
}
