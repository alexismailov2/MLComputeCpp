#include "CppMLCGramMatrixLayer.h"

#import <MLCompute/MLCGramMatrixLayer.h>

auto CppMLCGramMatrixLayer::scale() -> float {
    return ((MLCGramMatrixLayer*)self).scale;
}

auto CppMLCGramMatrixLayer::layerWithScale(float scale) -> CppMLCGramMatrixLayer {
    return CppMLCGramMatrixLayer{[MLCGramMatrixLayer layerWithScale:scale]};
}

CppMLCGramMatrixLayer::CppMLCGramMatrixLayer(void *self)
        : CppMLCLayer{self}
        , self{self}
{
}
