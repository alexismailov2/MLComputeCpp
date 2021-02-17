#include "CppMLCTensorParameter.h"
#include "CppMLCTensor.h"

//TODO: Missed forward declaration in the MLCTensorData in the MLCompute/MLCTensorParameter.h
@class MLCTensorData;
#import <MLCompute/MLCTensorParameter.h>

auto CppMLCTensorParameter::tensor() -> CppMLCTensor {
    return CppMLCTensor{((MLCTensorParameter*)self).tensor};
}

bool CppMLCTensorParameter::isUpdatable() {
    return ((MLCTensorParameter*)self).isUpdatable == YES;
}

void CppMLCTensorParameter::isUpdateble(bool value) {
    ((MLCTensorParameter*)self).isUpdatable = value ? YES : NO;
}

CppMLCTensorParameter CppMLCTensorParameter::parameterWithTensor(const CppMLCTensor &tensor) {
    return CppMLCTensorParameter{[MLCTensorParameter parameterWithTensor:(MLCTensor*)(tensor.self)]};
}

CppMLCTensorParameter
CppMLCTensorParameter::parameterWithTensor(const CppMLCTensor &tensor, std::vector<CppMLCTensorData *> optimizerData) {
    return {nullptr};//CppMLCTensorParameter{[MLCTensorParameter parameterWithTensor:(MLCTensor*)(tensor.self) optimizerData:]};
}

CppMLCTensorParameter::CppMLCTensorParameter(void *self)
    : self{self}
{
}
