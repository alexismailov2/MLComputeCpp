#pragma once

#include <vector>

class CppMLCTensor;
class CppMLCTensorData;
class CppMLCBatchNormalizationLayer;
class CppMLCConvolutionLayer;

/*! @class      MLCTensorParameter
    @discussion A tensor parameter object.  This is used to describe input tensors that are updated by the optimizer during training.
 */
class CppMLCTensorParameter
{
public:
/*! @property   tensor
    @abstract   The underlying tensor
 */
auto tensor() -> CppMLCTensor;

/*! @property   isUpdatable
    @abstract   Specifies whether this tensor parameter is updatable
 */
bool isUpdatable();
void isUpdateble(bool value);

/*! @abstract   Create a tensor parameter
    @param      tensor            The unedrlying tensor
    @return     A new tensor parameter object
 */
static CppMLCTensorParameter parameterWithTensor(CppMLCTensor const& tensor);

/*! @abstract   Create a tensor parameter
    @param      tensor            The unedrlying tensor
    @param      optimizerData   The optimizer data needed for this input tensor
    @return     A new tensor parameter object
 */
static CppMLCTensorParameter parameterWithTensor(CppMLCTensor const& tensor, std::vector<CppMLCTensorData*> optimizerData);

private:
    CppMLCTensorParameter(void* self);

private:
    void* self;
    friend CppMLCBatchNormalizationLayer;
    friend CppMLCConvolutionLayer;
};


