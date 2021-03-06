#pragma once

#import "CppMLCLayer.h"

#include <vector>

/*! @class      CppMLCUpsampleLayer
    @abstract   An upsample layer
 */
class CppMLCUpsampleLayer : public CppMLCLayer
{
public:
    /*! @property   shape
        @abstract   A NSArray<NSNumber *> representing just the width if number of entries in shape array is 1 or
                    the height followed by width of result tensor if the number of entries in shape array is 2.
     */
    auto shape() -> std::vector<uint32_t>;

    /*! @property   sampleMode
        @abstract   The sampling mode to use when performing the upsample.
     */
    auto sampleMode() -> eMLCSampleMode;

    /*! @property   alignsCorners
        @abstract   A boolean that specifies whether the corner pixels of the source and result tensors are aligned.
        @discussion If True, the corner pixels of the source and result tensors are aligned, and thus preserving the values at those pixels.
                    This only has effect when mode is 'bilinear'. Default is NO.
     */
    bool alignsCorners();

    /*! @abstract   Create an upsample layer
        @param      shape                   A NSArray<NSNumber *> representing the dimensions of the result tensor
        @return     A new upsample layer.
    */
    auto layerWithShape(std::vector<uint32_t> shape) -> CppMLCUpsampleLayer;

    /*! @abstract   Create an upsample layer
     *  @param      shape                   A NSArray<NSNumber *> representing the dimensions of the result tensor
     *  @param      sampleMode        The upsampling algorithm to use.  Default is nearest.
     *  @param      alignsCorners    Whether the corner pixels of the input and output tensors are aligned or not.
     *  @return     A new upsample layer.
     */
    auto layerWithShape(std::vector<uint32_t> shape,
                        eMLCSampleMode sampleMode,
                        bool alignsCorners) -> CppMLCUpsampleLayer;
private:
    CppMLCUpsampleLayer(void* self);
};