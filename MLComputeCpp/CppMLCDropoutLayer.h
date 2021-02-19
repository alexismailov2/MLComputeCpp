#pragma once

#include "CppMLCLayer.h"

/*! @class      MLCDropoutLayer
    @abstract   A dropout layer
 */
class CppMLCDropoutLayer : public CppMLCLayer
{
public:
    /*! @property   rate
        @abstract   The probability that each element is dropped
     */
    auto rate() -> float;

    /*! @property   seed
        @abstract   The initial seed used to generate random numbers
     */
    auto seed() -> uint32_t;

    /*! @abstract   Create a dropout layer
        @param      rate  A scalar float value. The probability that each element is dropped.
        @param      seed  The seed used to generate random numbers.
        @return     A new dropout layer
     */
    static auto layerWithRate(float rate, uint32_t seed) -> CppMLCDropoutLayer;

private:
    CppMLCDropoutLayer(void* self);

private:
    void* self;
};


