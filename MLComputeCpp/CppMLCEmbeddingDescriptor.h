#pragma once

#include <cstdint>

class CppMLCEmbeddingLayer;

/*! @class      MLCEmbeddingDescriptor
    @discussion The MLCEmbeddingDescriptor specifies an embedding layer descriptor
 */
class CppMLCEmbeddingDescriptor
{
public:
    /*! @property   embeddingCount
     *  @abstract   The size of the dictionary
     */
    auto embeddingCount() -> uint32_t;

    /*! @property   embeddingDimension
     *  @abstract   The dimension of embedding vectors
     */
    auto embeddingDimension() -> uint32_t;

    /*! @property   paddingIndex
     *  @abstract   If set, the embedding vector at paddingIndex is initialized with zero and will not be updated in gradient pass, Default=nil
     */
    auto paddingIndex() -> uint32_t;

    /*! @property   maximumNorm
     *  @abstract   A float, if set, in the forward pass only, the selected embedding vectors will be re-normalized to have an Lp norm of less than maximumNorm in the dictionary, Default=nil
     */
    auto maximumNorm() -> float;

    /*! @property   pNorm
     *  @abstract   A float, the p of the Lp norm, can be set to infinity norm by [NSNumber numberWithFloat:INFINITY]. Default=2.0
     */
    auto pNorm() -> float;

    /*! @property   scalesGradientByFrequency
     *  @abstract   If set, the gradients are scaled by the inverse of the frequency of the words in batch before the weight update. Default=NO
     */
    bool scalesGradientByFrequency();

    static auto descriptorWithEmbeddingCount(uint32_t embeddingCount,
                                             uint32_t embeddingDimension) -> CppMLCEmbeddingDescriptor;

    static auto descriptorWithEmbeddingCount(uint32_t embeddingCount,
                                             uint32_t embeddingDimension,
                                             uint32_t paddingIndex,
                                             uint32_t maximumNorm,
                                             uint32_t pNorm,
                                             bool scalesGradientByFrequency) -> CppMLCEmbeddingDescriptor;

private:
    CppMLCEmbeddingDescriptor(void* self);

private:
    void* self;
    friend CppMLCEmbeddingLayer;
};

